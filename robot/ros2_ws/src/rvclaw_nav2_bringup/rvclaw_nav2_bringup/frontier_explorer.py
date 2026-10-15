"""ROS 2 frontier explorer for supervised online SLAM.

The node never publishes a velocity command.  It selects one safe frontier at
a time and sends ``NavigateToPose`` through the existing Nav2, Collision
Monitor, and STM32 safety chain.  Any stale input, failed goal, timeout, or
lack of progress cancels the goal and disarms the controller.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Optional

from .frontier_exploration_policy import ExplorationLimits, choose_frontier


EXPLORATION_ARM_TOKEN = "WHEELS_GROUND_AUTONOMOUS_MAPPING_LOW_SPEED"


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def main(args=None) -> None:
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool, String
    from std_srvs.srv import SetBool
    from tf2_ros import Buffer, TransformException, TransformListener

    class FrontierExplorer(Node):
        def __init__(self) -> None:
            super().__init__("rvclaw_frontier_explorer")
            self.declare_parameter("start_exploration", False)
            self.declare_parameter("arm_token", "")
            self.declare_parameter("max_runtime_s", 60.0)
            self.declare_parameter("max_goals", 3)
            self.declare_parameter("max_exploration_radius_m", 5.0)
            self.declare_parameter("min_frontier_cells", 5)
            self.declare_parameter("goal_clearance_m", 0.30)
            self.declare_parameter("scan_max_age_s", 0.75)
            self.declare_parameter("map_max_age_s", 2.0)
            self.declare_parameter("goal_timeout_s", 30.0)
            self.declare_parameter("no_progress_timeout_s", 12.0)
            self.declare_parameter("map_update_timeout_s", 5.0)

            self.start_exploration = bool(self.get_parameter("start_exploration").value)
            self.arm_token = str(self.get_parameter("arm_token").value)
            self.limits = ExplorationLimits(
                max_runtime_s=float(self.get_parameter("max_runtime_s").value),
                max_goals=int(self.get_parameter("max_goals").value),
                max_exploration_radius_m=float(self.get_parameter("max_exploration_radius_m").value),
                min_frontier_cells=int(self.get_parameter("min_frontier_cells").value),
                goal_clearance_m=float(self.get_parameter("goal_clearance_m").value),
            )
            self.limits.validate()
            self.scan_max_age_s = float(self.get_parameter("scan_max_age_s").value)
            self.map_max_age_s = float(self.get_parameter("map_max_age_s").value)
            self.goal_timeout_s = float(self.get_parameter("goal_timeout_s").value)
            self.no_progress_timeout_s = float(self.get_parameter("no_progress_timeout_s").value)
            self.map_update_timeout_s = float(self.get_parameter("map_update_timeout_s").value)
            if self.arm_token and self.arm_token != EXPLORATION_ARM_TOKEN:
                raise ValueError("invalid autonomous mapping arm token")

            sensor_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            state_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.state_pub = self.create_publisher(String, "/rvclaw/exploration/state", state_qos)
            self.goal_pub = self.create_publisher(PoseStamped, "/rvclaw/exploration/current_goal", 10)
            self.estop_pub = self.create_publisher(Bool, "/rvclaw/control/emergency_stop", 10)
            self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._on_map, 10)
            self.scan_sub = self.create_subscription(LaserScan, "/scan", self._on_scan, sensor_qos)
            self.pose_sub = self.create_subscription(
                PoseWithCovarianceStamped, "/amcl_pose", self._on_pose, 10
            )
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.action_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
            self.disarm_client = self.create_client(SetBool, "/rvclaw/control/set_armed")
            self.map_message = None
            self.scan_message = None
            self.pose_message = None
            self.map_time = 0.0
            self.scan_time = 0.0
            self.pose_time = 0.0
            self.started_at = time.monotonic()
            self.goal_handle = None
            self.goal_future = None
            self.result_future = None
            self.goal_started_at = 0.0
            self.goal_completed_at = 0.0
            self.current_candidate = None
            self.best_distance = math.inf
            self.last_progress_at = time.monotonic()
            self.goals_sent = 0
            self.failed_goal_keys = set()
            self.completed_goal_keys = set()
            self.last_map_stamp = None
            self.state = "IDLE" if not self.start_exploration else "WAITING_FOR_INPUTS"
            self.deadman_pub = self.create_publisher(Bool, "/rvclaw/control/deadman", 10)
            self._publish_state(self.state)
            self.timer = self.create_timer(0.5, self._tick)
            self.deadman_timer = self.create_timer(0.05, self._publish_deadman)

        def _publish_state(self, value: str) -> None:
            self.state = value
            message = String()
            message.data = value
            self.state_pub.publish(message)

        def _publish_deadman(self) -> None:
            message = Bool()
            message.data = self.start_exploration and self.state not in ("STOPPED", "STOPPING")
            self.deadman_pub.publish(message)

        def _on_map(self, message) -> None:
            self.map_message = message
            self.map_time = time.monotonic()

        def _on_scan(self, message) -> None:
            self.scan_message = message
            self.scan_time = time.monotonic()

        def _on_pose(self, message) -> None:
            self.pose_message = message
            self.pose_time = time.monotonic()

        def _grid_from_message(self, message):
            width = int(message.info.width)
            height = int(message.info.height)
            data = list(message.data)
            if width <= 0 or height <= 0 or len(data) != width * height:
                raise ValueError("invalid OccupancyGrid dimensions")
            cells = [data[row * width : (row + 1) * width] for row in range(height)]
            return SimpleNamespace(
                cells=cells,
                resolution=float(message.info.resolution),
                origin_x=float(message.info.origin.position.x),
                origin_y=float(message.info.origin.position.y),
                origin_yaw=_yaw_from_quaternion(message.info.origin.orientation),
                width=width,
                height=height,
            )

        def _pose_tuple(self):
            if self.pose_message is not None:
                pose = self.pose_message.pose.pose
                return (float(pose.position.x), float(pose.position.y), _yaw_from_quaternion(pose.orientation))
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map", "base_footprint", rclpy.time.Time()
                ).transform
            except TransformException:
                return None
            return (
                float(transform.translation.x),
                float(transform.translation.y),
                _yaw_from_quaternion(transform.rotation),
            )

        def _inputs_ready(self) -> bool:
            now = time.monotonic()
            if self.map_message is None or now - self.map_time > self.map_max_age_s:
                return False
            if self.scan_message is None or now - self.scan_time > self.scan_max_age_s:
                return False
            if self.pose_message is not None and now - self.pose_time <= self.map_max_age_s:
                return True
            return self._pose_tuple() is not None

        def _goal_key(self, candidate) -> tuple:
            return (round(candidate.x, 2), round(candidate.y, 2))

        def _tick(self) -> None:
            if not self.start_exploration:
                return
            if self.arm_token != EXPLORATION_ARM_TOKEN:
                self._stop("ARM_TOKEN_REQUIRED")
                return
            if time.monotonic() - self.started_at > self.limits.max_runtime_s:
                self._stop("MAX_RUNTIME_EXCEEDED")
                return
            if self.goal_handle is not None:
                now = time.monotonic()
                if now - self.scan_time > self.scan_max_age_s:
                    self._cancel_and_stop("STALE_SCAN_DURING_GOAL")
                elif now - self.map_time > self.map_max_age_s:
                    self._cancel_and_stop("STALE_MAP_DURING_GOAL")
                elif now - self.goal_started_at > self.goal_timeout_s:
                    self._cancel_and_stop("GOAL_TIMEOUT")
                elif now - self.last_progress_at > self.no_progress_timeout_s:
                    self._cancel_and_stop("NO_PROGRESS")
                return
            if self.state == "WAITING_FOR_MAP_UPDATE":
                if (
                    self.map_time <= (self.last_map_stamp or 0.0)
                    and time.monotonic() - self.goal_completed_at < self.map_update_timeout_s
                ):
                    return
                self._publish_state("READY_FOR_FRONTIER")
            if self.goals_sent >= self.limits.max_goals:
                self._stop("MAX_GOALS_REACHED")
                return
            if not self._inputs_ready():
                self._publish_state("WAITING_FOR_INPUTS")
                return
            try:
                pose = self._pose_tuple()
                if pose is None:
                    self._publish_state("WAITING_FOR_TF")
                    return
                decision = choose_frontier(self._grid_from_message(self.map_message), pose, self.limits)
            except (TypeError, ValueError) as exc:
                self._stop("INVALID_MAP:%s" % exc)
                return
            candidates = [
                item
                for item in decision.candidates
                if self._goal_key(item) not in self.failed_goal_keys
                and self._goal_key(item) not in self.completed_goal_keys
            ]
            if not candidates:
                self._stop("NO_SAFE_FRONTIER")
                return
            self._send_goal(candidates[0])

        def _send_goal(self, candidate) -> None:
            if not self.action_client.wait_for_server(timeout_sec=0.5):
                self._stop("NAV2_ACTION_UNAVAILABLE")
                return
            message = PoseStamped()
            message.header.frame_id = "map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.pose.position.x = candidate.x
            message.pose.position.y = candidate.y
            message.pose.orientation.z = math.sin(candidate.yaw / 2.0)
            message.pose.orientation.w = math.cos(candidate.yaw / 2.0)
            goal = NavigateToPose.Goal()
            goal.pose = message
            self.goal_pub.publish(message)
            self.current_candidate = candidate
            self._publish_state("SENDING_GOAL")
            self.goal_future = self.action_client.send_goal_async(goal, feedback_callback=self._on_feedback)
            self.goal_future.add_done_callback(self._on_goal_response)
            self.goal_started_at = time.monotonic()
            self.best_distance = math.inf
            self.last_progress_at = self.goal_started_at

        def _on_goal_response(self, future) -> None:
            self.goal_handle = future.result()
            if self.goal_handle is None or not self.goal_handle.accepted:
                if self.current_candidate is not None:
                    self.failed_goal_keys.add(self._goal_key(self.current_candidate))
                self.goal_handle = None
                self._publish_state("GOAL_REJECTED_RETRY")
                return
            self.goals_sent += 1
            self._publish_state("EXPLORING")
            self.result_future = self.goal_handle.get_result_async()
            self.result_future.add_done_callback(self._on_result)

        def _on_feedback(self, feedback_message) -> None:
            feedback = feedback_message.feedback
            distance = getattr(feedback, "distance_remaining", None)
            if distance is None:
                return
            distance = float(distance)
            if math.isfinite(distance) and distance + 0.01 < self.best_distance:
                self.best_distance = distance
                self.last_progress_at = time.monotonic()

        def _on_result(self, future) -> None:
            result = future.result()
            status = int(result.status)
            self.goal_handle = None
            self.result_future = None
            if status == GoalStatus.STATUS_SUCCEEDED:
                if self.current_candidate is not None:
                    self.completed_goal_keys.add(self._goal_key(self.current_candidate))
                self._publish_state("WAITING_FOR_MAP_UPDATE")
                self.last_map_stamp = self.map_time
                self.goal_completed_at = time.monotonic()
                return
            if self.current_candidate is not None:
                self.failed_goal_keys.add(self._goal_key(self.current_candidate))
            self._publish_state("NAV2_GOAL_FAILED_RETRY:%d" % status)

        def _cancel_and_stop(self, reason: str) -> None:
            if self.goal_handle is not None:
                future = self.goal_handle.cancel_goal_async()
                future.add_done_callback(lambda _future: self._stop(reason))
            else:
                self._stop(reason)

        def _stop(self, reason: str) -> None:
            if self.state == "STOPPED":
                return
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
                self.goal_handle = None
            self._publish_state("STOPPING:%s" % reason)
            if self.disarm_client.service_is_ready():
                request = SetBool.Request()
                request.data = False
                self.disarm_client.call_async(request)
            estop = Bool()
            estop.data = True
            self.estop_pub.publish(estop)
            self._publish_state("STOPPED")

    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._stop("INTERRUPTED")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
