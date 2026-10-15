#!/usr/bin/env python3
"""Read /map once and verify that the live map server serves the selected site.

No publishers, services, navigation goals, or motor commands are created.
"""

import argparse
import json
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rvclaw_nav2_bringup.runtime_map_contract import compare_live_map


def wait_for_map_publisher(node, deadline):
    """Wait for discovery within the existing deadline; reject real conflicts."""
    publishers = []
    stable_since = None
    identity = None
    while time.monotonic() < deadline:
        publishers = node.get_publishers_info_by_topic('/map')
        if publishers:
            if len(publishers) != 1 or publishers[0].node_name != 'map_server':
                names = [p.node_name for p in publishers]
                raise RuntimeError('live /map must have one map_server publisher; observed %s' % names)
            current = bytes(publishers[0].endpoint_gid)
            if identity != current:
                identity, stable_since = current, time.monotonic()
            elif time.monotonic() - stable_since >= .3:
                return publishers
        else:
            identity, stable_since = None, None
        rclpy.spin_once(node, timeout_sec=min(.1, max(0.,deadline-time.monotonic())))
    raise RuntimeError('live /map publisher discovery timed out; observed %d publishers' % len(publishers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-file", required=True)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    args = parser.parse_args()
    if not 0.1 <= args.timeout_s <= 30.0:
        parser.error("timeout must be between 0.1 and 30 seconds")

    rclpy.init()
    node = rclpy.create_node("rvclaw_readonly_map_identity_gate", enable_rosout=False)
    received = []
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    subscription = node.create_subscription(
        OccupancyGrid, "/map", received.append, qos,
    )
    try:
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline and not received:
            rclpy.spin_once(node, timeout_sec=min(0.2, max(0.0, deadline - time.monotonic())))
        if not received:
            raise RuntimeError("no live /map sample before the timeout")
        # The only publisher must be the map server, not an unrelated map or
        # simulated publisher. Do not accept a transient-local sample alone.
        # DDS may deliver the latched map before graph discovery lists its
        # publisher. Give the graph a short bounded chance to catch up while
        # still refusing zero, multiple, or wrong publishers at the gate.
        publishers = wait_for_map_publisher(node, deadline)
        result = compare_live_map(args.map_file, received[-1], len(publishers))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
