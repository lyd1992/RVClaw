"""Fixed policy for the supervised rear-blind recovery action."""


BACKUP_DISTANCE_M = 0.15
BACKUP_SPEED_M_S = 0.03
ACTION_TIME_ALLOWANCE_S = 5
CLIENT_RESULT_TIMEOUT_S = 7.0
SERVER_WAIT_TIMEOUT_S = 2.0
ARM_TOKEN = "SUPERVISED_RECENT_PATH_BACKUP_015M_003MPS"

# The authorization corridor is fixed in the odom frame. The scan filter does
# not clear this entire corridor: it only clears current-footprint returns,
# which are the only returns that can be attributed to the installed chassis.
# Half a costmap cell is included so a discretized boundary cannot re-mark the
# vehicle itself as an obstacle.
FOOTPRINT_FRONT_M = 0.16275
FOOTPRINT_REAR_M = 0.16440
FOOTPRINT_HALF_WIDTH_M = 0.13725
COSTMAP_HALF_CELL_MARGIN_M = 0.025
CORRIDOR_X_MIN_M = -(
    FOOTPRINT_REAR_M + BACKUP_DISTANCE_M + COSTMAP_HALF_CELL_MARGIN_M
)
CORRIDOR_X_MAX_M = FOOTPRINT_FRONT_M + COSTMAP_HALF_CELL_MARGIN_M
CORRIDOR_Y_MIN_M = -(
    FOOTPRINT_HALF_WIDTH_M + COSTMAP_HALF_CELL_MARGIN_M
)
CORRIDOR_Y_MAX_M = FOOTPRINT_HALF_WIDTH_M + COSTMAP_HALF_CELL_MARGIN_M
CORRIDOR_AUTHORIZATION_TTL_S = 10.0
CORRIDOR_SETTLE_S = 1.0
PREARM_COLLISION_PROBE_S = 0.6


def describe_plan():
    """Return the immutable recovery boundary for inspection and tests."""
    return {
        "action_name": "/backup",
        "distance_m": BACKUP_DISTANCE_M,
        "speed_m_s": BACKUP_SPEED_M_S,
        "action_time_allowance_s": ACTION_TIME_ALLOWANCE_S,
        "client_result_timeout_s": CLIENT_RESULT_TIMEOUT_S,
        "straight_only": True,
        "automatic_bt_recovery": False,
        "recent_path_assumption": True,
        "operator_supervision_required": True,
        "direct_velocity_publish": False,
        "fixed_odom_swept_corridor": True,
        "corridor_authorization_ttl_s": CORRIDOR_AUTHORIZATION_TTL_S,
        "prearm_collision_probe_s": PREARM_COLLISION_PROBE_S,
    }


def require_arm_token(token):
    """Reject accidental execution and return the fixed policy on success."""
    if token != ARM_TOKEN:
        raise ValueError("exact supervised bounded-backup token is required")
    return describe_plan()


def point_in_authorized_swept_corridor(x_m, y_m):
    """Return whether a point lies in the fixed authorized swept rectangle."""
    return (
        CORRIDOR_X_MIN_M <= x_m <= CORRIDOR_X_MAX_M
        and CORRIDOR_Y_MIN_M <= y_m <= CORRIDOR_Y_MAX_M
    )


def point_in_current_footprint(x_m, y_m):
    """Return whether a base-frame point is inside the padded vehicle body."""
    return (
        -FOOTPRINT_REAR_M - COSTMAP_HALF_CELL_MARGIN_M <= x_m
        <= FOOTPRINT_FRONT_M + COSTMAP_HALF_CELL_MARGIN_M
        and -FOOTPRINT_HALF_WIDTH_M - COSTMAP_HALF_CELL_MARGIN_M
        <= y_m
        <= FOOTPRINT_HALF_WIDTH_M + COSTMAP_HALF_CELL_MARGIN_M
    )


def filter_authorized_scan_ranges(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    laser_x,
    laser_y,
    laser_yaw,
    current_pose,
    authorized_start_pose,
):
    """Clear finite returns only inside the odom-fixed authorized corridor."""
    import math

    output = list(ranges)
    masked = 0
    current_x, current_y, current_yaw = current_pose
    start_x, start_y, start_yaw = authorized_start_pose
    current_cos = math.cos(current_yaw)
    current_sin = math.sin(current_yaw)
    start_cos = math.cos(start_yaw)
    start_sin = math.sin(start_yaw)
    laser_cos = math.cos(laser_yaw)
    laser_sin = math.sin(laser_yaw)

    for index, distance in enumerate(ranges):
        if (
            not math.isfinite(distance)
            or distance < range_min
            or distance > range_max
        ):
            continue
        angle = angle_min + index * angle_increment
        laser_point_x = distance * math.cos(angle)
        laser_point_y = distance * math.sin(angle)
        base_x = laser_x + laser_cos * laser_point_x - laser_sin * laser_point_y
        base_y = laser_y + laser_sin * laser_point_x + laser_cos * laser_point_y
        odom_x = current_x + current_cos * base_x - current_sin * base_y
        odom_y = current_y + current_sin * base_x + current_cos * base_y
        # Only remove returns physically inside the current chassis footprint.
        # Returns elsewhere, including the rest of the recently traversed path,
        # remain real obstacle evidence and are never blanket-cleared.
        if point_in_current_footprint(base_x, base_y):
            output[index] = range_max
            masked += 1
    return output, masked
