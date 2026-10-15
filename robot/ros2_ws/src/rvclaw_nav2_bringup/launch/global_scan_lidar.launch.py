from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("lslidar_driver"),
                    "launch",
                    "lsm10p_uart_launch.py",
                ]
            )
        )
    )
    remapped_lidar = GroupAction(
        [
            SetRemap(src="/scan", dst="/scan_raw"),
            lidar_launch,
        ]
    )
    scan_filter = Node(
        package="rvclaw_nav2_bringup",
        executable="rvclaw-scan-self-filter",
        name="rvclaw_scan_self_filter",
        output="screen",
        parameters=[
            {
                "input_topic": "/scan_raw",
                "output_topic": "/scan",
                "localization_output_topic": "/scan_localization",
                "full_output_topic": "/scan_full",
                "localization_half_angle_deg": ParameterValue(
                    EnvironmentVariable("RVCLAW_LOCALIZATION_SCAN_HALF_ANGLE_DEG", default_value="180.0"),
                    value_type=float,
                ),
                "localization_min_range_m": ParameterValue(
                    EnvironmentVariable("RVCLAW_LOCALIZATION_MIN_RANGE_M", default_value="0.0"),
                    value_type=float,
                ),
                "forward_half_angle_deg": ParameterValue(
                    EnvironmentVariable("RVCLAW_FRONT_SCAN_HALF_ANGLE_DEG", default_value="180.0"),
                    value_type=float,
                ),
                "laser_x_m": 0.07725,
                "laser_y_m": 0.0,
                "laser_yaw_deg": 0.0,
                "body_front_m": 0.13275,
                "body_rear_m": 0.13440,
                "body_left_m": 0.10725,
                "body_right_m": 0.10725,
                "body_padding_m": 0.005,
            }
        ],
    )
    return LaunchDescription([remapped_lidar, scan_filter])
