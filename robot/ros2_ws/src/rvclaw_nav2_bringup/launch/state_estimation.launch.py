from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    config = PathJoinSubstitution(
        [FindPackageShare("rvclaw_nav2_bringup"), "config", "ekf_odom_only.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="rvclaw_ekf",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
            )
        ]
    )
