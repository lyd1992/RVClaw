from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    default_config = PathJoinSubstitution(
        [FindPackageShare("rvclaw_nav2_bringup"), "rviz", "rvclaw_navigation.rviz"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_config,
                description="RViz display-only configuration",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rvclaw_rviz",
                output="screen",
                arguments=["-d", config],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )
