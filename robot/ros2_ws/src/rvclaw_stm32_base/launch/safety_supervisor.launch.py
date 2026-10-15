from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [
            FindPackageShare("rvclaw_stm32_base"),
            "config",
            "safety_supervisor.yaml",
        ]
    )

    return LaunchDescription(
        [
            Node(
                package="rvclaw_stm32_base",
                executable="safety_supervisor_node",
                name="rvclaw_safety_supervisor",
                output="screen",
                parameters=[config],
            )
        ]
    )
