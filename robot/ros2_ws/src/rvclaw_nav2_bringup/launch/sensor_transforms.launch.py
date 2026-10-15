from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "laser_x",
            description="Measured base_footprint to laser X in metres",
        ),
        DeclareLaunchArgument(
            "laser_y",
            description="Measured base_footprint to laser Y in metres",
        ),
        DeclareLaunchArgument(
            "laser_z",
            description="Measured base_footprint to laser Z in metres",
        ),
        DeclareLaunchArgument(
            "laser_roll",
            description="Measured base_footprint to laser roll in radians",
        ),
        DeclareLaunchArgument(
            "laser_pitch",
            description="Measured base_footprint to laser pitch in radians",
        ),
        DeclareLaunchArgument(
            "laser_yaw",
            description="Measured base_footprint to laser yaw in radians",
        ),
    ]

    transform = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="rvclaw_base_to_laser",
        output="screen",
        arguments=[
            "--x",
            LaunchConfiguration("laser_x"),
            "--y",
            LaunchConfiguration("laser_y"),
            "--z",
            LaunchConfiguration("laser_z"),
            "--roll",
            LaunchConfiguration("laser_roll"),
            "--pitch",
            LaunchConfiguration("laser_pitch"),
            "--yaw",
            LaunchConfiguration("laser_yaw"),
            "--frame-id",
            "base_footprint",
            "--child-frame-id",
            "laser",
        ],
    )

    return LaunchDescription(arguments + [transform])
