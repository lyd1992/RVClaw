from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    device = LaunchConfiguration("device")
    config = PathJoinSubstitution(
        [FindPackageShare("rvclaw_stm32_base"), "config", "v550_4wd.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value=(
                    "/dev/serial/by-id/"
                    "usb-WCH.CN_USB_Single_Serial_0002-if00"
                ),
                description="Stable STM32 serial device path",
            ),
            Node(
                package="rvclaw_stm32_base",
                executable="telemetry_node",
                name="rvclaw_stm32_telemetry",
                output="screen",
                parameters=[config, {"device": device}],
            ),
        ]
    )
