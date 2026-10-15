from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    device = LaunchConfiguration("device")
    enabled = LaunchConfiguration("enable_hardware_output")
    config = PathJoinSubstitution(
        [FindPackageShare("rvclaw_stm32_base"), "config", "hardware.yaml"]
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
            DeclareLaunchArgument(
                "enable_hardware_output",
                default_value="false",
                choices=["true", "false"],
                description="Explicitly attach the O_RDWR hardware transport",
            ),
            Node(
                package="rvclaw_stm32_base",
                executable="hardware_node",
                name="rvclaw_stm32_hardware",
                output="screen",
                parameters=[
                    config,
                    {
                        "device": device,
                        "hardware_output_enabled": ParameterValue(
                            enabled,
                            value_type=bool,
                        ),
                    },
                ],
            ),
        ]
    )
