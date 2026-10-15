from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    heading_params_file = LaunchConfiguration("heading_params_file")
    default_params = PathJoinSubstitution(
        [FindPackageShare("rvclaw_nav2_bringup"), "config", "nav2_params.yaml"]
    )
    default_heading_params = PathJoinSubstitution(
        [
            FindPackageShare("rvclaw_nav2_bringup"),
            "config",
            "heading_gate_params.yaml",
        ]
    )
    tf_remaps = [("/tf", "tf"), ("/tf_static", "tf_static")]

    local_costmap = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        output="screen",
        parameters=[heading_params_file, {"use_sim_time": use_sim_time}],
        remappings=tf_remaps,
    )
    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[
            params_file,
            heading_params_file,
            {"use_sim_time": use_sim_time},
        ],
        remappings=tf_remaps + [("cmd_vel", "rvclaw/nav2_cmd_vel")],
    )
    command_adapter = Node(
        package="rvclaw_nav2_bringup",
        executable="cmd_vel_adapter",
        name="rvclaw_nav2_cmd_adapter",
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Reviewed RVClaw Nav2 parameter file",
            ),
            DeclareLaunchArgument(
                "heading_params_file",
                default_value=default_heading_params,
                description="Standalone costmap and heading-gate overrides",
            ),
            local_costmap,
            behavior_server,
            command_adapter,
        ]
    )
