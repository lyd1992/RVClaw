"""Online SLAM plus frontier exploration composition.

All hardware output and exploration are opt-in launch arguments.  The launch
uses Nav2's navigation-only entry point so the live SLAM map remains the sole
map publisher; no map_server or AMCL is started in this mode.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.descriptions import ParameterFile
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    package_share = FindPackageShare("rvclaw_nav2_bringup")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    hardware_output = LaunchConfiguration("enable_hardware_output")
    start_exploration = LaunchConfiguration("start_exploration")
    arm_token = LaunchConfiguration("arm_token")
    max_forward_speed = LaunchConfiguration("max_forward_speed_m_s")
    start_rviz = LaunchConfiguration("start_rviz")
    params_file = PathJoinSubstitution([package_share, "config", "nav2_params.yaml"])
    forward_nav_to_pose_bt = PathJoinSubstitution(
        [package_share, "behavior_trees", "navigate_to_pose_forward_only.xml"]
    )
    forward_nav_through_poses_bt = PathJoinSubstitution(
        [package_share, "behavior_trees", "navigate_through_poses_forward_only.xml"]
    )
    rewritten_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites={
            "default_nav_to_pose_bt_xml": forward_nav_to_pose_bt,
            "default_nav_through_poses_bt_xml": forward_nav_through_poses_bt,
        },
        convert_types=True,
    )
    configured_params = ParameterFile(rewritten_params, allow_substs=True)
    laser_arguments = ["laser_x", "laser_y", "laser_z", "laser_roll", "laser_pitch", "laser_yaw"]

    declarations = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("autostart", default_value="false"),
        DeclareLaunchArgument("enable_hardware_output", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("start_exploration", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("arm_token", default_value=""),
        DeclareLaunchArgument("max_runtime_s", default_value="60.0"),
        DeclareLaunchArgument("max_goals", default_value="3"),
        DeclareLaunchArgument("max_exploration_radius_m", default_value="5.0"),
        DeclareLaunchArgument("min_frontier_cells", default_value="5"),
        DeclareLaunchArgument("goal_clearance_m", default_value="0.30"),
        DeclareLaunchArgument("scan_max_age_s", default_value="0.75"),
        DeclareLaunchArgument("map_max_age_s", default_value="2.0"),
        DeclareLaunchArgument("goal_timeout_s", default_value="30.0"),
        DeclareLaunchArgument("no_progress_timeout_s", default_value="12.0"),
        DeclareLaunchArgument("map_update_timeout_s", default_value="5.0"),
        DeclareLaunchArgument("max_forward_speed_m_s", default_value="0.06"),
        DeclareLaunchArgument("start_rviz", default_value="false", choices=["true", "false"]),
    ] + [DeclareLaunchArgument(name, description=f"Measured laser transform: {name}") for name in laser_arguments]

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([package_share, "launch", "mapping.launch.py"])),
        launch_arguments={
            "use_sim_time": use_sim_time,
            **{name: LaunchConfiguration(name) for name in laser_arguments},
        }.items(),
    )
    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("rvclaw_stm32_base"), "launch", "hardware.launch.py"])),
        launch_arguments={"enable_hardware_output": hardware_output}.items(),
    )
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "global_scan_lidar.launch.py"])
        )
    )
    nav2 = GroupAction([
        SetRemap(src="cmd_vel", dst="rvclaw/nav2_cmd_vel_raw"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"])),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "params_file": rewritten_params,
                "autostart": autostart,
                "use_composition": "False",
                "use_respawn": "false",
            }.items(),
        ),
    ])
    collision_monitor = Node(
        package="nav2_collision_monitor", executable="collision_monitor", name="collision_monitor",
        output="screen", parameters=[configured_params],
    )
    collision_lifecycle = Node(
        package="nav2_lifecycle_manager", executable="lifecycle_manager",
        name="lifecycle_manager_collision_monitor", output="screen",
        parameters=[{"use_sim_time": use_sim_time}, {"autostart": autostart}, {"node_names": ["collision_monitor"]}],
    )
    adapter = Node(
        package="rvclaw_nav2_bringup", executable="cmd_vel_adapter",
        name="rvclaw_nav2_cmd_adapter", output="screen",
        parameters=[{"max_forward_speed_m_s": max_forward_speed}],
    )
    explorer = Node(
        package="rvclaw_nav2_bringup", executable="rvclaw-frontier-explorer",
        name="rvclaw_frontier_explorer", output="screen",
        condition=IfCondition(start_exploration),
        parameters=[{
            "start_exploration": start_exploration,
            "arm_token": arm_token,
            "max_runtime_s": ParameterValue(LaunchConfiguration("max_runtime_s"), value_type=float),
            "max_goals": ParameterValue(LaunchConfiguration("max_goals"), value_type=int),
            "max_exploration_radius_m": ParameterValue(LaunchConfiguration("max_exploration_radius_m"), value_type=float),
            "min_frontier_cells": ParameterValue(LaunchConfiguration("min_frontier_cells"), value_type=int),
            "goal_clearance_m": ParameterValue(LaunchConfiguration("goal_clearance_m"), value_type=float),
            "scan_max_age_s": ParameterValue(LaunchConfiguration("scan_max_age_s"), value_type=float),
            "map_max_age_s": ParameterValue(LaunchConfiguration("map_max_age_s"), value_type=float),
            "goal_timeout_s": ParameterValue(LaunchConfiguration("goal_timeout_s"), value_type=float),
            "no_progress_timeout_s": ParameterValue(LaunchConfiguration("no_progress_timeout_s"), value_type=float),
            "map_update_timeout_s": ParameterValue(LaunchConfiguration("map_update_timeout_s"), value_type=float),
        }],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rvclaw_rviz", output="screen",
        condition=IfCondition(start_rviz),
        arguments=["-d", PathJoinSubstitution([package_share, "rviz", "rvclaw_navigation.rviz"])],
        parameters=[{"use_sim_time": use_sim_time}],
    )
    return LaunchDescription(declarations + [hardware, lidar, slam, nav2, adapter, collision_monitor, collision_lifecycle, explorer, rviz])
