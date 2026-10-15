from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.descriptions import ParameterFile
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    package_share = FindPackageShare("rvclaw_nav2_bringup")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    start_safety_supervisor = LaunchConfiguration("start_safety_supervisor")
    map_file = LaunchConfiguration("map")
    nav2_params = LaunchConfiguration("params_file")
    default_nav2_params = PathJoinSubstitution(
        [package_share, "config", "nav2_params.yaml"]
    )
    forward_nav_to_pose_bt = PathJoinSubstitution(
        [package_share, "behavior_trees", "navigate_to_pose_forward_only.xml"]
    )
    forward_nav_through_poses_bt = PathJoinSubstitution(
        [
            package_share,
            "behavior_trees",
            "navigate_through_poses_forward_only.xml",
        ]
    )
    rewritten_nav2_params = RewrittenYaml(
        source_file=nav2_params,
        param_rewrites={
            "default_nav_to_pose_bt_xml": forward_nav_to_pose_bt,
            "default_nav_through_poses_bt_xml": forward_nav_through_poses_bt,
        },
        convert_types=True,
    )
    configured_nav2_params = ParameterFile(
        rewritten_nav2_params,
        allow_substs=True,
    )
    laser_arguments = [
        "laser_x",
        "laser_y",
        "laser_z",
        "laser_roll",
        "laser_pitch",
        "laser_yaw",
    ]

    declarations = [
        DeclareLaunchArgument(
            "map",
            description="Absolute path to the reviewed map YAML",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "autostart",
            default_value="false",
            description="Keep false until the supervised navigation acceptance stage",
        ),
        DeclareLaunchArgument(
            "start_safety_supervisor",
            default_value="true",
            choices=["true", "false"],
            description=(
                "Start the preview safety supervisor only when no hardware "
                "node already owns the control services"
            ),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_nav2_params,
            description="Nav2 parameter file; defaults to the reviewed profile",
        ),
    ] + [
        DeclareLaunchArgument(name, description=f"Measured laser transform: {name}")
        for name in laser_arguments
    ]

    sensor_transforms = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "sensor_transforms.launch.py"])
        ),
        launch_arguments={name: LaunchConfiguration(name) for name in laser_arguments}.items(),
    )
    odom_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "odom_tf.launch.py"])
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )
    safety_supervisor = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("rvclaw_stm32_base"), "launch", "safety_supervisor.launch.py"]
            )
        ),
        condition=IfCondition(start_safety_supervisor),
    )
    cmd_adapter = Node(
        package="rvclaw_nav2_bringup",
        executable="cmd_vel_adapter",
        name="rvclaw_nav2_cmd_adapter",
        output="screen",
    )
    collision_monitor = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[configured_nav2_params],
    )
    collision_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_collision_monitor",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": autostart},
            {"node_names": ["collision_monitor"]},
        ],
    )
    nav2 = GroupAction(
        [
            SetRemap(src="cmd_vel", dst="rvclaw/nav2_cmd_vel_raw"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"]
                    )
                ),
                launch_arguments={
                    "map": map_file,
                    "use_sim_time": use_sim_time,
                    "params_file": rewritten_nav2_params,
                    "autostart": autostart,
                    "use_composition": "False",
                    "use_respawn": "false",
                }.items(),
            ),
        ]
    )

    return LaunchDescription(
        declarations
        + [
            sensor_transforms,
            odom_tf,
            safety_supervisor,
            cmd_adapter,
            collision_monitor,
            collision_lifecycle,
            nav2,
        ]
    )
