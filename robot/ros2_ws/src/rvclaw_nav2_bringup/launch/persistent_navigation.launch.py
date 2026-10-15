from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("rvclaw_nav2_bringup")
    map_file = LaunchConfiguration("map")
    cache_file = LaunchConfiguration("cache_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    start_hardware = LaunchConfiguration("start_hardware")
    start_lidar = LaunchConfiguration("start_lidar")
    start_rviz = LaunchConfiguration("start_rviz")
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
            description="Absolute path to the reviewed static map YAML",
        ),
        DeclareLaunchArgument(
            "cache_file",
            default_value="/data/rvclaw-k3-lab/state/localization-pose.json",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("start_hardware", default_value="true"),
        DeclareLaunchArgument("start_lidar", default_value="true"),
        DeclareLaunchArgument("start_rviz", default_value="false"),
    ] + [
        DeclareLaunchArgument(name, description=f"Measured laser transform: {name}")
        for name in laser_arguments
    ]

    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("rvclaw_stm32_base"), "launch", "hardware.launch.py"]
            )
        ),
        launch_arguments={"enable_hardware_output": "true"}.items(),
        condition=IfCondition(start_hardware),
    )
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "global_scan_lidar.launch.py"])
        ),
        condition=IfCondition(start_lidar),
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "navigation.launch.py"])
        ),
        launch_arguments={
            "map": map_file,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "start_safety_supervisor": "false",
            **{name: LaunchConfiguration(name) for name in laser_arguments},
        }.items(),
    )
    localization_runtime = Node(
        package="rvclaw_nav2_bringup",
        executable="rvclaw-localization-runtime",
        name="rvclaw_localization_runtime",
        output="screen",
        parameters=[
            {
                "map_file": map_file,
                "cache_file": cache_file,
                "use_sim_time": use_sim_time,
            }
        ],
    )
    visualization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "visualization.launch.py"])
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription(
        declarations
        + [hardware, lidar, navigation, localization_runtime, visualization]
    )

