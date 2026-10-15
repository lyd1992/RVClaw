from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("rvclaw_nav2_bringup")
    map_file = LaunchConfiguration("map")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    start_sensor_transforms = LaunchConfiguration("start_sensor_transforms")
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
            description="Keep false until the supervised localization session",
        ),
        DeclareLaunchArgument(
            "start_sensor_transforms",
            default_value="true",
            choices=["true", "false"],
            description="Start project sensor TF, or reuse a separately verified publisher",
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
        condition=IfCondition(start_sensor_transforms),
    )
    odom_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "odom_tf.launch.py"])
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nav2_bringup"), "launch", "localization_launch.py"]
            )
        ),
        launch_arguments={
            "map": map_file,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "params_file": PathJoinSubstitution(
                [package_share, "config", "nav2_params.yaml"]
            ),
            "use_composition": "False",
            "use_respawn": "false",
        }.items(),
    )

    return LaunchDescription(
        declarations + [sensor_transforms, odom_tf, localization]
    )
