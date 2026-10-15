from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("rvclaw_nav2_bringup")
    use_sim_time = LaunchConfiguration("use_sim_time")
    laser_arguments = [
        "laser_x",
        "laser_y",
        "laser_z",
        "laser_roll",
        "laser_pitch",
        "laser_yaw",
    ]

    declarations = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
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
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_params_file": PathJoinSubstitution(
                [package_share, "config", "slam_toolbox.yaml"]
            ),
        }.items(),
    )

    return LaunchDescription(declarations + [sensor_transforms, odom_tf, slam])
