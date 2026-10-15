from glob import glob

from setuptools import find_packages, setup


package_name = "rvclaw_nav2_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        (
            "share/" + package_name + "/behavior_trees",
            glob("behavior_trees/*.xml"),
        ),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="RVClaw team",
    maintainer_email="devnull@example.com",
    description="Low-speed Nav2 and SLAM bringup for the RVClaw V550.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "cmd_vel_adapter = rvclaw_nav2_bringup.cmd_vel_adapter:main",
            "rvclaw-check-nav-readiness = "
            "rvclaw_nav2_bringup.nav_readiness_check:main",
            "rvclaw-generate-geometry = "
            "rvclaw_nav2_bringup.measurement_geometry:main",
            "rvclaw-record-odom-calibration = "
            "rvclaw_nav2_bringup.odom_calibration_recorder:main",
            "rvclaw-bounded-backup = "
            "rvclaw_nav2_bringup.bounded_backup_client:main",
            "rvclaw-backup-corridor-filter = "
            "rvclaw_nav2_bringup.bounded_backup_corridor_filter:main",
            "rvclaw-nav2-waypoint = "
            "rvclaw_nav2_bringup.waypoint_client:main",
            "rvclaw-odom-tf = rvclaw_nav2_bringup.odom_tf_broadcaster:main",
            "rvclaw-check-amcl-localization = "
            "rvclaw_nav2_bringup.amcl_localization_observer:main",
            "rvclaw-observe-amcl-motion = "
            "rvclaw_nav2_bringup.amcl_motion_observer:main",
            "rvclaw-observe-amcl-reposition = "
            "rvclaw_nav2_bringup.amcl_reposition_observer:main",
            "rvclaw-localization-runtime = "
            "rvclaw_nav2_bringup.localization_runtime_manager:main",
            "rvclaw-scan-self-filter = "
            "rvclaw_nav2_bringup.scan_self_filter:main",
             "rvclaw-frontier-explorer = "
             "rvclaw_nav2_bringup.frontier_explorer:main",
             "rvclaw-validate-mission = "
             "rvclaw_nav2_bringup.offline_mission_validator:main",
             "rvclaw-parse-voice-command = "
             "rvclaw_nav2_bringup.voice_command_contract:main",
             "rvclaw-validate-map-mission = "
             "rvclaw_nav2_bringup.map_acceptance:main",
             "rvclaw-validate-task-workers = "
             "rvclaw_nav2_bringup.task_worker_contract:main",
             "rvclaw-inspection-readiness = "
             "rvclaw_nav2_bringup.inspection_task_runtime:main",
             "rvclaw-mission-release-check = "
             "rvclaw_nav2_bringup.mission_release_check:main",
             "rvclaw-catalog-from-waypoints = "
             "rvclaw_nav2_bringup.catalog_from_waypoints:main",
         ],
     },
 )
