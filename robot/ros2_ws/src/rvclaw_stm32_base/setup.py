from glob import glob
from setuptools import find_packages, setup


package_name = "rvclaw_stm32_base"

setup(
    name=package_name,
    version="0.4.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="RVClaw team",
    maintainer_email="devnull@example.com",
    description=(
        "Single-owner STM32 hardware bridge, telemetry, and guarded control "
        "for RVClaw."
    ),
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "telemetry_node = rvclaw_stm32_base.telemetry_node:main",
            "safety_supervisor_node = rvclaw_stm32_base.safety_supervisor_node:main",
            "hardware_node = rvclaw_stm32_base.hardware_node:main",
        ],
    },
)
