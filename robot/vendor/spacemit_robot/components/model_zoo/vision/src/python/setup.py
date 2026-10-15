from setuptools import find_packages, setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Force platform-specific wheel so .so files are included."""
    def has_ext_modules(self):
        return True


setup(
    name="spacemit_vision",
    version="0.1.3",
    description="Python wheel for VisionService C++ bindings",
    package_dir={"": "."},
    packages=find_packages(where="."),
    include_package_data=True,
    package_data={
        "spacemit_vision": [
            "_vision_service_cpp*.so",
            "libvision.so",
        ]
    },
    install_requires=[
        "numpy>=1.26.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
    distclass=BinaryDistribution,
)

