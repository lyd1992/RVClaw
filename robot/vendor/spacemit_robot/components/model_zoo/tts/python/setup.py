"""
Space TTS Python Module Setup

Build and install:
    pip install .

Development install:
    pip install -e .

Build wheel:
    pip wheel . -w dist

Build only (inplace):
    python setup.py build_ext --inplace
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for pybind11 module"""

    def __init__(self, name, source_dir=""):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    """Build extension using CMake"""

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_dir = os.path.join(ext.source_dir, "build")

        # Create build directory
        os.makedirs(build_dir, exist_ok=True)

        # Clean CMakeCache.txt if source directory changed (avoid cache conflicts)
        cache_file = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                content = f.read()
                if ext.source_dir not in content:
                    import shutil
                    shutil.rmtree(build_dir)
                    os.makedirs(build_dir, exist_ok=True)

        # CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_BINDINGS=ON",
        ]

        # Build type
        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

        # Build arguments
        build_args = ["--config", build_type, "--target", "_spacemit_tts"]

        # Parallel build
        if hasattr(os, "cpu_count"):
            build_args.extend(["--", f"-j{os.cpu_count()}"])

        # Configure
        subprocess.check_call(
            ["cmake", ext.source_dir] + cmake_args,
            cwd=build_dir
        )

        # Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir
        )

        # Copy built module to package directory
        built_lib = None
        for suffix in [".so", ".dylib", ".pyd"]:
            pattern = f"_spacemit_tts*{suffix}"
            for path in Path(build_dir).rglob(pattern):
                built_lib = path
                break
            if built_lib:
                break

        if built_lib:
            import shutil
            dest = os.path.join(ext_dir, built_lib.name)
            shutil.copy2(built_lib, dest)


# Read version from C++ header
def get_version():
    return "1.0.4"


# Read long description
def get_long_description():
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text()
    return "Space TTS Python bindings for Matcha-TTS"


setup(
    name="spacemit_tts",
    version=get_version(),
    author="muggle",
    author_email="promuggle@gmail.com",
    description="Space TTS (Text-To-Speech) Python bindings",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/muggle/tts",
    packages=find_packages(),
    ext_modules=[CMakeExtension("spacemit_tts._spacemit_tts", source_dir="..")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="tts text-to-speech matcha synthesis",
)
