import platform

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def riscv_compile_args():
    machine = platform.machine().lower()
    if "riscv" not in machine:
        return []
    return [
        "-O2",
        "-fno-strict-overflow",
        "-mstrict-align",
        "-fno-strict-aliasing",
        "-fno-aggressive-loop-optimizations",
    ]


ext_modules = [
    Pybind11Extension(
        "spacemit_audio_process._spacemit_audio_process",
        [
            "src/bindings.cpp",
            "../src/sound_locator.cpp",
            "../src/gcc_phat_pair.cpp",
            "../src/multi_sound_locator.cpp",
        ],
        include_dirs=["../include"],
        libraries=["fftw3f", "m"],
        extra_compile_args=riscv_compile_args(),
        cxx_std=17,
    ),
]

setup(
    name="spacemit_audio_process",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["spacemit_audio_process"],
)
