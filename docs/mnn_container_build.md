# MNN Container Build On SG2044

This document describes the first RVClaw containerized backend workflow:
build MNN inside a Docker container with a prebuilt GCC 15.1 toolchain image.

## What Must Be Built First

Build this image once on SG2044:

```text
rvclaw/sg2044-gcc15.1:openeuler-riscv64
```

The image contains openEuler, GCC/G++ 15.1, CMake, Ninja, Python, git,
make, ccache, and common build dependencies.

MNN source and build artifacts are intentionally not baked into the image.
They are mounted from the RVClaw workspace so commits, CMake flags, logs,
and benchmark results stay visible on the host.

## Build The GCC 15.1 Image

Run on SG2044, not through x86 QEMU emulation:

```bash
cd RVClaw
bash docker/sg2044/gcc15.1/build_image.sh
```

Optional variables:

```bash
export RVCLAW_GCC15_IMAGE=rvclaw/sg2044-gcc15.1:openeuler-riscv64
export RVCLAW_OPENEULER_BASE_IMAGE=openeuler/openeuler:25.03
export RVCLAW_GCC_VERSION=15.1.0
export MAX_JOBS=32
bash docker/sg2044/gcc15.1/build_image.sh
```

If the public openEuler RISC-V image is unavailable, import an openEuler
riscv64 rootfs on SG2044 and set `RVCLAW_OPENEULER_BASE_IMAGE` to that local
image tag before building.

## Verify The Image

```bash
bash docker/sg2044/gcc15.1/verify_image.sh
```

The verification checks:

- `gcc --version`
- `g++ --version`
- `cmake --version`
- a small compile using `-march=rv64gcv -mabi=lp64d`

## Build MNN In The Container

Plan only:

```bash
export PYTHONPATH="$PWD/src"
python3 -m rvclaw container mnn plan
```

Run:

```bash
bash deploy/sg2044/mnn_container_build.sh
```

Equivalent direct command:

```bash
bash docker/sg2044/mnn/run_container.sh bash docker/sg2044/mnn/build_mnn.sh
```

The container mounts:

```text
RVClaw/              -> /workspace/RVClaw
RVClaw/third_party/  -> /workspace/RVClaw/third_party
RVClaw/build/        -> /workspace/RVClaw/build
RVClaw/runs/         -> /workspace/RVClaw/runs
RVClaw/.ccache/      -> /workspace/.ccache
```

Expected host artifacts:

```text
third_party/MNN/
build/mnn-sg2044-gcc15/
runs/env/<run_id>/container_manifest.json
runs/env/<run_id>/mnn_build.log
runs/env/<run_id>/mnn_cmake_cache.txt
runs/env/<run_id>/mnn_artifacts.txt
```

## MNN Build Defaults

```bash
cmake -S third_party/MNN -B build/mnn-sg2044-gcc15 \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_CONVERTER=OFF
```

RVV flags are checked before use. If GCC accepts
`-march=rv64gcv -mabi=lp64d`, the build script appends them to
`CFLAGS` and `CXXFLAGS`. Otherwise it continues without explicit RVV flags.

## Save Or Share The Image

Save locally:

```bash
docker save rvclaw/sg2044-gcc15.1:openeuler-riscv64 \
  -o rvclaw-sg2044-gcc15.1-openeuler-riscv64.tar
```

Load on another SG2044:

```bash
docker load -i rvclaw-sg2044-gcc15.1-openeuler-riscv64.tar
```

Or push to a private registry:

```bash
docker tag rvclaw/sg2044-gcc15.1:openeuler-riscv64 registry.example.com/rvclaw/sg2044-gcc15.1:openeuler-riscv64
docker push registry.example.com/rvclaw/sg2044-gcc15.1:openeuler-riscv64
```

## Device Backend Notes

Docker does not block later device backends. Use precise device passthrough
when needed:

```bash
docker run --device=/dev/video0 ...
docker run --device=/dev/ttyUSB0 ...
docker run --net=host --ipc=host ...
```

Avoid `--privileged` as the default. Reserve it for temporary hardware
debugging. Keep a bare-metal path for strong real-time or kernel-driver work.
