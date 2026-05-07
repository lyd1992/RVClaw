ARG BASE_IMAGE=openeuler/openeuler:25.03
FROM ${BASE_IMAGE}

ARG GCC_VERSION=15.1.0
ARG PREFIX=/opt/rvclaw/gcc-15.1
ARG MAX_JOBS=32

LABEL org.opencontainers.image.title="RVClaw SG2044 GCC 15.1 toolchain"
LABEL org.opencontainers.image.description="openEuler RISC-V toolchain image for RVClaw MNN builds"

RUN dnf install -y \
    git wget tar xz make cmake ninja-build \
    gcc gcc-c++ \
    glibc-devel \
    gmp-devel mpfr-devel libmpc-devel isl-devel \
    zlib-devel openssl-devel \
    python3 python3-pip \
    ccache findutils which file diffutils perl patch ca-certificates \
 && dnf clean all

WORKDIR /tmp/toolchain

RUN wget https://gcc.gnu.org/pub/gcc/releases/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz \
 && tar -xf gcc-${GCC_VERSION}.tar.xz \
 && mkdir gcc-build \
 && cd gcc-build \
 && ../gcc-${GCC_VERSION}/configure \
      --prefix=${PREFIX} \
      --enable-languages=c,c++ \
      --disable-multilib \
      --disable-bootstrap \
 && make -j${MAX_JOBS} \
 && make install \
 && cd / \
 && rm -rf /tmp/toolchain

ENV PATH="${PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${PREFIX}/lib64:${PREFIX}/lib:${LD_LIBRARY_PATH}"
ENV CC="${PREFIX}/bin/gcc"
ENV CXX="${PREFIX}/bin/g++"

RUN gcc --version && g++ --version && cmake --version && ninja --version

WORKDIR /workspace/RVClaw
