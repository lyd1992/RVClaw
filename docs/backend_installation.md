# Backend Installation

RVClaw 是软件底座，因此需要管理相关 runtime/framework 的源码安装。但这些后端不应成为 Demo Claw v0.1 的强依赖。当前策略是：

- 默认主链路继续保持零第三方依赖，保证 mock demo 可跑通。
- 后端安装作为可选能力，由 `rvclaw install` 统一生成计划、脚本或执行。
- 每个后端都固定源码目录、git ref、构建命令和验证命令。
- SG2044 / openEuler 上先建立可复现 baseline，再接性能优化分支。

## CLI

检查环境：

```bash
export PYTHONPATH="$PWD/src"
python3 -m rvclaw doctor
```

列出可安装后端：

```bash
python3 -m rvclaw install list
```

查看安装计划，不执行：

```bash
python3 -m rvclaw install plan llama_cpp mnn
python3 -m rvclaw install plan vllm
```

生成 shell 脚本：

```bash
python3 -m rvclaw install script llama_cpp mnn --output deploy/sg2044/install_backends.sh
bash deploy/sg2044/install_backends.sh
```

直接执行安装命令：

```bash
python3 -m rvclaw install run llama_cpp --yes
python3 -m rvclaw install run mnn --yes
```

默认源码目录是 `third_party/`，可以改为独立磁盘路径：

```bash
python3 -m rvclaw install run llama_cpp --deps-dir /data/rvclaw-third-party --yes
```

固定上游版本：

```bash
python3 -m rvclaw install script llama_cpp --ref llama_cpp=master
python3 -m rvclaw install script mnn --ref mnn=3.5.0
python3 -m rvclaw install script vllm --ref vllm=main
```

## Backends

| Backend | RVClaw 级别 | 当前安装动作 |
|---|---|---|
| llama.cpp | P0 baseline | clone 源码，CMake Release 构建，验证 `llama-cli` |
| MNN | P1 plugin | clone 源码，CMake Release 构建 CPU/tools/benchmark，关闭 converter/test |
| vLLM | P1/P2 experimental | clone 源码，执行 Python development install；不承诺 RISC-V 优化后端 |

## openEuler Dependencies

建议先安装基础构建工具：

```bash
sudo dnf install -y git gcc gcc-c++ make cmake python3 python3-pip ccache
```

如果需要 Python 虚拟环境：

```bash
sudo dnf install -y python3-virtualenv
python3 -m venv .venv
source .venv/bin/activate
```

## Notes

### llama.cpp

llama.cpp 官方构建入口是 CMake。RVClaw 默认只做 CPU Release build：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j ${MAX_JOBS:-$(nproc)}
```

SG2044 上如果要固定 `-march` / `-mtune` / RVV flags，建议通过外部环境变量传入，例如：

```bash
export CFLAGS="-O3"
export CXXFLAGS="-O3"
```

不要在通用 installer 中硬编码某个实验性编译参数。

### MNN

MNN 官方使用 CMake 构建，并通过大量 CMake option 控制功能。RVClaw 默认构建 CPU baseline 和 benchmark：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_CONVERTER=OFF
```

`MNN_BUILD_CONVERTER=ON` 通常会引入 protobuf 等额外依赖，建议等基础 runtime 可复现后再打开。

### vLLM

vLLM 最新稳定文档采用硬件插件化口径，但当前官方硬件列表没有把 RISC-V 作为一等支持后端。RVClaw 因此只提供源码/Python development checkout：

```bash
VLLM_TARGET_DEVICE=empty python3 -m pip install -e .
```

这一步用于代码研究、接口适配和后续 RVV backend 工作，不代表已经得到可用于 SG2044 的高性能 vLLM runtime。

## Official References

- llama.cpp build docs: <https://www.mintlify.com/ggml-org/llama.cpp/development/building>
- MNN CMake docs: <https://github.com/alibaba/MNN/wiki/cmake>
- vLLM installation docs: <https://docs.vllm.ai/en/stable/getting_started/installation/>
