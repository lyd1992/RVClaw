# SG2044 Install Notes

Demo Claw v0.1 的 Python skeleton 不依赖第三方包。建议先完成零依赖 mock run，再接入 llama.cpp、MNN、vLLM 等后端。

```bash
git clone https://github.com/lyd1992/RVClaw.git
cd RVClaw
source deploy/sg2044/env.sh
python3 -m rvclaw run --planner mock
```

## Optional Backend Installs

RVClaw can generate or run source-install plans for optional backends:

```bash
python3 -m rvclaw doctor
python3 -m rvclaw install list
python3 -m rvclaw install plan llama_cpp mnn
python3 -m rvclaw install script llama_cpp mnn --output deploy/sg2044/install_backends.generated.sh
bash deploy/sg2044/install_backends.generated.sh
```

Direct execution is also supported:

```bash
bash deploy/sg2044/install_backends.sh llama_cpp
bash deploy/sg2044/install_backends.sh mnn
```

The default source/build directory is `third_party/`. Use `--deps-dir /data/rvclaw-third-party` if the system disk is small.

验收时记录：

- SG2044 机器型号和 core 数
- OS 版本
- GCC/Clang 版本
- RVV VLEN=128
- Python 版本
- Git commit
- 模型和后端版本
