# RVClaw Agent Notes

RVClaw 是 Demo Claw v0.1 的代码仓。进入本仓库时，请优先遵循以下约定：

1. 产品主线是 RISC-V 端侧 Agent Runtime，不要把项目叙述成 vLLM/MNN/Knowhere 三仓优化合集。
2. v0.1 必须先跑通 mock/baseline 闭环，再接优化后端。
3. 当前 demo 部署优先面向 K3 Pico-ITX 32GB + Bianbu/Ubuntu + llama.cpp；SG2044/openEuler 保留为服务器型验证和历史辅助平台。
4. 每次 demo run 必须生成 `task.yaml`、`metrics.json`、`trace.jsonl`、`report.md`、`raw.log`。
5. 新增后端要挂到已有接口：RuntimeBackend、MemoryBackend、Skill API、Device Adapter API。
6. 性能数字必须记录来源和环境，不要混用本地报告、PR 描述和 PPT 展示口径。
7. K3 机器侧 SSH 部署、模型路径、`spacemit-llama.cpp` 包和环境变量配置必须放在 `deploy/k3/` 或 `docs/k3_ssh_deployment.md`，不要写死进核心 Python 代码。
8. RVClaw 代码以 GitHub 上游仓库为准；K3 运行产物、模型文件和官方二进制包放在 `/data/rvclaw`，不要提交进仓库。

默认本地检查：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m rvclaw run --planner mock
```

K3 SSH 验收入口：

```bash
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
bash deploy/k3/run_demo.sh
```

本地 `llama.cpp` Planner 入口：

```bash
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" --planner llama_cpp --runs-dir /data/rvclaw/runs
```
