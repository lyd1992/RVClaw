# RVClaw Agent Notes

RVClaw 是 Demo Claw v0.1 的代码仓。进入本仓库时，请优先遵循以下约定：

1. 产品主线是 RISC-V 端侧 Agent Runtime，不要把项目叙述成 vLLM/MNN/Knowhere 三仓优化合集。
2. v0.1 必须先跑通 mock/baseline 闭环，再接优化后端。
3. 主验收平台是 SG2044 / RVV VLEN=128；K3 / VLEN=256 只作为后续兼容目标。
4. 每次 demo run 必须生成 `task.yaml`、`metrics.json`、`trace.jsonl`、`report.md`、`raw.log`。
5. 新增后端要挂到已有接口：RuntimeBackend、MemoryBackend、Skill API、Device Adapter API。
6. 性能数字必须记录来源和环境，不要混用本地报告、PR 描述和 PPT 展示口径。

默认本地检查：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m rvclaw run --planner mock
```
