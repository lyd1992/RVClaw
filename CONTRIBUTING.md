# Contributing

RVClaw 当前阶段优先保证 Demo Claw v0.1 可运行、可复现、可验收。

## 协作原则

- 先交 mock 或 baseline，再交真实优化后端。
- 任一后端未完成时，不阻塞 Demo 主链路。
- 统一接口优先于单点性能。
- 性能数字必须标注平台、模型、后端、commit/build flags 和测试环境。

## 推荐分工

| 方向 | 近期重点 |
|---|---|
| Agent Core / Benchmark | skeleton、mock skills、metrics、trace、report |
| SG2044 / llama.cpp | GGUF baseline、SQLite/flat memory、runbook |
| MNN / ONNX | 视觉 baseline、MNN 插件接口 |
| vLLM | 服务化 LLM 后端、KV cache 路线验证 |
| Knowhere / Milvus | MemoryBackend 插件、向量检索 benchmark |
| Device Adapter | Mock Device -> ROS2/OpenClaw adapter |

## 提交前检查

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m rvclaw run --planner mock
```
