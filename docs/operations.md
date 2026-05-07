# Operations Runbook

## 本地运行

```powershell
$env:PYTHONPATH = "src"
python -m rvclaw run --planner mock
```

## SG2044 运行

参考 `deploy/sg2044/install.md`。首次验收建议固定记录：

- SoC：SG2044
- core：64 core
- ISA：RVV 1.0
- VLEN：128
- OS：openEuler/EulixOS 或实际发行版
- compiler：GCC/Clang 版本
- runtime：engine、commit、build flags
- model：model、dtype、quant、kv_dtype、kv_len

## Run Artifacts

每次运行必须保留：

- `task.yaml`
- `metrics.json`
- `trace.jsonl`
- `report.md`
- `raw.log`

这些产物用于 demo replay、benchmark 汇总、问题定位和验收归档。
