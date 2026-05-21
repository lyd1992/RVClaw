# Operations Runbook

## 本地运行

```powershell
$env:PYTHONPATH = "src"
python -m rvclaw run --planner mock
```

## K3 运行

当前 K3 smoke checkpoint 使用：

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
```

另一个 SSH 终端运行：

```bash
source deploy/k3/env.sh
bash deploy/k3/run_demo.sh
```

预期默认巡检输出：

```text
completed
memory_query
move_to
capture_image
detect_status
speak
upload_report
```

支持的基础动作：

```bash
python3 -m rvclaw run "返回 BASE" --planner mock --runs-dir /data/rvclaw/runs --json
```

超出白名单的任务应返回 `failed` 并保留 run artifacts。详见：

```text
docs/k3_ssh_deployment.md
docs/development_status.md
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
