#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

GOAL="${1:-检查 A-03 区域设备状态并生成报告}"
PLANNER="${RVCLAW_PLANNER:-llama_cpp}"

cd "$RVCLAW_HOME"
python3 -m rvclaw run "$GOAL" \
  --planner "$PLANNER" \
  --runs-dir "$RVCLAW_RUNS_DIR" \
  --json
