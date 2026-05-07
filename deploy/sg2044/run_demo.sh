#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" --planner "${RVCLAW_PLANNER:-mock}"
