#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export RVCLAW_DEVICE_BACKEND="${RVCLAW_DEVICE_BACKEND:-demozoo}"
export RVCLAW_VISION_BACKEND="demozoo"
export RVCLAW_REQUIRE_REAL_VISION="1"

exec bash "${SCRIPT_DIR}/run_web_demo.sh"
