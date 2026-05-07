#!/usr/bin/env bash
set -euo pipefail

export RVCLAW_HOME="${RVCLAW_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${RVCLAW_HOME}/src:${PYTHONPATH:-}"
export RVCLAW_PLATFORM_SOC="${RVCLAW_PLATFORM_SOC:-SG2044}"
export RVCLAW_RVV_VLEN="${RVCLAW_RVV_VLEN:-128}"
