#!/bin/bash
# Gated TCM hygiene for the SpacemiT K3 board (RVClaw).
#
# WHY THIS EXISTS
#   The camera pipeline (yolov8_mjpeg) hardware-decodes MJPEG using TCM blocks.
#   When every block is held by a process/thread that no longer exists (an
#   orphaned / leaked pool), the decoder dies with:
#       terminate called after throwing an instance of 'std::runtime_error'
#         what():  tcm buffer acquire failed for core id N.
#   The service supervisor then restarts it, it dies again, and the web UI shows
#   no picture (online=false, fps=0) while the child PID churns every ~15s.
#
# WHY IT IS DELIBERATELY CONSERVATIVE
#   `spacemit-tcm-smi -c` frees ALL blocks -- it cannot target only the orphans.
#   On a HEALTHY board the camera legitimately holds all 8 blocks while decoding a
#   frame and releases them again, so the pool oscillates 0/8 <-> 8/8 about once a
#   second.  Calling -c during that churn would yank memory out from under live
#   decoders and could cause the very crash we are trying to prevent.
#   So we release ONLY when several consecutive samples show busy blocks with
#   ZERO live owners: a state where nothing can be harmed, and which is exactly
#   the leaked state that starves the camera.
#
# NOTE: the "PID" column holds THREAD ids.  On Linux /proc/<tid> exists for live
#   threads, so the liveness test below covers them.
#
# Healthy board => strict no-op.  Safe for ExecStartPre and ExecStopPost.
#
# Env overrides (used by the offline self-test):
#   RVCLAW_TCM_SMI    path to the smi binary (default /usr/bin/spacemit-tcm-smi)
#   RVCLAW_TCM_LOG    path to the log file (default /data/RVClaw/logs/tcm-hygiene.log)
#   RVCLAW_TCM_DRY=1  detect and log, but never actually call -c
#
# Usage: tcm_release_if_leaked.sh [tag]
set -u

SMI="${RVCLAW_TCM_SMI:-/usr/bin/spacemit-tcm-smi}"
LOG="${RVCLAW_TCM_LOG:-/data/RVClaw/logs/tcm-hygiene.log}"
TAG="${1:-manual}"
DRY="${RVCLAW_TCM_DRY:-0}"
SAMPLES=3          # consecutive fully-orphaned samples required before releasing
SAMPLE_DELAY=1     # seconds between samples

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [$TAG] $*" >> "$LOG" 2>/dev/null || true; }

if [ ! -x "$SMI" ]; then
    log "smi not found or not executable at $SMI -> skip"
    exit 0
fi

# One sample -> "busy_total live_owners dead_owners ownerless avail"
sample() {
    local out avail rows busy_total pids owned_total owned_alive owned_dead ownerless p
    out=$("$SMI" 2>/dev/null | tr -d '\r') || { echo "ERR"; return; }

    avail=$(echo "$out" | grep -oE 'available_blocks=[0-9]+/[0-9]+' | head -1)
    [ -n "$avail" ] || avail="available_blocks=unknown"

    # Busy rows: first field is a block id, second field is the literal "busy".
    rows=$(echo "$out" | awk '$1 ~ /^[0-9]+$/ && $2 == "busy"')
    busy_total=$(printf '%s\n' "$rows" | grep -c . || true)
    [ -n "$busy_total" ] || busy_total=0

    pids=$(printf '%s\n' "$rows" | awk '{print $NF}' | grep -E '^[0-9]+$' || true)
    owned_total=$(printf '%s\n' "$pids" | grep -c . || true)
    [ -n "$owned_total" ] || owned_total=0

    owned_alive=0
    owned_dead=0
    for p in $pids; do
        if [ -d "/proc/$p" ]; then owned_alive=$((owned_alive + 1)); else owned_dead=$((owned_dead + 1)); fi
    done
    ownerless=$((busy_total - owned_total))

    echo "$busy_total $owned_alive $owned_dead $ownerless $avail"
}

orphan_streak=0
summary=""
for i in $(seq 1 "$SAMPLES"); do
    s=$(sample)
    if [ "$s" = "ERR" ]; then
        log "smi query failed (sample $i/$SAMPLES) -> abort, no action"
        exit 0
    fi
    # shellcheck disable=SC2086
    set -- $s
    busy=$1; alive=$2; dead=$3; ownerless=$4; avail=$5
    summary="$summary [busy=$busy alive=$alive dead=$dead unowned=$ownerless $avail]"

    if [ "$busy" -gt 0 ] && [ "$alive" -eq 0 ]; then
        orphan_streak=$((orphan_streak + 1))
    else
        orphan_streak=0     # any live owner (or an idle pool) resets confidence
    fi
    [ "$i" -lt "$SAMPLES" ] && sleep "$SAMPLE_DELAY"
done

if [ "$orphan_streak" -lt "$SAMPLES" ]; then
    log "ok$summary -> no action (need $SAMPLES consecutive fully-orphaned samples)"
    exit 0
fi

# All samples agreed: busy blocks, zero live owners.  Nothing can be harmed.
if [ "$DRY" = "1" ]; then
    log "FULLY-ORPHANED$summary -> DRY-RUN, would force release (not executed)"
    exit 0
fi

log "FULLY-ORPHANED$summary -> force release"
res=$("$SMI" -c 2>&1 | tr -d '\r')
log "  release result: $res"
after=$("$SMI" 2>/dev/null | tr -d '\r' | grep -oE 'available_blocks=[0-9]+/[0-9]+' | head -1)
log "  after: ${after:-unknown}"
exit 0