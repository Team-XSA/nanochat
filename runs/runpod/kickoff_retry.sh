#!/usr/bin/env bash
# Retry-wrapper around kickoff.sh. Repeatedly attempts to allocate a pod,
# backing off with jitter when capacity is tight. Stops on first success
# or on a fatal (auth / usage / runner-unreachable) error.
#
# Usage (forwards every arg to kickoff.sh):
#   bash runs/runpod/kickoff_retry.sh d12_xsa
#   GPU_COUNT=6 bash runs/runpod/kickoff_retry.sh d12_xsa
#
# Tunables (env vars):
#   BASE_INTERVAL     default 120  — seconds between attempts (jitter applied)
#   JITTER_PCT        default 50   — ± this % of BASE_INTERVAL
#   MIN_INTERVAL      default 60   — floor on actual sleep
#   MAX_INTERVAL      default 300  — cap on actual sleep
#   MAX_ATTEMPTS      default 60   — hard attempt cap
#   TOTAL_BUDGET_MIN  default 240  — total wall-clock cap (minutes)
#   GPU_ID            default "NVIDIA H100 80GB HBM3" — passed to gpu-list precheck
#   SKIP_PRECHECK     default 0    — set 1 to skip `runpodctl gpu list` precheck

set -uo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash runs/runpod/kickoff_retry.sh <runner-name> [extra args]" >&2
  exit 2
fi

BASE_INTERVAL="${BASE_INTERVAL:-120}"
JITTER_PCT="${JITTER_PCT:-50}"
MIN_INTERVAL="${MIN_INTERVAL:-60}"
MAX_INTERVAL="${MAX_INTERVAL:-300}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-60}"
TOTAL_BUDGET_MIN="${TOTAL_BUDGET_MIN:-240}"
GPU_ID="${GPU_ID:-NVIDIA H100 80GB HBM3}"
SKIP_PRECHECK="${SKIP_PRECHECK:-0}"

START_TS=$(date +%s)

# Compute a jittered sleep value: BASE ± (BASE * JITTER_PCT/100), clamped.
jittered_sleep() {
  local base="$1"
  local pct="$2"
  local span=$(( base * pct / 100 ))
  # range: -span..+span
  local jitter=$(( (RANDOM % (2 * span + 1)) - span ))
  local s=$(( base + jitter ))
  [ "$s" -lt "$MIN_INTERVAL" ] && s="$MIN_INTERVAL"
  [ "$s" -gt "$MAX_INTERVAL" ] && s="$MAX_INTERVAL"
  echo "$s"
}

# Check the GPU type is currently in the available list. Cheap; doesn't allocate.
# Returns 0 if available, 1 otherwise. 0 also if precheck disabled or runpodctl
# missing (let kickoff handle it).
gpu_available() {
  [ "$SKIP_PRECHECK" = "1" ] && return 0
  command -v runpodctl >/dev/null 2>&1 || return 0
  runpodctl gpu list -o json 2>/dev/null | python3 - "$GPU_ID" <<'PY' || return 1
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
target = sys.argv[1]
for g in data:
    if g.get("gpuId") == target and g.get("available"):
        print(f"[precheck] {target} stockStatus={g.get('stockStatus')}", file=sys.stderr)
        sys.exit(0)
print(f"[precheck] {target} not currently available", file=sys.stderr)
sys.exit(1)
PY
}

# Patterns that mean "do not retry" — auth/usage/local errors.
fatal_pattern() {
  grep -qiE 'unauthorized|invalid.*(token|api.?key)|forbidden|HTTP.40[13]|runner not reachable|HF_TOKEN must be set|WANDB_API_KEY must be set|RUNPOD_TEMPLATE_ID not set|Usage:'
}

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  elapsed_min=$(( ($(date +%s) - START_TS) / 60 ))
  if [ "$elapsed_min" -ge "$TOTAL_BUDGET_MIN" ]; then
    echo "[retry] hit total time budget ($TOTAL_BUDGET_MIN min) at attempt $attempt — giving up" >&2
    exit 1
  fi

  echo "[retry] attempt $attempt/$MAX_ATTEMPTS at $(date -Iseconds) (elapsed ${elapsed_min}m)"

  if ! gpu_available; then
    s=$(jittered_sleep "$BASE_INTERVAL" "$JITTER_PCT")
    echo "[retry] GPU not available — sleeping ${s}s before next precheck"
    sleep "$s"
    attempt=$((attempt + 1))
    continue
  fi

  TMP_OUT=$(mktemp)
  set +e
  GPU_ID="$GPU_ID" bash runs/runpod/kickoff.sh "$@" 2>&1 | tee "$TMP_OUT"
  rc=${PIPESTATUS[0]}
  set -e

  if [ "$rc" -eq 0 ]; then
    echo "[retry] kickoff succeeded on attempt $attempt — pod allocation accepted"
    rm -f "$TMP_OUT"
    exit 0
  fi

  # Decide: fatal or transient?
  if fatal_pattern < "$TMP_OUT"; then
    echo "[retry] fatal error pattern matched (rc=$rc) — not retrying" >&2
    rm -f "$TMP_OUT"
    exit "$rc"
  fi

  rm -f "$TMP_OUT"
  s=$(jittered_sleep "$BASE_INTERVAL" "$JITTER_PCT")
  echo "[retry] kickoff failed (rc=$rc) — treating as transient, sleeping ${s}s"
  sleep "$s"
  attempt=$((attempt + 1))
done

echo "[retry] exceeded MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up" >&2
exit 1
