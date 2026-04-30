#!/usr/bin/env bash
# XSA overhead runner. Runs INSIDE a RunPod pod.
# Sequence:
#   baseline, xsa, baseline, xsa, baseline, xsa, baseline_profile, xsa_profile

set -euo pipefail

NANOCHAT_REPO="${NANOCHAT_REPO:-Team-XSA/nanochat}"
NANOCHAT_REF="${NANOCHAT_REF:-dev}"
HF_REPO="${HF_REPO:-Team-XSA/nanochat-xsa-overhead}"
TOKENIZER_HF_REPO="${TOKENIZER_HF_REPO:-Team-XSA/1.3B_baseline}"
WANDB_RUN="${WANDB_RUN:-xsa_overhead}"

DATA_SHARDS="${DATA_SHARDS:-8}"
NPROC="${NPROC:-2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-80}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-40}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1048576}"
FP8="${FP8:-TRUE}"
XSA_ALPHA="${XSA_ALPHA:-1.0}"
XSA_LAYER_INDICES="${XSA_LAYER_INDICES:-}"
UPLOAD_FAILURE_CACHE="${UPLOAD_FAILURE_CACHE:-0}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="xsa_overhead_d24_${TS}"
WORKDIR="/workspace/nanochat"
LOG_FILE="/workspace/runner.log"
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
RESULTS_DIR="$NANOCHAT_BASE_DIR/$EXPERIMENT_NAME"

mkdir -p /workspace "$NANOCHAT_BASE_DIR" "$RESULTS_DIR/logs" "$RESULTS_DIR/profiles"

echo "[overhead] $(date -Iseconds) starting pod=${RUNPOD_POD_ID:-unknown}"
echo "[overhead] repo=$NANOCHAT_REPO ref=$NANOCHAT_REF hf_repo=$HF_REPO"
echo "[overhead] nproc=$NPROC shards=$DATA_SHARDS iters=$NUM_ITERATIONS profile_iters=$PROFILE_ITERATIONS"

{ pip3 install --break-system-packages --quiet --upgrade huggingface_hub 2>&1 || \
  python3 -m pip install --break-system-packages --quiet --upgrade huggingface_hub 2>&1 || \
  echo "[overhead] WARN: could not pre-install huggingface_hub"; } || true

cleanup() {
  local rc=$?
  set +e
  echo "[overhead] cleanup rc=$rc at $(date -Iseconds)"

  mkdir -p "$RESULTS_DIR/logs"
  cp /workspace/*.log "$RESULTS_DIR/logs/" 2>/dev/null || true
  [ -d "$WORKDIR" ] && (cd "$WORKDIR" && git rev-parse HEAD > "$RESULTS_DIR/git-head.txt" 2>/dev/null || true)
  echo "rc=$rc ts=$TS pod=${RUNPOD_POD_ID:-unknown}" > "$RESULTS_DIR/result.txt"

  if [ "$rc" -eq 0 ]; then
    hf upload "$HF_REPO" "$RESULTS_DIR" "$EXPERIMENT_NAME" \
      --repo-type model --commit-message "xsa overhead rc=0 $TS" || \
      echo "[overhead] WARN: result upload failed"
    echo "[overhead] artifacts: https://huggingface.co/$HF_REPO/tree/main/$EXPERIMENT_NAME"
  else
    hf upload "$HF_REPO" "$RESULTS_DIR" "_failures/${EXPERIMENT_NAME}-rc${rc}" \
      --repo-type model --commit-message "xsa overhead failure rc=$rc $TS" || \
      echo "[overhead] WARN: failure upload failed"
    if [ "$UPLOAD_FAILURE_CACHE" = "1" ]; then
      hf upload "$HF_REPO" "$NANOCHAT_BASE_DIR" "_failures/${EXPERIMENT_NAME}-rc${rc}/cache" \
        --repo-type model --commit-message "xsa overhead failure cache rc=$rc $TS" \
        --exclude "base_data_climbmix/**" --exclude "wandb/**" || true
    fi
  fi

  if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo "[overhead] self-deleting pod $RUNPOD_POD_ID"
    curl -fsS -X DELETE \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" 2>&1 || \
      echo "[overhead] WARN: pod delete failed; delete manually"
  fi
  exit "$rc"
}
trap cleanup EXIT

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

rm -rf "$WORKDIR"
git clone "https://github.com/${NANOCHAT_REPO}.git" "$WORKDIR"
cd "$WORKDIR"
git checkout "$NANOCHAT_REF" --
echo "[overhead] HEAD=$(git rev-parse HEAD)"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR
export HF_HUB_TOKEN="$HF_TOKEN"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
uv pip install --quiet --upgrade huggingface_hub hf_transfer 'kernels>=0.13.0'

python -c "import torch; print('[overhead] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
python "$WORKDIR/runs/runpod/probe_fa3.py" || echo "[overhead] FA3 probe reported issues; continuing"

echo "[overhead] downloading tokenizer from $TOKENIZER_HF_REPO"
hf download "$TOKENIZER_HF_REPO" --repo-type model \
  --include "tokenizer/**" --local-dir "$NANOCHAT_BASE_DIR"
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
  echo "[overhead] tokenizer missing after download; training tokenizer from downloaded shards"
  python -m nanochat.dataset -n "$DATA_SHARDS"
  python -m scripts.tok_train --max-chars=50000000
else
  echo "[overhead] tokenizer ready: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
fi

echo "[overhead] downloading dataset shards"
python -m nanochat.dataset -n "$DATA_SHARDS"

run_one() {
  local label="$1"
  local type="$2"
  local profile="$3"
  local iterations="$4"

  local fp8_arg=()
  local xsa_arg=()
  local profile_arg=()
  [ "$FP8" = "TRUE" ] && fp8_arg=(--fp8)
  if [ "$type" = "xsa" ]; then
    xsa_arg=(--xsa "--xsa-alpha=$XSA_ALPHA")
    [ -n "$XSA_LAYER_INDICES" ] && xsa_arg+=("--xsa-layer-indices=$XSA_LAYER_INDICES")
  fi
  [ "$profile" = "TRUE" ] && profile_arg=(--profile "--profile-dir=$RESULTS_DIR/profiles/$label")

  echo "[overhead] === $label type=$type profile=$profile iterations=$iterations ==="
  {
    echo "label=$label"
    echo "type=$type"
    echo "profile=$profile"
    echo "iterations=$iterations"
    echo "started=$(date -Iseconds)"
    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
      --depth=24 \
      --target-param-data-ratio=8 \
      --total-batch-size="$TOTAL_BATCH_SIZE" \
      --device-batch-size="$DEVICE_BATCH_SIZE" \
      --num-iterations="$iterations" \
      --eval-every=-1 \
      --core-metric-every=-1 \
      --sample-every=-1 \
      --save-every=-1 \
      --no-save \
      --run="${WANDB_RUN}_${label}" \
      --model-tag="$label" \
      "${fp8_arg[@]}" \
      "${xsa_arg[@]}" \
      "${profile_arg[@]}"
    echo "finished=$(date -Iseconds)"
  } 2>&1 | tee "$RESULTS_DIR/logs/${label}.log"
}

run_one baseline_timing_1 baseline FALSE "$NUM_ITERATIONS"
run_one xsa_timing_1 xsa FALSE "$NUM_ITERATIONS"
run_one baseline_timing_2 baseline FALSE "$NUM_ITERATIONS"
run_one xsa_timing_2 xsa FALSE "$NUM_ITERATIONS"
run_one baseline_timing_3 baseline FALSE "$NUM_ITERATIONS"
run_one xsa_timing_3 xsa FALSE "$NUM_ITERATIONS"
run_one baseline_profile_1 baseline TRUE "$PROFILE_ITERATIONS"
run_one xsa_profile_1 xsa TRUE "$PROFILE_ITERATIONS"

echo "[overhead] complete at $(date -Iseconds)"
