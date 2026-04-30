#!/bin/bash
#SBATCH -N 1
#SBATCH -p ice-gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --constraint="gpu-h100|gpu-h200"
#SBATCH --mem-per-gpu=48G
#SBATCH -t 00:10:00
#SBATCH -J nanochat-xsa-overhead
#SBATCH -o runs/logs/xsa_overhead_%x_%j.out
#SBATCH -e runs/logs/xsa_overhead_%x_%j.err

set -e
cd "$HOME/scratch/nanochat"

export OMP_NUM_THREADS=1
mkdir -p runs/logs

LABEL="${LABEL:-baseline_timing_1}"
TYPE="${TYPE:-baseline}"
PROFILE="${PROFILE:-FALSE}"

mkdir -p "$RESULTS_DIR/profiles"

FP8_ARG=""
[ "$FP8" = "TRUE" ] && FP8_ARG="--fp8"

XSA_ARG=""
[ "$TYPE" = "xsa" ] && XSA_ARG="--xsa --xsa-alpha=$XSA_ALPHA"
[ "$TYPE" = "xsa" ] && [ -n "$XSA_LAYER_INDICES" ] && XSA_ARG="$XSA_ARG --xsa-layer-indices=$XSA_LAYER_INDICES"

PROFILE_ARG=""
[ "$PROFILE" = "TRUE" ] && PROFILE_ARG="--profile --profile-dir=$RESULTS_DIR/profiles/$LABEL"

RUN_NAME="$WANDB_RUN"
[ "$WANDB_RUN" != "dummy" ] && RUN_NAME="${WANDB_RUN}_${LABEL}"

echo "=== XSA overhead job: $LABEL ==="
echo "Type: $TYPE"
echo "Profile: $PROFILE"
echo "Num iterations: $NUM_ITERATIONS"
echo "Results dir: $RESULTS_DIR"
echo "Started: $(date)"

source .venv/bin/activate

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --num-iterations=$NUM_ITERATIONS \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --no-save \
    --run=$RUN_NAME \
    --model-tag=$LABEL \
    $FP8_ARG \
    $XSA_ARG \
    $PROFILE_ARG

echo "=== XSA overhead job complete: $(date) ==="
