#!/bin/bash

# Pipeline:
#   3x baseline timing runs
#   3x XSA timing runs
#   1 baseline profile run 
#   1 XSA profile run
#
# Jobs run sequentially in alternating timing order:
#   baseline, xsa, baseline, xsa, baseline, xsa, baseline_profile, xsa_profile
#
# Usage (from repo root):
#   bash runs/overhead/pace_submit_xsa_overhead_experiment.sh

set -e
cd "$HOME/scratch/nanochat"
mkdir -p runs/logs

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/scratch/nanochat}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-xsa_overhead_d24_$(date +%Y%m%d_%H%M%S)}"
export RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_BASE_DIR/$EXPERIMENT_NAME}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-80}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1048576}"
export WANDB_RUN="${WANDB_RUN:-dummy}"
export FP8="${FP8:-TRUE}"
export XSA_ALPHA="${XSA_ALPHA:-1.0}"
export XSA_LAYER_INDICES="${XSA_LAYER_INDICES:-}"

echo "=== Submitting XSA overhead experiment ==="

JOB1=$(LABEL=baseline_timing_1 TYPE=baseline PROFILE=FALSE sbatch --parsable \
    --job-name=baseline_timing_1 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "baseline_timing_1 submitted: job $JOB1"

JOB2=$(LABEL=xsa_timing_1 TYPE=xsa PROFILE=FALSE sbatch --parsable \
    --dependency=afterany:$JOB1 --job-name=xsa_timing_1 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "xsa_timing_1 submitted: job $JOB2"

JOB3=$(LABEL=baseline_timing_2 TYPE=baseline PROFILE=FALSE sbatch --parsable \
    --dependency=afterany:$JOB2 --job-name=baseline_timing_2 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "baseline_timing_2 submitted: job $JOB3"

JOB4=$(LABEL=xsa_timing_2 TYPE=xsa PROFILE=FALSE sbatch --parsable \
    --dependency=afterany:$JOB3 --job-name=xsa_timing_2 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "xsa_timing_2 submitted: job $JOB4"

JOB5=$(LABEL=baseline_timing_3 TYPE=baseline PROFILE=FALSE sbatch --parsable \
    --dependency=afterany:$JOB4 --job-name=baseline_timing_3 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "baseline_timing_3 submitted: job $JOB5"

JOB6=$(LABEL=xsa_timing_3 TYPE=xsa PROFILE=FALSE sbatch --parsable \
    --dependency=afterany:$JOB5 --job-name=xsa_timing_3 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "xsa_timing_3 submitted: job $JOB6"

JOB7=$(NUM_ITERATIONS=40 LABEL=baseline_profile_1 TYPE=baseline PROFILE=TRUE sbatch --parsable \
    --dependency=afterany:$JOB6 --job-name=baseline_profile_1 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "baseline_profile_1 submitted: job $JOB7"

JOB8=$(NUM_ITERATIONS=40 LABEL=xsa_profile_1 TYPE=xsa PROFILE=TRUE sbatch --parsable \
    --dependency=afterany:$JOB7 --job-name=xsa_profile_1 --export=ALL \
    runs/overhead/pace_xsa_overhead_job.sh)
echo "xsa_profile_1 submitted: job $JOB8"

echo ""
echo "All jobs queued. Results will be in:"
echo "  $RESULTS_DIR"
echo ""
echo "To cancel everything:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6 $JOB7 $JOB8"
