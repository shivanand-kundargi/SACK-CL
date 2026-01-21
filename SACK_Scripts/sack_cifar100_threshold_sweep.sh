#!/bin/bash

# Run SACK threshold sweeps for multiple methods on CIFAR-100 in parallel.
# Each method is pinned to a dedicated GPU and iterates over the list of
# similarity percentiles defined below.

set -euo pipefail

# Adjust if you want a different set of thresholds.
THRESHOLDS=(60 65 70 75 80 85)

# Change these defaults as needed (can also export them before calling the script).
SEED="${SEED:-0}"
SACK_SCORES_TYPE="${SACK_SCORES_TYPE:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-abcxyz8431-cl}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-Final}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
pushd "${REPO_ROOT}" >/dev/null

run_method () {
    local method="$1"
    local gpu="$2"
    local dataset="$3"
    shift 3
    local extra_args=("$@")

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        for percentile in "${THRESHOLDS[@]}"; do
            echo "[${method}] GPU=${gpu} percentile=${percentile}"
            python main.py \
                --dataset="${dataset}" \
                --model="${method}" \
                --cog_cl=1 \
                --sack_scores_type="${SACK_SCORES_TYPE}" \
                --sack_similarity_percentile="${percentile}" \
                --wandb_entity="${WANDB_ENTITY}" \
                --wandb_project="${WANDB_PROJECT_PREFIX}-${method}-cifar100-SACK" \
                --wandb_name="SACK-${method}-thr${percentile}-seed${SEED}" \
                --enable_other_metrics=True \
                --permute_classes=True \
                --log_perf_metrics=1 \
                --seed="${SEED}" \
                "${extra_args[@]}"
        done
    ) &
}

# GPU assignments (0-4). Adjust if your GPU order differs.
run_method icarl      0 seq-cifar100     --buffer_size=2000 --model_config=best
run_method lwf        1 seq-cifar100     --lr=0.003
run_method coda_prompt 2 seq-cifar100-224 --model_config=best
run_method der        3 seq-cifar100     --buffer_size=2000 --model_config=best
run_method derpp      4 seq-cifar100     --buffer_size=2000 --model_config=best

wait
popd >/dev/null
