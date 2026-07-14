#!/usr/bin/env bash
set -Eeuo pipefail

# Runs SACK iCaRL experiments on sequential CIFAR-100 with and without
# uncertainty-based sampling.
#
# Usage:
#   ./scripts/SACK_CIFAR100_uncertainity.sh
# Optional environment variables:
#   SEEDS="0,1"                       # Comma-separated list of seeds
#   MODEL="icarl"                     # Backbone continual model to run
#   BUFFER_SIZE="2000"                # Replay buffer size
#   BATCH_SIZE="128"                  # Training batch size
#   WANDB_ENTITY="my-entity"          # WandB entity override
#   WANDB_PROJECT="my-project"        # WandB project override
#   WANDB_GROUP="SACK-CIFAR100"       # WandB group tag
#   NUM_WORKERS="8"                   # DataLoader workers (optional)
#   UNCERTAINTY_EVAL_BATCH_SIZE="64"  # Batch size for uncertainty scoring

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS_ENV="${SEEDS:-1}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS_ENV}"

MODELS_ENV="${MODEL:-coda-prompt}"
IFS=',' read -r -a MODEL_LIST <<< "${MODELS_ENV}"
BUFFER_SIZE="${BUFFER_SIZE:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SACK_SCORES_TYPE="${SACK_SCORES_TYPE:-0}"

WANDB_ENTITY="${WANDB_ENTITY:-abcxyz8431-cl}"
WANDB_PROJECT="${WANDB_PROJECT:-Final-icarl-cifar100-SACK}"


NUM_WORKERS_FLAG=()
if [[ -n "${NUM_WORKERS:-}" ]]; then
    NUM_WORKERS_FLAG=(--num_workers="${NUM_WORKERS}")
fi

UNC_BATCH_FLAG=()
if [[ -n "${UNCERTAINTY_EVAL_BATCH_SIZE:-}" ]]; then
    UNC_BATCH_FLAG=(--uncertainty_eval_batch_size="${UNCERTAINTY_EVAL_BATCH_SIZE}")
fi

for model in "${MODEL_LIST[@]}"; do
    # Trim whitespace from model name
    model=$(echo "$model" | xargs)
    
    for seed in "${SEED_LIST[@]}"; do

        echo "[info] Starting uncertainty-based SACK run (model=${model}, seed=${seed})"
        python "${ROOT_DIR}/main.py" \
            --dataset=seq-cifar100-224 \
            --model="${model}" \
            --model_config=best \
            --sack=1 \
            --sack_scores_type="${SACK_SCORES_TYPE}" \
            --batch_size="${BATCH_SIZE}" \
            --enable_other_metrics=True \
            --permute_classes=True \
            --wandb_entity="${WANDB_ENTITY}" \
            --wandb_project="${WANDB_PROJECT}" \
            --wandb_name="SACK-${model}-seed${seed}-uncertainty" \
            --seed="${seed}" \
            --use_uncertainty_sampling=1 \
            "${NUM_WORKERS_FLAG[@]}" \
            "${UNC_BATCH_FLAG[@]}"



        echo "[info] Starting baseline SACK run (weighted sampling, model=${model}, seed=${seed})"
        python "${ROOT_DIR}/main.py" \
            --dataset=seq-cifar100-224 \
            --model="${model}" \
            --model_config=best \
            --sack=1 \
            --sack_scores_type="${SACK_SCORES_TYPE}" \
            --batch_size="${BATCH_SIZE}" \
            --enable_other_metrics=True \
            --permute_classes=True \
            --wandb_entity="${WANDB_ENTITY}" \
            --wandb_project="${WANDB_PROJECT}" \
            --wandb_name="SACK-${model}-seed${seed}-concept-score-weighted" \
            --seed="${seed}" \
            --use_uncertainty_sampling=0 \
            "${NUM_WORKERS_FLAG[@]}"
        
    done
done
