#!/bin/bash
# Evaluate continual-learning checkpoints on CIFAR-100-C corruptions.
#
# Before running, export the checkpoint locations for each baseline/SACK model:
#   export ICARL_BASELINE_CKPT=/path/to/icarl/original_final.pt
#   export ICARL_SACK_CKPT=/path/to/icarl/sack_final.pt
# (repeat for LWF, DER, DERPP, CODA_PROMPT). Any method without a checkpoint is skipped.
#
# Optionally adjust WANDB_ENTITY/PROJECT_PREFIX/SACK_SCORES_TYPE/SEED below.

set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
: "${WANDB_ENTITY:=abcxyz8431-cl}"
: "${WANDB_PROJECT_PREFIX:=cifar100c-robustness}"
: "${SACK_SCORES_TYPE:=0}"
: "${SEED:=0}"

CORRUPTIONS=(
  gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur
  snow frost fog brightness contrast elastic_transform pixelate jpeg_compression
  gaussian_blur saturate spatter speckle_noise
)
SEVERITIES=(1 2 3 4 5)
METHODS=(icarl der derpp lwf coda_prompt)

declare -A EXTRA_ARGS=(
  [icarl]="--buffer_size=2000 --model_config=best"
  [der]="--buffer_size=2000 --model_config=best"
  [derpp]="--buffer_size=2000 --model_config=best"
  [lwf]="--lr=0.003"
  [coda_prompt]="--model_config=best"
)

declare -A DATASET_FOR_METHOD=(
  [coda_prompt]="seq-cifar100-c-224"
)

# Allow overriding checkpoint paths via per-method environment variables.
declare -A BASELINE_CKPTS=(
  [icarl]="${ICARL_BASELINE_CKPT:-}"
  [der]="${DER_BASELINE_CKPT:-}"
  [derpp]="${DERPP_BASELINE_CKPT:-}"
  [lwf]="${LWF_BASELINE_CKPT:-}"
  [coda_prompt]="${CODA_PROMPT_BASELINE_CKPT:-}"
)

declare -A SACK_CKPTS=(
  [icarl]="${ICARL_SACK_CKPT:-}"
  [der]="${DER_SACK_CKPT:-}"
  [derpp]="${DERPP_SACK_CKPT:-}"
  [lwf]="${LWF_SACK_CKPT:-}"
  [coda_prompt]="${CODA_PROMPT_SACK_CKPT:-}"
)

run_suite() {
  local tag="$1"
  local cog_flag="$2"
  local ckpt_map_name="$3"
  declare -n ckpts="$ckpt_map_name"

  for method in "${METHODS[@]}"; do
    local ckpt="${ckpts[$method]:-}"
    if [[ -z "$ckpt" ]]; then
      echo "[$tag/$method] Skipping (checkpoint not set)." >&2
      continue
    fi

    if [[ ! -f "$ckpt" ]]; then
      echo "[$tag/$method] Checkpoint '$ckpt' not found." >&2
      continue
    fi

    local dataset="${DATASET_FOR_METHOD[$method]:-seq-cifar100-c}"
    local extra="${EXTRA_ARGS[$method]:-}"

    for corruption in "${CORRUPTIONS[@]}"; do
      for severity in "${SEVERITIES[@]}"; do
        echo "[${tag}] method=${method} corruption=${corruption} severity=${severity}"
        python main.py \
          --dataset="${dataset}" \
          --model="${method}" \
          --inference_only=1 \
          --loadcheck="${ckpt}" \
          --cifar_c_corruption="${corruption}" \
          --cifar_c_severity="${severity}" \
          --cog_cl="${cog_flag}" \
          --sack_scores_type="${SACK_SCORES_TYPE}" \
          --wandb_entity="${WANDB_ENTITY}" \
          --wandb_project="${WANDB_PROJECT_PREFIX}-${tag}-${method}" \
          --wandb_name="${tag}-${method}-${corruption}-s${severity}" \
          --enable_other_metrics=True \
          --log_perf_metrics=1 \
          --permute_classes=True \
          --seed="${SEED}" \
          ${extra}
      done
    done
  done
}

run_suite baseline 0 BASELINE_CKPTS
run_suite sack 1 SACK_CKPTS
