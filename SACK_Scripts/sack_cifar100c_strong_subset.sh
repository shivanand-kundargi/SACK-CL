#!/bin/bash
# Evaluate CL baselines vs SACK on selected "strong" CIFAR-100-C corruptions.
# Corruptions: gaussian_noise, motion_blur, frost, contrast, jpeg_compression
# Severities: 1 (mild), 3 (medium), 5 (severe)
# Methods: icarl, lwf, der, derpp, coda_prompt (each baseline + SACK variant)
#
# Checkpoints are auto-discovered from ./checkpoints (last experience, suffix *_9.pt).
# You can still override by exporting ICARL_BASELINE_CKPT, etc.
# Optional env vars: WANDB_ENTITY, WANDB_PROJECT_PREFIX, SACK_SCORES_TYPE, SEED.

set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-online}"
: "${WANDB_ENTITY:=abcxyz8431-cl}"
: "${WANDB_PROJECT_PREFIX:=cifar100c-strong}"
: "${SACK_SCORES_TYPE:=0}"
: "${SEED:=0}"

CORRUPTIONS=(gaussian_noise motion_blur frost contrast jpeg_compression)
SEVERITIES=(1 3 5)
METHODS=(icarl lwf der derpp coda_prompt)

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

# Baseline checkpoints (no SACK)
declare -A BASELINE_PATTERNS=(
  [icarl]="icarl-cifar100-original"
  [lwf]="lwf-cifar100-original"
  [der]="der-cifar100-original"
  [derpp]="derpp-cifar100-original"
  [coda_prompt]="coda_prompt-cifar100-original"
)

declare -A SACK_PATTERNS=(
  [icarl]="icarl-cifar100-sack"
  [lwf]="lwf-cifar100-sack"
  [der]="der-cifar100-sack"
  [derpp]="derpp-cifar100-sack"
  [coda_prompt]="coda_prompt-cifar100-sack"
)

find_checkpoint() {
  local override="$1"
  local pattern="$2"
  if [[ -n "$override" ]]; then
    if [[ -f "$override" ]]; then
      echo "$override"
      return
    else
      echo "[warn] override '$override' not found; falling back to auto-discovery" >&2
    fi
  fi
  local latest
  latest=$(ls -1 checkpoints/"${pattern}"*"_9.pt" 2>/dev/null | sort | tail -n1 || true)
  echo "$latest"
}

declare -A BASELINE_CKPTS
declare -A SACK_CKPTS

for method in "${METHODS[@]}"; do
  local_override="${method^^}_BASELINE_CKPT"
  BASELINE_CKPTS["$method"]=$(find_checkpoint "${!local_override:-}" "${BASELINE_PATTERNS[$method]}")
  local_override="${method^^}_SACK_CKPT"
  SACK_CKPTS["$method"]=$(find_checkpoint "${!local_override:-}" "${SACK_PATTERNS[$method]}")
done

run_suite() {
  local tag="$1"
  local sack_flag="$2"
  local ckpt_assoc_name="$3"
  declare -n ckpts="$ckpt_assoc_name"

  for method in "${METHODS[@]}"; do
    local ckpt="${ckpts[$method]:-}"
    if [[ -z "$ckpt" ]]; then
      echo "[$tag/$method] Skipping: checkpoint not set." >&2
      continue
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "[$tag/$method] Missing checkpoint file: $ckpt" >&2
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
          --sack="${sack_flag}" \
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
