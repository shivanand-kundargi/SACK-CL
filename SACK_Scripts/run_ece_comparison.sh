#!/bin/bash
# Compare per-experience ECE between original and SACK variants for CIFAR-100 runs.
# Usage: ./run_ece_comparison.sh [seed]
# Optional environment overrides:
#   CHECKPOINT_DIR - directory containing Mammoth checkpoints (default: checkpoints)
#   OUTPUT_ROOT    - where to store logits + summaries (default: ece_comparisons)
#   PYTHON_BIN     - python executable to use (default: python)

set -euo pipefail

seed="${1:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
checkpoint_dir="${CHECKPOINT_DIR:-checkpoints}"
output_root="${OUTPUT_ROOT:-ece_comparisons}"

mkdir -p "${output_root}"

# Model-specific configuration: dataset, additional CLI args, and checkpoint name root.
declare -A DATASET_MAP=(
    [icarl]="seq-cifar100"
    [der]="seq-cifar100"
    [derpp]="seq-cifar100"
    [lwf]="seq-cifar100"
    [coda_prompt]="seq-cifar100-224"
)

declare -A EXTRA_ARGS_MAP=(
    [icarl]="--buffer_size=2000 --model_config=best"
    [der]="--buffer_size=2000 --model_config=best"
    [derpp]="--buffer_size=2000 --model_config=best"
    [lwf]="--lr=0.003"
    [coda_prompt]="--model_config=best"
)

declare -A CKPT_ROOT_MAP=(
    [icarl]="icarl-cifar100"
    [der]="der-cifar100"
    [derpp]="derpp-cifar100"
    [lwf]="lwf-cifar100"
    [coda_prompt]="coda_prompt-cifar100"
)

common_args=(
    --sack_scores_type=0
    --permute_classes=True
    --seed="${seed}"
)

run_variant() {
    local model="$1"
    local variant_label="$2"   # "original" or "sack"
    local sack_flag="$3"

    local dataset="${DATASET_MAP[$model]}"
    local ckpt_root="${CKPT_ROOT_MAP[$model]}"
    local ckpt_prefix="${ckpt_root}-${variant_label}-seed-${seed}"
    local extra_args=( )

    if [[ -n "${EXTRA_ARGS_MAP[$model]}" ]]; then
        # shellcheck disable=SC2206
        extra_args=( ${EXTRA_ARGS_MAP[$model]} )
    fi

    local logits_dir="${output_root}/${model}/${variant_label}/logits"
    local summary_path="${output_root}/${model}/${variant_label}/ece_summary.json"
    mkdir -p "${logits_dir}"

    echo "[INFO] Running ${variant_label^^} ECE for ${model} (seed ${seed})"
    "${PYTHON_BIN}" SACK_ECE.py \
        --dataset="${dataset}" \
        --model="${model}" \
        --sack="${sack_flag}" \
        "${common_args[@]}" \
        "${extra_args[@]}" \
        --checkpoint_dir="${checkpoint_dir}" \
        --checkpoint_prefix="${ckpt_prefix}" \
        --logits_dir="${logits_dir}" \
        --summary_path="${summary_path}" \
        --validation_mode=complete
}

models=(icarl der derpp lwf coda_prompt)

for model in "${models[@]}"; do
    run_variant "${model}" "original" 0
    run_variant "${model}" "sack" 1
    echo
done

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

output_root = Path(os.environ.get('OUTPUT_ROOT', 'ece_comparisons'))
models = ['icarl', 'der', 'derpp', 'lwf', 'coda_prompt']

final = {}

for model in models:
    original_path = output_root / model / 'original' / 'ece_summary.json'
    sack_path = output_root / model / 'sack' / 'ece_summary.json'
    if not original_path.exists() or not sack_path.exists():
        continue

    with original_path.open() as f:
        original = json.load(f)
    with sack_path.open() as f:
        sack = json.load(f)

    orig_map = {entry['experience']: entry['ece'] for entry in original['experiences']}
    sack_map = {entry['experience']: entry['ece'] for entry in sack['experiences']}
    experiences = sorted(set(orig_map) | set(sack_map))

    print(f"Model: {model}")
    print("exp\toriginal\tsack\tdelta")
    per_exp = []
    for exp in experiences:
        orig_ece = orig_map.get(exp, float('nan'))
        sack_ece = sack_map.get(exp, float('nan'))
        delta = sack_ece - orig_ece
        print(f"{exp}\t{orig_ece:.6f}\t{sack_ece:.6f}\t{delta:+.6f}")
        per_exp.append({
            "experience": int(exp),
            "original": float(orig_ece),
            "sack": float(sack_ece),
            "delta": float(delta),
        })
    print()

    final[model] = {
        "weighted_ece": {
            "original": float(original.get("weighted_ece", float('nan'))),
            "sack": float(sack.get("weighted_ece", float('nan'))),
            "delta": float(sack.get("weighted_ece", float('nan')) - original.get("weighted_ece", float('nan')))
        },
        "per_experience": per_exp,
    }

# Save aggregated comparison to final_logs.json
out_path = output_root / 'final_logs.json'
with out_path.open('w') as f:
    json.dump(final, f, indent=2)
print(f"Saved aggregated comparisons to {out_path}")
PY
