#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
DATASET="seq-cub200"
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-gradcam_outputs}
CLASS_LIMIT=${CLASS_LIMIT:-20}
MODELS_ENV=${MODELS:-"icarl der derpp lwf coda_prompt"}
LAYERS_ENV=${LAYERS:-"layer4 layer3"}
REMAINING_ARGS=("$@")

read -r -a MODEL_LIST <<< "$MODELS_ENV"
read -r -a LAYER_LIST <<< "$LAYERS_ENV"

sanitize() {
    local input="$1"
    input="${input,,}"
    input="${input// /_}"
    input="${input//-/_}"
    input="${input//./_}"
    input="${input//[^a-z0-9_]/}"
    while [[ "$input" == *__* ]]; do
        input="${input//__/_}"
    done
    input="${input##_}"
    input="${input%%_}"
    echo "$input"
}

DATA_PATH="$SCRIPT_DIR/data/CUB200/train_data.npz"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "[ERROR] Expected dataset file not found at $DATA_PATH" >&2
    exit 1
fi

mapfile -t RAW_CLASSES < <(CLASS_LIMIT="$CLASS_LIMIT" DATA_PATH="$DATA_PATH" python - <<'PY'
import numpy as np
import os
from pathlib import Path
limit = int(os.environ.get("CLASS_LIMIT", "20"))
data_path = Path(os.environ["DATA_PATH"])
classes = np.load(data_path, allow_pickle=True)["classes"]
for _, name in classes[:limit]:
    print(name)
PY
)

if [[ ${#RAW_CLASSES[@]} -eq 0 ]]; then
    echo "[ERROR] Could not load class names." >&2
    exit 1
fi

for model in "${MODEL_LIST[@]}"; do
  for layer in "${LAYER_LIST[@]}"; do
    for raw_name in "${RAW_CLASSES[@]}"; do
      display_name="${raw_name//_/ }"
      class_key=$(sanitize "$display_name")
      output_dir="${ROOT_OUTPUT_DIR}/${DATASET}/${model}/${layer}/${class_key}"
      mkdir -p "$output_dir"
      echo "[INFO] Running model $model layer $layer class '$display_name'"
      (cd "$SCRIPT_DIR" && DATASET="$DATASET" MODEL_NAME="$model" OUTPUT_DIR="$output_dir" \
        ./rungradcamviz.sh "$display_name" --layer-name "$layer" --verbose "${REMAINING_ARGS[@]}")
    done
  done
done
