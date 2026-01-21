#!/usr/bin/env bash
set -euo pipefail

print_usage() {
    cat <<USAGE
Usage: $(basename "$0") <class_name> [extra gradcamviz.py args]

<class_name> matches a label from the selected dataset (default: CIFAR-100).
Names are case-insensitive; spaces, hyphens, and punctuation are normalized to underscores.

Environment overrides:
  MODEL_NAME     (default: icarl)
  DATASET        (default: seq-cifar100)
  DATASET_TAG    (default: cifar100 if dataset is seq-cifar100, cub200 if seq-cub200)
  SEED           (default: 0)
  CHECKPOINT_DIR (default: checkpoints)
  SPLIT          (default: test)
  SAMPLE_INDEX   (default: 0)
  LAYER_NAME     (default: layer4)
  BASELINE_TAG   (default: original)
  SACK_TAG       (default: sack)
  OUTPUT_DIR     (default: gradcam_outputs/<model>_<sanitized_class>)

Any additional arguments after <class_name> are forwarded to gradcamviz.py.
USAGE
}

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

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

CLASS_INPUT=$1
shift

MODEL_NAME=${MODEL_NAME:-icarl}
DATASET=${DATASET:-seq-cifar100}
if [[ -z ${DATASET_TAG:-} ]]; then
    case "$DATASET" in
        seq-cub200) DATASET_TAG="cub200" ;;
        *) DATASET_TAG="cifar100" ;;
    esac
fi
SEED=${SEED:-0}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
SPLIT=${SPLIT:-test}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}
LAYER_NAME=${LAYER_NAME:-layer4}
BASELINE_TAG=${BASELINE_TAG:-original}
SACK_TAG=${SACK_TAG:-sack}

CLASS_KEY=$(sanitize "$CLASS_INPUT")

declare -A CLASS_TO_INDEX=()

if [[ "$DATASET" == "seq-cifar100" ]]; then
    declare -A CLASS_TO_INDEX=(
        [apple]=0
        [aquarium_fish]=1
        [baby]=2
        [bear]=3
        [beaver]=4
        [bed]=5
        [bee]=6
        [beetle]=7
        [bicycle]=8
        [bottle]=9
        [bowl]=10
        [boy]=11
        [bridge]=12
        [bus]=13
        [butterfly]=14
        [camel]=15
        [can]=16
        [castle]=17
        [caterpillar]=18
        [cattle]=19
        [chair]=20
        [chimpanzee]=21
        [clock]=22
        [cloud]=23
        [cockroach]=24
        [couch]=25
        [crab]=26
        [crocodile]=27
        [cup]=28
        [dinosaur]=29
        [dolphin]=30
        [elephant]=31
        [flatfish]=32
        [forest]=33
        [fox]=34
        [girl]=35
        [hamster]=36
        [house]=37
        [kangaroo]=38
        [keyboard]=39
        [lamp]=40
        [lawn_mower]=41
        [leopard]=42
        [lion]=43
        [lizard]=44
        [lobster]=45
        [man]=46
        [maple_tree]=47
        [motorcycle]=48
        [mountain]=49
        [mouse]=50
        [mushroom]=51
        [oak_tree]=52
        [orange]=53
        [orchid]=54
        [otter]=55
        [palm_tree]=56
        [pear]=57
        [pickup_truck]=58
        [pine_tree]=59
        [plain]=60
        [plate]=61
        [poppy]=62
        [porcupine]=63
        [possum]=64
        [rabbit]=65
        [raccoon]=66
        [ray]=67
        [road]=68
        [rocket]=69
        [rose]=70
        [sea]=71
        [seal]=72
        [shark]=73
        [shrew]=74
        [skunk]=75
        [skyscraper]=76
        [snail]=77
        [snake]=78
        [spider]=79
        [squirrel]=80
        [streetcar]=81
        [sunflower]=82
        [sweet_pepper]=83
        [table]=84
        [tank]=85
        [telephone]=86
        [television]=87
        [tiger]=88
        [tractor]=89
        [train]=90
        [trout]=91
        [tulip]=92
        [turtle]=93
        [wardrobe]=94
        [whale]=95
        [willow_tree]=96
        [wolf]=97
        [woman]=98
        [worm]=99
    )
elif [[ "$DATASET" == "seq-cub200" ]]; then
    mapfile -t CUB_MAPPING < <(python - <<'PY'
import numpy as np
from pathlib import Path
path = Path("data/CUB200/train_data.npz")
if not path.exists():
    raise SystemExit("CUB-200 train_data.npz not found in data/CUB200/." )
data = np.load(path, allow_pickle=True)
classes = data["classes"]
for idx, (cid, name) in enumerate(classes):
    print(f"{idx}\t{name}")
PY
    )
    for entry in "${CUB_MAPPING[@]}"; do
        idx=${entry%%$'\t'*}
        raw=${entry#*$'\t'}
        key=$(sanitize "$raw")
        CLASS_TO_INDEX[$key]=$idx
    done
else
    echo "[ERROR] Unsupported dataset '$DATASET'." >&2
    exit 1
fi

if [[ ! -v CLASS_TO_INDEX[$CLASS_KEY] ]]; then
    echo "[ERROR] Unknown class '$CLASS_INPUT' for dataset '$DATASET'." >&2
    echo "Available classes:" >&2
    printf '  %s\n' "${!CLASS_TO_INDEX[@]}" | sort >&2
    exit 1
fi

CLASS_INDEX=${CLASS_TO_INDEX[$CLASS_KEY]}

if [[ -z ${OUTPUT_DIR:-} ]]; then
    OUTPUT_DIR="gradcam_outputs/${MODEL_NAME}_${CLASS_KEY}"
fi

CMD=(
    python gradcamviz.py
    --model-name "$MODEL_NAME"
    --dataset "$DATASET"
    --dataset-tag "$DATASET_TAG"
    --seed "$SEED"
    --target-class "$CLASS_INDEX"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --split "$SPLIT"
    --sample-index "$SAMPLE_INDEX"
    --layer-name "$LAYER_NAME"
    --baseline-tag "$BASELINE_TAG"
    --sack-tag "$SACK_TAG"
    --output-dir "$OUTPUT_DIR"
)

if [[ $# -gt 0 ]]; then
    CMD+=("$@")
fi

echo "[INFO] Running: ${CMD[*]}"
exec "${CMD[@]}"
