#!/bin/bash

## Runs SACK experiments on the 1000-class sequential iNaturalist dataset.
SEEDS=(0)
WANDB_ENTITY="abcxyz8431-cl"
WANDB_PROJECT="Final-icarl-inaturalist1000-SACK"

BASE_PATH_OVERRIDE="${BASE_PATH_OVERRIDE:-./data/}"
BASE_PATH_OVERRIDE="${BASE_PATH_OVERRIDE%/}/"
DATA_ROOT="${BASE_PATH_OVERRIDE}inaturalist1000"

echo "Using base path: ${BASE_PATH_OVERRIDE}"
echo "Expected metadata root: ${DATA_ROOT}"
echo "Ensure metadata exists via: python SACK/scripts/download_inaturalist_1000.py --root ${DATA_ROOT}"

for seed in "${SEEDS[@]}"; do
        echo "Starting iNaturalist-1000 iCaRL run with seed ${seed}"
        BASE_PATH="${BASE_PATH_OVERRIDE}" \
        python main.py \
                --dataset=seq-inaturalist-1000 \
                --model=icarl \
                --buffer_size=20000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity="${WANDB_ENTITY}" \
                --wandb_project="${WANDB_PROJECT}" \
                --wandb_name=SACK1000-${seed} \
                --enable_other_metrics=True \
                --permute_classes=True \
                --seed="${seed}" \
                --base_path="${BASE_PATH_OVERRIDE}"
done
