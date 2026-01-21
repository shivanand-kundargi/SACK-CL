#!/bin/bash

# Runs SACK experiments on the sequential iNaturalist dataset limited to the first 300 classes.
SEEDS=(0 1 2)
WANDB_ENTITY="abcxyz8431-cl"

for seed in "${SEEDS[@]}"; do
        echo "Starting iNaturalist-300 iCaRL run with seed $seed (CoG-CL disabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=coda_prompt \
                --model_config=best \
                --cog_cl 0 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-coda-prompt-inaturalist300-SACK \
                --wandb_name=original-inat300-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --seed="$seed"

        # echo "Starting iNaturalist-300 iCaRL run with seed $seed (CoG-CL enabled)"
        # python main.py \
        #         --dataset=seq-inaturalist-300 \
        #         --model=icarl \
        #         --buffer_size=2000 \
        #         --model_config=best \
        #         --cog_cl 1 \
        #         --sack_scores_type=0 \
        #         --wandb_entity="$WANDB_ENTITY" \
        #         --wandb_project=Final-icarl-inaturalist300-SACK \
        #         --wandb_name=SACKKK-300-$seed \
        #         --enable_other_metrics=True \
        #         --permute_classes=True \
        #         --seed="$seed"
done
