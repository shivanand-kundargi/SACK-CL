#!/bin/bash


for seed in 0 1 2
do
    echo "Starting GEM run with seed $seed"
    python main.py \
        --dataset=seq-cub200 \
        --model=gem \
        --buffer_size=2000 \
        --lr=0.03 \
        --cog_cl 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-gem-cub200-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=gem-cub200-original-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dpp" \
    echo "completed GEM run with seed $seed"

done