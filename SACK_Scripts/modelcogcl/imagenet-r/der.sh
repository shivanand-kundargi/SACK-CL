#!/bin/bash

for seed in 0 1 2
do
    echo "Starting der run with $seed"
    python main.py \
        --dataset=seq-imagenet-r \
        --model=der \
        --buffer_size=2000 \
d        --backbone=resnet50 \
        --model_config=best \
        --cog_cl 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-der-imagenet-r-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=der-imagenet-r-cogcl-run-seed-$seed \
        # --device='0,1' \
        # --distributed="dp"        
    echo "completed der run with seed $seed"    
done
