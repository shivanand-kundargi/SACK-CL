#!/bin/bash

for seed in 0 1 2
do
    echo "Starting der run with $seed"
    python main.py \
        --dataset=seq-cub200 \
        --model=der \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --model_config=best \
        --sack 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-der-cub200-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=der-cub200-original-run-seed-$seed \
        --device='0,1' \
        --distributed="dp"        
    echo "completed der run with seed $seed"    
done
