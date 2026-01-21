#!/bin/bash

for seed in 0 1 2
do

    echo "Starting derpp run with seed"
    python main.py \
        --dataset=seq-imagenet-r \
        --model=derpp \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --model_config=best \
        --cog_cl 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-derpp-imagenet-r-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=derpp-imagenet-r-original-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \       

    echo "completed derpp run with seed $seed"    
done
