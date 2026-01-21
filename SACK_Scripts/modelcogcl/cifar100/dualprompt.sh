#!/bin/bash


for seed in 0 1 2
do

    echo "Starting dualprompt run with seed $seed"
    python main.py \
        --dataset=seq-cifar100-224 \
        --model=dualprompt \
        --model_config=best\
        --cog_cl 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-dualprompt-cifar100-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --device='1'  
     

    echo "completed dualprompt run with seed $seed"    
done
