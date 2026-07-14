#!/bin/bash


for seed in 0 1 2
do
    echo "Starting bic run with seed  $seed"    
    python main.py \
        --dataset=seq-cifar100 \
        --model=bic \
        --buffer_size=2000 \
        --model_config=best \
        --sack 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-bic-cifar100-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=bic-cifar100-cogcl-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \      
    echo "completed bic run with seed $seed"    
done