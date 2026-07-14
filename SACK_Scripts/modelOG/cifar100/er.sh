#!/bin/bash


for seed in 0 1 2
do
    echo "Starting ER run with $seed"
    python main.py \
        --dataset=seq-cifar100 \
        --model=er \
        --buffer_size=2000 \
        --lr=0.03 \
        --sack 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-er-cifar100-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=er-cifar100-original-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \      
    echo "completed ER run with seed $seed"    
done
