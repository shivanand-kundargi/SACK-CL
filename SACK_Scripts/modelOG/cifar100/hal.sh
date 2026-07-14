#!/bin/bash


for seed in 0 1 2
do
    echo "Starting hal run with seed $seed"
    python main.py \
        --dataset=seq-cifar100 \
        --model=hal \
        --buffer_size=2000 \
        --hal_lambda=0.1 \
        --lr=0.03 \
        --beta=0.3 \
        --gamma=0.1 \
        --sack 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-hal-cifar100-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --permute_classes=True \
        --seed=$seed \
        # --ckpt_name=hal-cifar100-original-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \


    echo "completed hal run with seed $seed "    
done