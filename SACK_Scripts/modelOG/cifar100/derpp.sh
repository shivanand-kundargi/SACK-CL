#!/bin/bash


for seed in 0 1 2
do

    echo "Starting derpp run with seed"
    python main.py \
        --dataset=seq-cifar100 \
        --model=derpp \
        --buffer_size=2000 \
        --model_config=best \
        --sack 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-derpp-cifar100-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=derpp-cifar100-original-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \       

    echo "completed derpp run with seed $seed"    
done
