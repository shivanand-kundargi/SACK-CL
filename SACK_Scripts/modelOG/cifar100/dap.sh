#!/bin/bash


for seed in 0 1 2
do

    echo "Starting dap run with seed $seed"
    python main.py \
        --dataset=seq-cifar100 \
        --model=dap \
        --model_config=best\
        --sack 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-dap-cifar100-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=dap-cifar100-original-run-seed-$seed  \
        --device='0,1' \
        --distributed="dp"       

    echo "completed dap run with seed $seed"    
done

