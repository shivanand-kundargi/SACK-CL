#!/bin/bash


for seed in 0 1 2
do

    echo "Starting coda-prompt run with seed $seed"
    python main.py \
        --dataset=seq-cifar100-224 \
        --model=coda-prompt \
        --model_config=best\
        --cog_cl 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-coda_prompt-cifar100-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --permute_classes=True \
        --savecheck=task \
        --device='0,1' \
        --distributed="dp" \
        # --ckpt_name=codaprompt-cifar100-original-run-seed-$seed \      

    echo "completed coda-prompt run with seed $seed"    
done