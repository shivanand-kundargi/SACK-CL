#!/bin/bash

for seed in 0 1 2
do

    echo "Starting coda-prompt run with seed $seed"
    python main.py \
        --dataset=seq-cub200 \
        --model=coda_prompt \
        --model_config=best\
        --sack 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-coda_prompt-cub200-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=codaprompt-cub200-cogcl-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \       

    echo "completed coda-prompt run with seed $seed"    
done

