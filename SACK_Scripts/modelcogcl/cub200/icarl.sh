#!/bin/bash

for seed in 0 1 2
do
    echo "Starting iCaRL run with seed $seed"
    python main.py \
        --dataset=seq-cub200 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-icarl-cub200-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=icarl-cub200-cogcl-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \


    echo "completed iCaRL run with seed $seed"    
done