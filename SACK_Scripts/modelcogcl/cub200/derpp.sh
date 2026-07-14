#!/bin/bash



for seed in 0 1 2
do

    echo "Starting derpp run with seed"
    python main.py \
        --dataset=seq-cub200 \
        --model=derpp \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --model_config=best \
        --sack 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-derpp-cub200-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=derpp-cub200-cogcl-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \       

    echo "completed derpp run with seed $seed"    
done

