#!/bin/bash

export seed=0
python main.py \
    --dataset=seq-imagenet-r \
    --model=icarl \
    --buffer_size=2000 \
    --model_config=best \
    --sack 1 \
    --wandb_entity=abcxyz8431-cl \
    --wandb_project=Final-icarl-imagenet-r-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=icarl-imagenet-r-cogcl-run-seed-$seed 

python main.py \
    --dataset=seq-imagenet-r \
    --model=icarl \
    --buffer_size=2000 \
    --model_config=best \
    --sack 0 \
    --wandb_entity=abcxyz8431-cl \
    --wandb_project=Final-icarl-imagenet-r-cogcl-mammoth \
    --wandb_name=original-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=icarl-imagenet-r-cogcl-run-seed-$seed 