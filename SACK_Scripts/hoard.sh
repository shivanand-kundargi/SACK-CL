#!/bin/bash

while true; do
  python main.py \
    --dataset=seq-cifar100 \
    --model=icarl \
    --buffer_size=2000 \
    --model_config=best \
    --cog_cl 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=hoarding \
    --wandb_name=hoarding \
    --enable_other_metrics=True \
    --permute_classes=True 
done