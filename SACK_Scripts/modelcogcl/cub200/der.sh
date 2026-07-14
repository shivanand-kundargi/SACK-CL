#!/bin/bash


# python main.py \
#     --dataset=seq-cub200 \
#     --model=der \
#     --buffer_size=500 \
#     --backbone=resnet50 \
#     --model_config=best \
#     --sack 0 \
#     --wandb_entity=shiva-umbc \
#     --wandb_project=Final-der-cub200-cogcl-mammoth \
#     --wandb_name=original-run-seed-exemplar500 \
#     --enable_other_metrics=True \
#     --savecheck=task \
#     --permute_classes=True \
#     --ckpt_name=der-cub200-original-run-seed-exemplar500


python main.py \
    --dataset=seq-cub200 \
    --model=der \
    --backbone=resnet50 \
    --buffer_size=500 \
    --model_config=best \
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-der-cub200-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-exempplar500 \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --ckpt_name=der-cub200-cogcl-run-seed-exemplar500



