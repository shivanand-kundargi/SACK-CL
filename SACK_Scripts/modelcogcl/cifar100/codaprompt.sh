#!/bin/bash


$seed=0
python main.py \
    --dataset=seq-cifar100-224 \
    --model=coda_prompt \
    --model_config=best\
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-coda_prompt-cifar100-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-weighted-loss \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --ckpt_name=codaprompt-cifar100-cogcl-run-seed
