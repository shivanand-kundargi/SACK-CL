#!/bin/bash


seed=0

# python main.py \
#     --dataset=perm-mnist \
#     --model=lwf \
#     --lr=0.03\
#     --n_epochs=10 \
#     --dataset_config=10tasks \
#     --sack 1 \
#     --wandb_entity=shiva-umbc \
#     --wandb_project=Final-lwf-perm-mnist-cogcl-mammoth \
#     --wandb_name=cogcl-run-seed-$seed \
#     --enable_other_metrics=True \
#     --savecheck=task \
#     --permute_classes=True \
#     --seed=$seed \
#     --ckpt_name=lwf-perm-mnist-cogcl-run-seed-$seed 
     
python main.py \
    --dataset=perm-mnist \
    --model=icarl \
    --lr=0.03\
    --n_epochs=10 \
    --buffer_size=2000 \
    --dataset_config=10tasks \
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-icarl-perm-mnist-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=icarl-perm-mnist-cogcl-run-seed-$seed 

python main.py \
    --dataset=perm-mnist \
    --model=der \
    --lr=0.03\
    --n_epochs=10 \
    --buffer_size=2000 \
    --dataset_config=10tasks \
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-der-perm-mnist-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=der-perm-mnist-cogcl-run-seed-$seed 

python main.py \
    --dataset=perm-mnist \
    --model=derpp \
    --lr=0.03\
    --n_epochs=10 \
    --buffer_size=2000 \
    --dataset_config=10tasks \
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-derpp-perm-mnist-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=derpp-perm-mnist-cogcl-run-seed-$seed 

python main.py \
    --dataset=perm-mnist \
    --model=coda_prompt \
    --lr=0.03\
    --n_epochs=10 \
    --dataset_config=10tasks \
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-coda_prompt-perm-mnist-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=coda_prompt-perm-mnist-cogcl-run-seed-$seed 