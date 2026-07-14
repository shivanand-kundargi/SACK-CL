#!/bin/bash


for seed in 0 1 2
do
    echo "Starting gss run with seed $seed"
    python main.py \
        --dataset=seq-cifar100 \
        --model=gss \
        --buffer_size=2000 \
        --n_epochs=5 \
        --lr=0.05 \
        --gss_minibatch_size=128 \
        --batch_size=128 \
        --sack 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-gss-cifar100-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --permute_classes=True \
        --seed=$seed \
        # --ckpt_name=gss-cifar100-cogcl-run-seed-$seed \
        # --savecheck=task \
        # --device='0,1,2,3' \
        # --distributed="dp" \      

    echo "completed gss run with seed $seed"    
done

