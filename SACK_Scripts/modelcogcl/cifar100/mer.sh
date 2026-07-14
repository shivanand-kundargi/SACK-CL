#!/bin/bash



for seed in 0 1 2
do

    echo "Starting mer run with seed $seed"
    python main.py \
        --dataset=seq-cifar100 \
        --model=mer \
        --buffer_size=2000 \
        --minibatch_size 25 \
        --n_epochs 5 \
        --lr=0.1 \
        --beta=0.01 \
        --gamma=0.03 \
        --sack 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-mer-cifar100-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=mer-cifar100-cogcl-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \ 
    echo "completed mer run with seed $seed "          
done