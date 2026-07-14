#!/bin/bash

for seed in 0 1 2
do
    echo "Starting iCaRL run with seed $seed"
    python main.py \
        --dataset=seq-cifar10 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --sack 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-icarl-cifar10-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=icarl-cifar10-cogcl-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \


    echo "completed iCaRL run with seed $seed"    
done



$seed=0

echo "Starting der run with $seed"
python main.py \
    --dataset=seq-cifar10 \
    --model=der \
    --buffer_size=2000 \
    --model_config=best \
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-der-cifar10-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=der-cifar100-cogcl-run-seed-$seed \
    # --device='0,1,2,3' \
    # --distributed="dp" \       
echo "completed der run with seed $seed"    







echo "Starting derpp run with seed"
python main.py \
    --dataset=seq-cifar10 \
    --model=derpp \
    --buffer_size=2000 \
    --model_config=best \
    --sack 0 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-derpp-cifar10-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=derpp-cifar10-cogcl-run-seed-$seed \
    # --device='0,1,2,3' \
    # --distributed="dp" \       

echo "completed derpp run with seed $seed"    




echo "Starting lwf run with seed $seed "
python main.py \
    --dataset=seq-cifar10 \
    --model=lwf \
    --lr=0.03\
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-lwf-cifar10-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=lwf-cifar10-cogcl-run-seed-$seed \
    # --device='0,1,2,3' \
    # --distributed="dp" \       
echo "completed lwf run with seed $seed"    








echo "Starting coda-prompt run with seed $seed"
python main.py \
    --dataset=seq-cifar10 \
    --model=coda_prompt \
    --model_config=best\
    --sack 1 \
    --wandb_entity=shiva-umbc \
    --wandb_project=Final-coda_prompt-cifar10-cogcl-mammoth \
    --wandb_name=cogcl-run-seed-$seed \
    --enable_other_metrics=True \
    --savecheck=task \
    --permute_classes=True \
    --seed=$seed \
    --ckpt_name=codaprompt-cifar10-cogcl-run-seed-$seed \
    # --device='0,1,2,3' \
    # --distributed="dp" \       

echo "completed coda-prompt run with seed $seed"    


