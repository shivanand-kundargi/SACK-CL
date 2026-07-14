
#!/bin/bash


for seed in 0 1 2
do
    echo "Starting lwf run with seed $seed "
    python main.py \
        --dataset=seq-cub200 \
        --model=lwf \
        --lr=0.03\
        --backbone=resnet50 \
        --sack 0 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-lwf-cub200-cogcl-mammoth \
        --wandb_name=original-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=lwf-cub200-original-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \       
    echo "completed lwf run with seed $seed"    
done

