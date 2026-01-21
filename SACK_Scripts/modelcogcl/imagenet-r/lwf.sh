
#!/bin/bash


for seed in 0 1 2
do
    echo "Starting lwf run with seed $seed "
    python main.py \
        --dataset=seq-imagenet-r \
        --model=lwf \
        --lr=0.03\
        --backbone=resnet50 \
        --cog_cl 1 \
        --wandb_entity=shiva-umbc \
        --wandb_project=Final-lwf-imagenet-r-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed \
        --ckpt_name=lwf-imagenet-r-cogcl-run-seed-$seed \
        # --device='0,1,2,3' \
        # --distributed="dp" \       
    echo "completed lwf run with seed $seed"    
done

