#!/bin/bash

export seed=0

python main.py \
        --dataset=seq-core50 \
        --model=lwf \
        --lr=0.03\
        --backbone=resnet50 \
        --cog_cl 1 \
        --batch_size=256 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-core50-SACK-mammoth \
        --wandb_name=-SACK_unifrom2weights \
        --enable_other_metrics=True \
        --permute_classes=True 



python main.py \
        --dataset=seq-core50 \
        --model=lwf \
        --lr=0.03\
        --backbone=resnet50 \
        --cog_cl 0 \
        --batch_size=128 \
        --sack_scores_type=0 \
        --num_workers=8 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-core50-SACK-mammoth \
        --wandb_name=-original_run \
        --enable_other_metrics=True \
        --permute_classes=True 

# python main.py \
#         --dataset=seq-core50 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 0 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-core50-SACK-mammoth \
#         --wandb_name=original-run-seed-$seed-Weights2uniform\
#         --enable_other_metrics=True \
#         --permute_classes=True \
#         --seed=$seed 



# python main.py \
#         --dataset=seq-core50 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-core50-SACK-mammoth \
#         --wandb_name=SACK-run-seed-$seed-Weights2uniform\
#         --enable_other_metrics=True \
#         --permute_classes=True \
#         --seed=$seed 






# python main.py \
#         --dataset=seq-core50 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-core50-SACK-mammoth \
#         --wandb_name=SACK-run-seed-$seed-uniform2weights\
#         --enable_other_metrics=True \
#         --permute_classes=True \
#         --seed=$seed 


# python main.py \
#         --dataset=seq-core50 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-core50-SACK-mammoth \
#         --wandb_name=SACK-run-seed-$seed-random274\
#         --enable_other_metrics=True \
#         --permute_classes=True \
#         --seed=$seed 