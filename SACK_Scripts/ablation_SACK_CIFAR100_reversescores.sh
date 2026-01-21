#!/bin/bash

# export seed=0
# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cifar100-cogcl-mammoth \
#         --wandb_name=reversed-weights-sack-run-seed-$seed-Uniform2Weights\
#         --enable_other_metrics=True \
#         --permute_classes=True 


python main.py \
        --dataset=seq-cifar100 \
        --model=der \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-cifar100-cogcl-mammoth \
        --wandb_name=reversed-weights-sack-run-seed-$seed-Uniform2Weights \
        --enable_other_metrics=True \
        --permute_classes=True 

python main.py \
        --dataset=seq-cifar100 \
        --model=derpp \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-cifar100-cogcl-mammoth \
        --wandb_name=reversed-weights-sack-run-seed-$seed-Uniform2Weights \
        --enable_other_metrics=True \
        --permute_classes=True 

python main.py \
        --dataset=seq-cifar100 \
        --model=lwf \
        --lr=0.03\
        --cog_cl 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-cifar100-cogcl-mammoth \
        --wandb_name=reversed-weights-sack-run-seed-$seed-Uniform2Weights \
        --enable_other_metrics=True \
        --permute_classes=True 

python main.py \
        --dataset=seq-cifar100-224 \
        --model=coda_prompt \
        --model_config=best\
        --cog_cl 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cifar100-cogcl-mammoth \
        --wandb_name=reversed-weights-sack-run-seed-$seed-Uniform2Weights \
        --enable_other_metrics=True \
        --permute_classes=True

