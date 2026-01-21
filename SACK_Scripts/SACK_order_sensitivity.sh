#!/bin/bash

# export seed=0

# #SACK(U->W)

# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-uniform2weights\
#         --enable_other_metrics=True \
#         --permute_classes=True \
#         --order_change=1 \
#         --seed=$seed

# export seed=1
# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-uniform2weights\
#         --enable_other_metrics=True \
#         --permute_classes=True  \
#         --order_change=1 \
#         --seed=$seed
             

# export seed=2
# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-uniform2weights\
#         --enable_other_metrics=True \
#         --permute_classes=True  \
#         --order_change=1 \
#         --seed=$seed

# export seed=3
# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-uniform2weights\
#         --enable_other_metrics=True \
#         --permute_classes=True  \
#         --order_change=1 \
#         --seed=$seed

# export seed=4
# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-uniform2weights\
#         --enable_other_metrics=True \
#         --permute_classes=True  \
#         --order_change=1 \
#         --seed=$seed







export seed=0
python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 0 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
        --wandb_name=original-run-$seed\
        --enable_other_metrics=True \
        --permute_classes=True  \
        --order_change=1 \
        --seed=$seed 




export seed=1

python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 0 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
        --wandb_name=original-run-$seed\
        --enable_other_metrics=True \
        --permute_classes=True  \
        --order_change=1 \
        --seed=$seed 








export seed=2

python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 0 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
        --wandb_name=original-run-$seed\
        --enable_other_metrics=True \
        --permute_classes=True  \
        --order_change=1 \
        --seed=$seed 









export seed=3

python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 0 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
        --wandb_name=original-run-$seed\
        --enable_other_metrics=True \
        --permute_classes=True  \
        --order_change=1 \
        --seed=$seed













export seed=4

python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 0 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=order-sensitivity-icarl-cifar100-SACK-mammoth \
        --wandb_name=original-run-$seed\
        --enable_other_metrics=True \
        --permute_classes=True  \
        --order_change=1 \
        --seed=$seed       