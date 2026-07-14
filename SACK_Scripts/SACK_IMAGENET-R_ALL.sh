#!/bin/bash

for seed in 0 1 2
do
        echo "Starting iCaRL run with seed $seed"

        python main.py \
                --dataset=seq-imagenet-r \
                --model=icarl \
                --buffer_size=2000 \
                --backbone=resnet50 \
                --model_config=best \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-icarl-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --ckpt_name=icarl-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 

        python main.py \
                --dataset=seq-imagenet-r \
                --model=lwf \
                --lr=0.03\
                --backbone=resnet50 \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-lwf-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --ckpt_name=lwf-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 


        python main.py \
                --dataset=seq-imagenet-r \
                --model=der \
                --buffer_size=2000 \
                --backbone=resnet50 \
                --model_config=best \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-der-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed\
                --enable_other_metrics=True \
                --savecheck=task \
                --ckpt_name=der-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 


        python main.py \
                --dataset=seq-imagenet-r \
                --model=derpp \
                --buffer_size=2000 \
                --backbone=resnet50 \
                --model_config=best \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-derpp-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --ckpt_name=derpp-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed


        python main.py \
                --dataset=seq-imagenet-r \
                --model=coda_prompt \
                --model_config=best\
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-coda_prompt-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --ckpt_name=coda_prompt-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed       

done




# python main.py \
#         --dataset=seq-imagenet-r \
#         --model=lwf \
#         --lr=0.03\
#         --backbone=resnet50 \
#         --sack 1 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-lwf-imagenet-r-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-Weight2uniform \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed 


# python main.py \
#         --dataset=seq-imagenet-r \
#         --model=der \
#         --buffer_size=2000 \
#         --backbone=resnet50 \
#         --model_config=best \
#         --sack 1 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-der-imagenet-r-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-Weight2uniform \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed 


# python main.py \
#         --dataset=seq-imagenet-r \
#         --model=derpp \
#         --buffer_size=2000 \
#         --backbone=resnet50 \
#         --model_config=best \
#         --sack 1 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-derpp-imagenet-r-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-Weight2uniform \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed


# python main.py \
#         --dataset=seq-imagenet-r \
#         --model=coda_prompt \
#         --model_config=best\
#         --sack 1 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-coda_prompt-imagenet-r-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-Weight2uniform \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed                        


python main.py \
        --dataset=seq-imagenet-r \
        --model=lwf \
        --lr=0.03\
        --backbone=resnet50 \
        --sack 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-imagenet-r-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-uniform2Weight \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed 


python main.py \
        --dataset=seq-imagenet-r \
        --model=der \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-imagenet-r-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-uniform2Weight \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed 


python main.py \
        --dataset=seq-imagenet-r \
        --model=derpp \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-imagenet-r-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-uniform2Weight \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed


python main.py \
        --dataset=seq-imagenet-r \
        --model=coda_prompt \
        --model_config=best\
        --sack 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-imagenet-r-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-uniform2weights \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed  
















python main.py \
        --dataset=seq-imagenet-r \
        --model=lwf \
        --lr=0.03\
        --backbone=resnet50 \
        --sack 1 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-imagenet-r-cogcl-mammoth \
        --wandb_name=random \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed 


python main.py \
        --dataset=seq-imagenet-r \
        --model=der \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-imagenet-r-cogcl-mammoth \
        --wandb_name=random \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed 


python main.py \
        --dataset=seq-imagenet-r \
        --model=derpp \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-imagenet-r-cogcl-mammoth \
        --wandb_name=random \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed


python main.py \
        --dataset=seq-imagenet-r \
        --model=coda_prompt \
        --model_config=best\
        --sack 1 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-imagenet-r-cogcl-mammoth \
        --wandb_name=random \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed
