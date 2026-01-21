seed=0

        python main.py \
                --dataset=seq-imagenet-r \
                --model=icarl \
                --buffer_size=2000 \
                --backbone=resnet50 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-icarl-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=icarl-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 

        
        python main.py \
                --dataset=seq-imagenet-r \
                --model=lwf \
                --lr=0.03\
                --backbone=resnet50 \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-lwf-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=lwf-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 


        python main.py \
                --dataset=seq-imagenet-r \
                --model=der \
                --buffer_size=500 \
                --backbone=resnet50 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-der-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed\
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=der-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 


        python main.py \
                --dataset=seq-imagenet-r \
                --model=derpp \
                --buffer_size=500 \
                --backbone=resnet50 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-derpp-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=derpp-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed


        python main.py \
                --dataset=seq-imagenet-r \
                --model=coda_prompt \
                --model_config=best\
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-coda_prompt-imagenet-r-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=coda_prompt-imagenet-r-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 