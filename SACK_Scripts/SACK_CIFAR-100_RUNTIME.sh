seed=0
echo "Starting run with seed $seed"   
        python main.py \
                --dataset=seq-cifar100 \
                --model=icarl \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-icarl-cifar100-SACK-mammoth \
                --wandb_name=SACK-$seed\
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=icarl-cifar100-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed 

        python main.py \
                --dataset=seq-cifar100 \
                --model=der \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-der-cifar100-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=der-cifar100-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed

        python main.py \
                --dataset=seq-cifar100 \
                --model=derpp \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-derpp-cifar100-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=derpp-cifar100-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed

        python main.py \
                --dataset=seq-cifar100 \
                --model=lwf \
                --lr=0.003\
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-lwf-cifar100-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=lwf-cifar100-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed
        python main.py \
                --dataset=seq-cifar100-224 \
                --model=coda_prompt \
                --model_config=best\
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=coda_prompt-cifar100-sack-seed-$seed \
                --log_perf_metrics=1 \
                --weighted_gradient 0 \
                --seed=$seed


        python main.py \
                --dataset=seq-cifar100-224 \
                --model=coda_prompt \
                --model_config=best\
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=coda_prompt-cifar100-sack-seed-$seed \
                --log_perf_metrics=1 \
                --weighted_gradient 1 \
                --seed=$seed