seed=0
echo "Starting run with seed $seed"                                                                                                          
        python main.py \
                --dataset=seq-cub200 \
                --model=icarl \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-icarl-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed\
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=icarl-cub200-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed
                echo "Finished run with seed $seed"

        # DER
        python main.py \
                --dataset=seq-cub200 \
                --model=der \
                --buffer_size=500 \
                --backbone=resnet50 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-der-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=der-cub200-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed


        #DER++
        python main.py \
                --dataset=seq-cub200 \
                --model=derpp \
                --buffer_size=500 \
                --backbone=resnet50 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-derpp-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=derpp-cub200-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed
        #LWF

        python main.py \
                --dataset=seq-cub200 \
                --model=lwf \
                --lr=0.03\
                --backbone=resnet50 \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-lwf-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=lwf-cub200-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed


        # CODA_PROMPT
        python main.py \
                --dataset=seq-cub200 \
                --model=coda_prompt \
                --model_config=best\
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-coda_prompt-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --ckpt_name=coda_prompt-cub200-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed=$seed
