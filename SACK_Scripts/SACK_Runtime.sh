seed=69
#CIFAR-100

python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-icarl-cifar100-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed\
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --ckpt_name=icarl-cifar100-sack-seed-$seed \
        --log_perf_metrics=1 \
        --seed=$seed 


python main.py \
        --dataset=seq-cifar100 \
        --model=der \
        --buffer_size=500 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-cifar100-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --ckpt_name=der-cifar100-sack-seed-$seed \
        --log_perf_metrics=1 \
        --seed=$seed

python main.py \
        --dataset=seq-cifar100 \
        --model=derpp \
        --buffer_size=500 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-cifar100-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-cifar100-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --ckpt_name=coda_prompt-cifar100-sack-seed-$seed \
        --log_perf_metrics=1 \
        --seed=$seed


#CUB200

python main.py \
        --dataset=seq-cub200 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-icarl-cub200-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed\
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-cub200-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-cub200-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-cub200-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --ckpt_name=lwf-cub200-sack-seed-$seed \
        --log_perf_metrics=1 \
        --seed=$seed


#CODA_PROMPT
python main.py \
        --dataset=seq-cub200 \
        --model=coda_prompt \
        --model_config=best\
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cub200-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --ckpt_name=coda_prompt-cub200-sack-seed-$seed \
        --log_perf_metrics=1 \
        --seed=$seed

#IMAGENET-R
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
        --wandb_name=SACK_Runtime_Analysis-$seed \
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-imagenet-r-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-imagenet-r-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed\
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-imagenet-r-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
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
        --sack 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-imagenet-r-SACK-mammoth \
        --wandb_name=SACK_Runtime_Analysis-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --ckpt_name=coda_prompt-imagenet-r-sack-seed-$seed \
        --log_perf_metrics=1 \
        --seed=$seed 