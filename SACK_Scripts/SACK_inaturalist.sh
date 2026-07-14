#!/bin/bash

# Runs SACK experiments on the sequential iNaturalist dataset.
SEEDS=(0 1 2)
WANDB_ENTITY="abcxyz8431-cl"

for seed in "${SEEDS[@]}"; do
        echo "Starting iNaturalist iCaRL run with seed $seed"
        python main.py \
                --dataset=seq-inaturalist-300    \
                --model=icarl \
                --buffer_size=2000 \
                --model_config=best \
                --sack 0 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-icarl-inaturalist-SACK \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=icarl-inaturalist-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        echo "Starting iNaturalist iCaRL run with seed $seed"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=icarl \
                --buffer_size=2000 \
                --model_config=best \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-icarl-inaturalist-SACK \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=icarl-inaturalist-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"
        echo "Starting iNaturalist DER run with seed $seed"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=der \
                --buffer_size=2000 \
                --model_config=best \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-der-inaturalist-SACK \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=der-inaturalist-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        echo "Starting iNaturalist DER++ run with seed $seed"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=derpp \
                --buffer_size=2000 \
                --model_config=best \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-derpp-inaturalist-SACK \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --seed="$seed"

        echo "Starting iNaturalist LwF run with seed $seed"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=lwf \
                --lr=0.003 \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-lwf-inaturalist-SACK \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=lwf-inaturalist-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        echo "Starting iNaturalist CoDA-Prompt run with seed $seed"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=coda_prompt \
                --model_config=best \
                --sack 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-coda_prompt-inaturalist-SACK \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --ckpt_name=coda_prompt-inaturalist-sack-seed-$seed \
                --log_perf_metrics=1 \
                --permute_classes=True \
                --seed="$seed"
done
