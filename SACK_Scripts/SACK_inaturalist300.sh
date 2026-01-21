#!/bin/bash

# Runs SACK experiments on the sequential iNaturalist dataset limited to the first 300 classes.
SEEDS=(0 1 2)
WANDB_ENTITY="abcxyz8431-cl"

for seed in "${SEEDS[@]}"; do

        # iCaRL original (no SACK, cog_cl 0)
        echo "Starting iNaturalist-300 iCaRL ORIGINAL run with seed $seed (CoG-CL disabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=icarl \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 0 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-icarl-inat300-ORIGINAL \
                --wandb_name=icarl-original-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=icarl-inat300-original-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # iCaRL with SACK (cog_cl 1)
        echo "Starting iNaturalist-300 iCaRL SACK run with seed $seed (CoG-CL enabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=icarl \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-icarl-inat300-SACK \
                --wandb_name=icarl-sack-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=icarl-inat300-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # DER original (no SACK, cog_cl 0)
        echo "Starting iNaturalist-300 DER ORIGINAL run with seed $seed (CoG-CL disabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=der \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 0 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-der-inat300-ORIGINAL \
                --wandb_name=der-original-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=der-inat300-original-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # DER with SACK (cog_cl 1)
        echo "Starting iNaturalist-300 DER SACK run with seed $seed (CoG-CL enabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=der \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-der-inat300-SACK \
                --wandb_name=der-sack-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=der-inat300-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # DER++ original (no SACK, cog_cl 0)
        echo "Starting iNaturalist-300 DER++ ORIGINAL run with seed $seed (CoG-CL disabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=derpp \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 0 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-derpp-inat300-ORIGINAL \
                --wandb_name=derpp-original-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=derpp-inat300-original-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # DER++ with SACK (cog_cl 1)
        echo "Starting iNaturalist-300 DER++ SACK run with seed $seed (CoG-CL enabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=derpp \
                --buffer_size=2000 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-derpp-inat300-SACK \
                --wandb_name=derpp-sack-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=derpp-inat300-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # LwF original (no SACK, cog_cl 0)
        echo "Starting iNaturalist-300 LwF ORIGINAL run with seed $seed (CoG-CL disabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=lwf \
                --lr=0.003 \
                --cog_cl 0 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-lwf-inat300-ORIGINAL \
                --wandb_name=lwf-original-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=lwf-inat300-original-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # LwF with SACK (cog_cl 1)
        echo "Starting iNaturalist-300 LwF SACK run with seed $seed (CoG-CL enabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=lwf \
                --lr=0.003 \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-lwf-inat300-SACK \
                --wandb_name=lwf-sack-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=lwf-inat300-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # CoDA-Prompt original (no SACK, cog_cl 0)
        echo "Starting iNaturalist-300 CoDA-Prompt ORIGINAL run with seed $seed (CoG-CL disabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=coda_prompt \
                --model_config=best \
                --cog_cl 0 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-coda-prompt-inat300-ORIGINAL \
                --wandb_name=coda_prompt-original-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=coda_prompt-inat300-original-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"

        # CoDA-Prompt with SACK (cog_cl 1)
        echo "Starting iNaturalist-300 CoDA-Prompt SACK run with seed $seed (CoG-CL enabled)"
        python main.py \
                --dataset=seq-inaturalist-300 \
                --model=coda_prompt \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity="$WANDB_ENTITY" \
                --wandb_project=Final-coda-prompt-inat300-SACK \
                --wandb_name=coda_prompt-sack-$seed \
                --enable_other_metrics=True \
                --permute_classes=True \
                --savecheck=task \
                --ckpt_name=coda_prompt-inat300-sack-seed-$seed \
                --log_perf_metrics=1 \
                --seed="$seed"


        done