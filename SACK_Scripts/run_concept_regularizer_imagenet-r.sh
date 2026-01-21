#!/bin/bash

# Script to run icarl, der, derpp, lwf, and coda_prompt with concept units regularizer enabled on ImageNet-R
# Usage: bash run_concept_regularizer_imagenet-r.sh

echo "=========================================="
echo "Running experiments with Concept Units Regularizer on ImageNet-R"
echo "=========================================="

# Common concept regularizer flags
CONCEPT_REG_FLAGS="--concept_units_reg=1 --concept_units_lambda=0.1 --concept_units_percentile=80.0 --concept_units_freq=1"

# Common flags for all models
COMMON_FLAGS="--model_config=best --enable_other_metrics=True --savecheck=task --permute_classes=True --sack_scores_type=0 --log_perf_metrics=1"

# Run iCaRL
echo ""
echo "=========================================="
echo "Running iCaRL with concept regularizer"
echo "=========================================="
for seed in 0 1 2
do
    echo "Starting iCaRL run with seed $seed"
    python main.py \
        --dataset=seq-imagenet-r \
        --model=icarl \
        --buffer_size=2000 \
        --backbone=resnet50 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-icarl-imagenet-r-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=icarl-imagenet-r-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_icarl_seq-imagenet-r \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_FLAGS
    echo "Completed iCaRL run with seed $seed"
done

# Run DER
echo ""
echo "=========================================="
echo "Running DER with concept regularizer"
echo "=========================================="
for seed in 0 1 2
do
    echo "Starting DER run with seed $seed"
    python main.py \
        --dataset=seq-imagenet-r \
        --model=der \
        --backbone=resnet50 \
        --buffer_size=2000 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-imagenet-r-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=der-imagenet-r-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_der_seq-imagenet-r \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_FLAGS
    echo "Completed DER run with seed $seed"
done

# Run DER++
echo ""
echo "=========================================="
echo "Running DER++ with concept regularizer"
echo "=========================================="
for seed in 0 1 2
do
    echo "Starting DER++ run with seed $seed"
    python main.py \
        --dataset=seq-imagenet-r \
        --model=derpp \
        --backbone=resnet50 \
        --buffer_size=2000 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-imagenet-r-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=derpp-imagenet-r-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_derpp_seq-imagenet-r \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_FLAGS
    echo "Completed DER++ run with seed $seed"
done

# Run LWF
echo ""
echo "=========================================="
echo "Running LWF with concept regularizer"
echo "=========================================="
for seed in 0 1 2
do
    echo "Starting LWF run with seed $seed"
    python main.py \
        --dataset=seq-imagenet-r \
        --model=lwf \
        --lr=0.03 \
        --backbone=resnet50 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-imagenet-r-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=lwf-imagenet-r-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --seed=$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --sack_scores_type=0 \
        --log_perf_metrics=1 \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_lwf_seq-imagenet-r \
        $CONCEPT_REG_FLAGS
    echo "Completed LWF run with seed $seed"
done

# Run CODA-Prompt
echo ""
echo "=========================================="
echo "Running CODA-Prompt with concept regularizer"
echo "=========================================="
for seed in 0 1 2
do
    echo "Starting CODA-Prompt run with seed $seed"
    python main.py \
        --dataset=seq-imagenet-r \
        --model=coda_prompt \
        --model_config=best \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-imagenet-r-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=codaprompt-imagenet-r-concept-reg-seed-$seed \
        --concept_units_layer=last \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_coda_prompt_seq-imagenet-r \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_FLAGS
    echo "Completed CODA-Prompt run with seed $seed"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
