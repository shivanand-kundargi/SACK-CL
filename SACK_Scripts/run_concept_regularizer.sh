#!/bin/bash

# Script to run icarl, der, derpp, lwf, and coda_prompt with concept units regularizer enabled
# Usage: bash run_concept_regularizer.sh

echo "=========================================="
echo "Running experiments with Concept Units Regularizer"
echo "=========================================="

# Common concept regularizer flags
CONCEPT_REG_FLAGS="--concept_units_reg=1 --concept_units_lambda=0.1 --concept_units_percentile=80.0 --concept_units_freq=1"

# Common flags for all models
COMMON_FLAGS="--model_config=best --enable_other_metrics=True --savecheck=task --permute_classes=True --sack_scores_type=0"

# Run iCaRL
echo ""
echo "=========================================="
echo "Running iCaRL with concept regularizer"
echo "=========================================="
for seed in 0 1 2
do
    echo "Starting iCaRL run with seed $seed"
    python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --n_epochs=5 \
        --sack=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-icarl-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=icarl-cifar100-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_icarl_seq-cifar100 \
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
        --dataset=seq-cifar100 \
        --model=der \
        --buffer_size=2000 \
        --alpha=0.5 \
        --sack=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=der-cifar100-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_der_seq-cifar100 \
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
        --dataset=seq-cifar100 \
        --model=derpp \
        --buffer_size=2000 \
        --alpha=0.5 \
        --beta=0.5 \
        --sack=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=derpp-cifar100-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_derpp_seq-cifar100 \
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
        --dataset=seq-cifar100 \
        --model=lwf \
        --lr=0.003 \
        --sack=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=lwf-cifar100-concept-reg-seed-$seed \
        --concept_units_layer=classifier \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_lwf_seq-cifar100 \
        --seed=$seed \
        $COMMON_FLAGS \
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
        --dataset=seq-cifar100-224 \
        --model=coda_prompt \
        --sack=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-run-seed-$seed \
        --ckpt_name=codaprompt-cifar100-concept-reg-seed-$seed \
        --concept_units_layer=last \
        --concept_units_stats_dir=/p/lustre1/kundargi1/SACK/saved_activations_coda_prompt_seq-cifar100 \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_FLAGS
    echo "Completed CODA-Prompt run with seed $seed"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

