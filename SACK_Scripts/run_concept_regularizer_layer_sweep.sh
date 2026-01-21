#!/bin/bash

# Script to run icarl, der, derpp, lwf, and coda_prompt with concept units regularizer
# Testing different layer combinations for each method
# Records final avg accuracy and BTW results with mean ± std across 3 seeds
# Usage: bash run_concept_regularizer_layer_sweep.sh
# Edit layer definitions below to specify which layers to use for each method (comma-separated)

# Define layer combinations for each method (comma-separated, e.g., "layer1,classifier")
# ResNet-based models (icarl, der, derpp, lwf): classifier, layer4, layer3, features
ICARL_LAYERS="layer1, layer4, layer3, classifier"
DER_LAYERS="layer1, layer4, layer3, classifier"
DERPP_LAYERS="layer1, layer4, layer3, classifier"
LWF_LAYERS="layer1, layer4, layer3, classifier"
# ViT-based model (coda_prompt): last, blocks.11, blocks.10, blocks.9
CODAPROMPT_LAYERS="last, blocks.11, blocks.10, blocks.9"

echo "=========================================="
echo "Running experiments with Concept Units Regularizer - Layer Sweep"
echo "=========================================="
echo "iCaRL layers: $ICARL_LAYERS"
echo "DER layers: $DER_LAYERS"
echo "DER++ layers: $DERPP_LAYERS"
echo "LWF layers: $LWF_LAYERS"
echo "CODA-Prompt layers: $CODAPROMPT_LAYERS"

# Common concept regularizer flags (layer will be set per method)
CONCEPT_REG_BASE_FLAGS="--concept_units_reg=1 --concept_units_lambda=0.1 --concept_units_percentile=80.0 --concept_units_freq=1"

# Common flags for all models
COMMON_FLAGS="--model_config=best --enable_other_metrics=True --savecheck=task --permute_classes=True --sack_scores_type=0"

# Results storage directory
RESULTS_DIR="/p/lustre1/kundargi1/SACK/results/concept_reg_layer_sweep"
mkdir -p "$RESULTS_DIR"

# Function to normalize layer string (remove spaces after commas)
normalize_layers() {
    local layers="$1"
    # Remove spaces after commas and trim
    echo "$layers" | sed 's/,[ ]*/,/g' | sed 's/^[ ]*//;s/[ ]*$//'
}

# Function to extract final average accuracy and BTW from output
extract_results() {
    local output_file="$1"
    local model="$2"
    local layers="$3"
    local seed="$4"
    
    # Sanitize layers string for filename (replace commas with underscores)
    local layers_sanitized=$(echo "$layers" | tr ',' '_' | tr -d ' ')
    
    # Try to extract from log output - adjust patterns based on actual output format
    # Common patterns: "Average accuracy: X.XX", "Final accuracy: X.XX", "BTW: X.XX", "Backward Transfer: X.XX"
    local avg_acc=$(grep -iE "(average|final|mean).*accuracy|accuracy.*average|avg.*acc" "$output_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    local btw=$(grep -iE "backward.*transfer|BTW|bwt" "$output_file" | tail -1 | grep -oE "[-]?[0-9]+\.[0-9]+" | head -1)
    
    # If not found in output, try to parse from checkpoint or results file
    if [ -z "$avg_acc" ] || [ -z "$btw" ]; then
        # Try to find results in checkpoint directory or CSV files
        local ckpt_name="${model}-cifar100-concept-reg-layer-${layers_sanitized}-seed-${seed}"
        local results_file=$(find . -name "*${ckpt_name}*" -type f \( -name "*.csv" -o -name "*.json" -o -name "*.txt" \) 2>/dev/null | head -1)
        
        if [ -n "$results_file" ]; then
            # Try to extract from CSV or JSON
            if [[ "$results_file" == *.csv ]]; then
                avg_acc=$(tail -1 "$results_file" | cut -d',' -f1 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | head -1)
                btw=$(tail -1 "$results_file" | cut -d',' -f2 2>/dev/null | grep -oE "-?[0-9]+\.[0-9]+" | head -1)
            fi
        fi
    fi
    
    # Save to results file
    echo "${avg_acc},${btw}" > "$RESULTS_DIR/${model}_${layers_sanitized}_seed${seed}.txt"
    echo "$avg_acc" > "$RESULTS_DIR/${model}_${layers_sanitized}_seed${seed}_avgacc.txt"
    echo "$btw" > "$RESULTS_DIR/${model}_${layers_sanitized}_seed${seed}_btw.txt"
}

# Function to calculate mean and std from results
calculate_stats() {
    local model="$1"
    local layers="$2"
    local metric="$3"  # "avgacc" or "btw"
    
    # Sanitize layers string for filename (replace commas with underscores and remove spaces)
    local layers_sanitized=$(echo "$layers" | tr ',' '_' | tr -d ' ')
    
    local values=()
    for seed in 0 1 2; do
        local file="$RESULTS_DIR/${model}_${layers_sanitized}_seed${seed}_${metric}.txt"
        if [ -f "$file" ]; then
            local val=$(cat "$file" | grep -oE "[0-9]+\.[0-9]+|[-]?[0-9]+\.[0-9]+" | head -1)
            if [ -n "$val" ]; then
                values+=("$val")
            fi
        fi
    done
    
    if [ ${#values[@]} -eq 0 ]; then
        echo "N/A"
        return
    fi
    
    # Calculate mean using awk
    local mean=$(printf '%s\n' "${values[@]}" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    
    # Calculate std using awk
    local std=$(printf '%s\n' "${values[@]}" | awk -v mean="$mean" '{sum+=(($1-mean)^2); count++} END {if(count>1) printf "%.2f", sqrt(sum/(count-1)); else if(count==1) print "0.00"; else print "N/A"}')
    
    echo "${mean} ± ${std}"
}

# Run iCaRL with combined layers
echo ""
echo "=========================================="
echo "Running iCaRL with concept regularizer - Layer Sweep"
echo "=========================================="
for seed in 0 1 2
do
    layers_normalized=$(normalize_layers "$ICARL_LAYERS")
    layers_sanitized=$(echo "$layers_normalized" | tr ',' '_')
    echo "Starting iCaRL run with seed $seed, layers $layers_normalized"
    output_file="$RESULTS_DIR/icarl_${layers_sanitized}_seed${seed}_output.log"
    python main.py \
        --dataset=seq-cifar100 \
        --model=icarl \
        --buffer_size=2000 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-icarl-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-layer-${layers_sanitized}-seed-$seed \
        --ckpt_name=icarl-cifar100-concept-reg-layer-${layers_sanitized}-seed-$seed \
        --concept_units_layer="$layers_normalized" \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_BASE_FLAGS 2>&1 | tee "$output_file"
    
    extract_results "$output_file" "icarl" "$layers_normalized" "$seed"
    echo "Completed iCaRL run with seed $seed, layers $layers_normalized"
done

# Run DER with combined layers
echo ""
echo "=========================================="
echo "Running DER with concept regularizer - Layer Sweep"
echo "=========================================="
for seed in 0 1 2
do
    layers_normalized=$(normalize_layers "$DER_LAYERS")
    layers_sanitized=$(echo "$layers_normalized" | tr ',' '_')
    echo "Starting DER run with seed $seed, layers $layers_normalized"
    output_file="$RESULTS_DIR/der_${layers_sanitized}_seed${seed}_output.log"
    python main.py \
        --dataset=seq-cifar100 \
        --model=der \
        --buffer_size=2000 \
        --alpha=0.5 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-layer-${layers_sanitized}-seed-$seed \
        --ckpt_name=der-cifar100-concept-reg-layer-${layers_sanitized}-seed-$seed \
        --concept_units_layer="$layers_normalized" \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_BASE_FLAGS 2>&1 | tee "$output_file"
    
    extract_results "$output_file" "der" "$layers_normalized" "$seed"
    echo "Completed DER run with seed $seed, layers $layers_normalized"
done

# Run DER++ with combined layers
echo ""
echo "=========================================="
echo "Running DER++ with concept regularizer - Layer Sweep"
echo "=========================================="
for seed in 0 1 2
do
    layers_normalized=$(normalize_layers "$DERPP_LAYERS")
    layers_sanitized=$(echo "$layers_normalized" | tr ',' '_')
    echo "Starting DER++ run with seed $seed, layers $layers_normalized"
    output_file="$RESULTS_DIR/derpp_${layers_sanitized}_seed${seed}_output.log"
    python main.py \
        --dataset=seq-cifar100 \
        --model=derpp \
        --buffer_size=2000 \
        --alpha=0.5 \
        --beta=0.5 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-derpp-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-layer-${layers_sanitized}-seed-$seed \
        --ckpt_name=derpp-cifar100-concept-reg-layer-${layers_sanitized}-seed-$seed \
        --concept_units_layer="$layers_normalized" \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_BASE_FLAGS 2>&1 | tee "$output_file"
    
    extract_results "$output_file" "derpp" "$layers_normalized" "$seed"
    echo "Completed DER++ run with seed $seed, layers $layers_normalized"
done

# Run LWF with combined layers
echo ""
echo "=========================================="
echo "Running LWF with concept regularizer - Layer Sweep"
echo "=========================================="
for seed in 0 1 2
do
    layers_normalized=$(normalize_layers "$LWF_LAYERS")
    layers_sanitized=$(echo "$layers_normalized" | tr ',' '_')
    echo "Starting LWF run with seed $seed, layers $layers_normalized"
    output_file="$RESULTS_DIR/lwf_${layers_sanitized}_seed${seed}_output.log"
    python main.py \
        --dataset=seq-cifar100 \
        --model=lwf \
        --lr=0.003 \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-lwf-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-layer-${layers_sanitized}-seed-$seed \
        --ckpt_name=lwf-cifar100-concept-reg-layer-${layers_sanitized}-seed-$seed \
        --concept_units_layer="$layers_normalized" \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_BASE_FLAGS 2>&1 | tee "$output_file"
    
    extract_results "$output_file" "lwf" "$layers_normalized" "$seed"
    echo "Completed LWF run with seed $seed, layers $layers_normalized"
done

# Run CODA-Prompt with combined layers
echo ""
echo "=========================================="
echo "Running CODA-Prompt with concept regularizer - Layer Sweep"
echo "=========================================="
for seed in 0 1 2
do
    layers_normalized=$(normalize_layers "$CODAPROMPT_LAYERS")
    layers_sanitized=$(echo "$layers_normalized" | tr ',' '_')
    echo "Starting CODA-Prompt run with seed $seed, layers $layers_normalized"
    output_file="$RESULTS_DIR/codaprompt_${layers_sanitized}_seed${seed}_output.log"
    python main.py \
        --dataset=seq-cifar100-224 \
        --model=coda_prompt \
        --cog_cl=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
        --wandb_name=concept-reg-layer-${layers_sanitized}-seed-$seed \
        --ckpt_name=codaprompt-cifar100-concept-reg-layer-${layers_sanitized}-seed-$seed \
        --concept_units_layer="$layers_normalized" \
        --seed=$seed \
        $COMMON_FLAGS \
        $CONCEPT_REG_BASE_FLAGS 2>&1 | tee "$output_file"
    
    extract_results "$output_file" "codaprompt" "$layers_normalized" "$seed"
    echo "Completed CODA-Prompt run with seed $seed, layers $layers_normalized"
done

# Generate summary report
echo ""
echo "=========================================="
echo "RESULTS SUMMARY - Final Avg Accuracy and BTW (Mean ± Std over 3 seeds)"
echo "=========================================="
echo ""
printf "%-15s %-30s %-25s %-25s\n" "Method" "Layers" "Final Avg Accuracy" "BTW"
echo "--------------------------------------------------------------------------------"

# iCaRL
icarl_layers_norm=$(normalize_layers "$ICARL_LAYERS")
avg_acc_stats=$(calculate_stats "icarl" "$icarl_layers_norm" "avgacc")
btw_stats=$(calculate_stats "icarl" "$icarl_layers_norm" "btw")
printf "%-15s %-30s %-25s %-25s\n" "iCaRL" "$icarl_layers_norm" "$avg_acc_stats" "$btw_stats"

# DER
der_layers_norm=$(normalize_layers "$DER_LAYERS")
avg_acc_stats=$(calculate_stats "der" "$der_layers_norm" "avgacc")
btw_stats=$(calculate_stats "der" "$der_layers_norm" "btw")
printf "%-15s %-30s %-25s %-25s\n" "DER" "$der_layers_norm" "$avg_acc_stats" "$btw_stats"

# DER++
derpp_layers_norm=$(normalize_layers "$DERPP_LAYERS")
avg_acc_stats=$(calculate_stats "derpp" "$derpp_layers_norm" "avgacc")
btw_stats=$(calculate_stats "derpp" "$derpp_layers_norm" "btw")
printf "%-15s %-30s %-25s %-25s\n" "DER++" "$derpp_layers_norm" "$avg_acc_stats" "$btw_stats"

# LWF
lwf_layers_norm=$(normalize_layers "$LWF_LAYERS")
avg_acc_stats=$(calculate_stats "lwf" "$lwf_layers_norm" "avgacc")
btw_stats=$(calculate_stats "lwf" "$lwf_layers_norm" "btw")
printf "%-15s %-30s %-25s %-25s\n" "LWF" "$lwf_layers_norm" "$avg_acc_stats" "$btw_stats"

# CODA-Prompt
codaprompt_layers_norm=$(normalize_layers "$CODAPROMPT_LAYERS")
avg_acc_stats=$(calculate_stats "codaprompt" "$codaprompt_layers_norm" "avgacc")
btw_stats=$(calculate_stats "codaprompt" "$codaprompt_layers_norm" "btw")
printf "%-15s %-30s %-25s %-25s\n" "CODA-Prompt" "$codaprompt_layers_norm" "$avg_acc_stats" "$btw_stats"

echo ""
echo "=========================================="
echo "All layer sweep experiments completed!"
echo "Results stored in: $RESULTS_DIR"
echo "=========================================="
