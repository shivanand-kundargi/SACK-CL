# SACK: Sequentially Acquiring Concept Knowledge to Guide Continual Learning

**Abstract**:
*The goal of continual learning (CL) is to adapt to new data (plasticity) while retaining the knowledge acquired from old data (stability). Existing methods focus on balancing stability and plasticity to mitigate the challenge of catastrophic forgetting while promoting learning. However, the impact of order and nature of new samples that the network is trained on remains an underexplored factor. A CL algorithm should ideally also have the ability to rank incoming samples in terms of their relationship with prior data and their effect on the learning process. In this work, we investigate if scoring and prioritizing incoming data based on their semantic relationships with the model's current knowledge can boost CL performance. We propose SACK, short for Sequentially Acquiring Concept Knowledge, a scalable and model-agnostic two-step technique for continual learning. SACK dissects categorical knowledge of the model into fine-grained concepts, computes the relationships between previously learned concepts and new concepts in each experience, and uses this relationship knowledge for prioritizing new samples. Experiments across several types of CL methods (regularization, replay, and prompt-based) in class-incremental and task-incremental settings demonstrate that our approach consistently results in higher accuracy, reduces forgetting, enhances plasticity, and handles long-tail distribution.*

Below we layout a comprehensive guide which documents all functionalities available in SACK (Sequentially Acquiring Concept Knowledge to Guide Continual Learning) and how to execute them.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Core Training Functionality](#core-training-functionality)
3. [Dataset-Specific Scripts](#dataset-specific-scripts)
4. [Evaluation and Analysis Tools](#evaluation-and-analysis-tools)
5. [Advanced Features](#advanced-features)
6. [Visualization Tools](#visualization-tools)
7. [Model-Specific Configurations](#model-specific-configurations)
8. [Troubleshooting](#troubleshooting)

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch ≥ 2.1.0
- CUDA-capable GPU (recommended)

### Installation

```bash
cd SACK
pip install -r requirements.txt
```

### Dataset Preparation

#### iNaturalist Dataset
```bash
# One-time setup before running seq-inaturalist experiments
python scripts/download_inaturalist.py
```

#### CIFAR-100-C (for robustness evaluation)
```bash
# Download CIFAR-100-C corruptions
python scripts/download_cifar100c.py
```

---

## Core Training Functionality

### Basic Training (Without SACK)

Train a baseline model without SACK:

```bash
python main.py \
    --dataset=seq-cifar100-224 \
    --model=coda_prompt \
    --model_config=best \
    --cog_cl=0 \
    --sack_scores_type=0 \
    --wandb_entity=your-entity \
    --wandb_project=your-project \
    --wandb_name=baseline-seed-0 \
    --enable_other_metrics=True \
    --permute_classes=True \
    --seed=0
```

### Training with SACK

Enable SACK by setting `--cog_cl=1`:

```bash
python main.py \
    --dataset=seq-cifar100-224 \
    --model=coda_prompt \
    --model_config=best \
    --cog_cl=1 \
    --sack_scores_type=0 \
    --wandb_entity=your-entity \
    --wandb_project=your-project \
    --wandb_name=SACK-seed-0 \
    --enable_other_metrics=True \
    --permute_classes=True \
    --seed=0
```

### Key Parameters

- `--cog_cl`: Enable (1) or disable (0) SACK/CoG-CL
- `--sack_scores_type`: Type of SACK scoring (0=default, 1=uniform-to-weights, 2=alternative)
- `--dataset`: Dataset identifier (see [Supported Datasets](#supported-datasets))
- `--model`: Model architecture (see [Supported Models](#supported-models))
- `--seed`: Random seed for reproducibility
- `--permute_classes=True`: Randomize class order
- `--enable_other_metrics=True`: Compute forward/backward transfer and forgetting
- `--savecheck=task`: Save checkpoint after each task
- `--log_perf_metrics=1`: Log performance metrics

---

## Dataset-Specific Scripts

### CIFAR-100

#### Full Training Suite
```bash
# Run all models on CIFAR-100 with SACK
./SACK_Scripts/SACK_CIFAR-100_RUNTIME.sh

# Run baseline (without SACK)
./SACK_Scripts/OG_CIFAR-100_RUNTIME.sh
```

#### Individual Model Runs
```bash
# Run specific models on CIFAR-100
./SACK_Scripts/SACK_CIFAR100_runs.sh
```

### CUB-200

#### Full Training Suite
```bash
# Run all models on CUB-200 with SACK
./SACK_Scripts/SACK_CUB-200_RUNTIME.sh

# Run baseline (without SACK)
./SACK_Scripts/OG_CUB-200_RUNTIME.sh
```

#### Batch Processing
```bash
# Process multiple classes for visualization
./run_cub200_batch.sh
```

### ImageNet-R

#### Full Training Suite
```bash
# Run all models on ImageNet-R with SACK
./SACK_Scripts/SACK_IMAGENET-R_RUNTIME.sh

# Run baseline (without SACK)
./SACK_Scripts/OG_IMAGENET-R_RUNTIME.sh
```

#### All Models
```bash
# Run comprehensive ImageNet-R experiments
./SACK_Scripts/SACK_IMAGENET-R_ALL.sh
```

### iNaturalist

#### iNaturalist-300 (300 classes)
```bash
# Run SACK experiments on iNaturalist-300
./SACK_Scripts/SACK_inaturalist300.sh

# Run baseline
./SACK_Scripts/OG_inaturalist300.sh
```

#### iNaturalist-1000 (1000 classes)
```bash
# Run SACK experiments on iNaturalist-1000
./SACK_Scripts/SACK_inaturalis1000.sh
```

#### Full iNaturalist
```bash
# Run SACK experiments on full iNaturalist
./SACK_Scripts/SACK_inaturalist.sh
```

### CIFAR-10
```bash
# Run with SACK
./SACK_Scripts/cifar10_cogcl.sh

# Run baseline
./SACK_Scripts/cifar10og.sh
```

---

## Evaluation and Analysis Tools

### Expected Calibration Error (ECE) Computation

Compute ECE for trained checkpoints:

```bash
python SACK_ECE.py \
    --dataset=seq-cifar100 \
    --model=icarl \
    --cog_cl=1 \
    --checkpoint_dir=checkpoints \
    --checkpoint_prefix=icarl-cifar100-sack-seed-0 \
    --logits_dir=ece_comparisons/icarl/sack/logits \
    --summary_path=ece_comparisons/icarl/sack/ece_summary.json \
    --validation_mode=complete \
    --sack_scores_type=0 \
    --permute_classes=True \
    --seed=0 \
    --buffer_size=2000 \
    --model_config=best
```

#### Batch ECE Comparison

Compare ECE between baseline and SACK variants:

```bash
# Compare all models
./SACK_Scripts/run_ece_comparison.sh [seed]

# With custom checkpoint directory
CHECKPOINT_DIR=/path/to/checkpoints ./SACK_Scripts/run_ece_comparison.sh 0
```

### Robustness Evaluation (CIFAR-100-C)

Evaluate model robustness on CIFAR-100-C corruptions:

```bash
# Export checkpoint paths
export ICARL_BASELINE_CKPT=/path/to/icarl/original_final.pt
export ICARL_SACK_CKPT=/path/to/icarl/sack_final.pt
export DER_BASELINE_CKPT=/path/to/der/original_final.pt
export DER_SACK_CKPT=/path/to/der/sack_final.pt
# ... repeat for other methods

# Run robustness evaluation
./SACK_Scripts/sack_cifar100c_robustness.sh
```

**Configuration Options:**
- `WANDB_ENTITY`: WandB entity (default: abcxyz8431-cl)
- `WANDB_PROJECT_PREFIX`: Project prefix (default: cifar100c-robustness)
- `SACK_SCORES_TYPE`: SACK scoring type (default: 0)
- `SEED`: Random seed (default: 0)

**Supported Corruptions:**
- gaussian_noise, shot_noise, impulse_noise
- defocus_blur, glass_blur, motion_blur, zoom_blur, gaussian_blur
- snow, frost, fog
- brightness, contrast, saturate
- elastic_transform, pixelate, jpeg_compression, spatter, speckle_noise

**Severity Levels:** 1-5

### Strong Subset Evaluation

Evaluate on a strong subset of CIFAR-100-C:

```bash
./SACK_Scripts/sack_cifar100c_strong_subset.sh
```

---

## Advanced Features

### Uncertainty-Based Sampling

Run experiments with uncertainty-based sampling instead of concept-score-weighted sampling:

```bash
./SACK_Scripts/SACK_CIFAR100_uncertainity.sh
```

**Environment Variables:**
- `SEEDS`: Comma-separated list of seeds (default: "1")
- `MODEL`: Model name (default: "coda-prompt")
- `BUFFER_SIZE`: Replay buffer size (default: "2000")
- `BATCH_SIZE`: Training batch size (default: "128")
- `WANDB_ENTITY`: WandB entity override
- `WANDB_PROJECT`: WandB project override
- `NUM_WORKERS`: DataLoader workers (optional)
- `UNCERTAINTY_EVAL_BATCH_SIZE`: Batch size for uncertainty scoring (optional)

**Example:**
```bash
SEEDS="0,1,2" MODEL="icarl" BUFFER_SIZE="2000" \
./SACK_Scripts/SACK_CIFAR100_uncertainity.sh
```

### Threshold Sweep

Perform threshold sweeps to find optimal similarity percentile:

```bash
./SACK_Scripts/sack_cifar100_threshold_sweep.sh
```

**Configuration:**
- `THRESHOLDS`: Array of similarity percentiles (default: 60 65 70 75 80 85)
- `SEED`: Random seed (default: 0)
- `SACK_SCORES_TYPE`: SACK scoring type (default: 0)

**Custom Thresholds:**
```bash
# Edit the script to modify THRESHOLDS array
THRESHOLDS=(50 55 60 65 70 75 80 85 90) ./SACK_Scripts/sack_cifar100_threshold_sweep.sh
```

### Order Sensitivity Analysis

Analyze sensitivity to class ordering:

```bash
./SACK_Scripts/SACK_order_sensitivity.sh
```

This script runs experiments with different class orders to evaluate robustness to task ordering.

### Ablation Studies

#### Reverse Scores Ablation
```bash
./SACK_Scripts/ablation_SACK_CIFAR100_reversescores.sh
```

---

## Visualization Tools

### Grad-CAM Visualization

Generate Grad-CAM visualizations comparing baseline and SACK models:

#### Single Class Visualization

```bash
# Basic usage
./rungradcamviz.sh "apple"

# With custom parameters
MODEL_NAME=icarl \
DATASET=seq-cifar100 \
SEED=0 \
LAYER_NAME=layer4 \
./rungradcamviz.sh "apple" --verbose
```

#### Batch Processing (CUB-200)

```bash
# Process multiple classes
./run_cub200_batch.sh

# Customize
CLASS_LIMIT=20 \
MODELS="icarl der derpp lwf coda_prompt" \
LAYERS="layer4 layer3" \
ROOT_OUTPUT_DIR=gradcam_outputs \
./run_cub200_batch.sh
```

**Environment Variables:**
- `MODEL_NAME`: Model name (default: icarl)
- `DATASET`: Dataset identifier (default: seq-cifar100)
- `DATASET_TAG`: Dataset tag for checkpoints (default: cifar100 or cub200)
- `SEED`: Random seed (default: 0)
- `CHECKPOINT_DIR`: Checkpoint directory (default: checkpoints)
- `SPLIT`: Data split (default: test)
- `SAMPLE_INDEX`: Sample index (default: 0)
- `LAYER_NAME`: Layer for Grad-CAM (default: layer4)
- `BASELINE_TAG`: Baseline checkpoint tag (default: original)
- `SACK_TAG`: SACK checkpoint tag (default: sack)
- `OUTPUT_DIR`: Output directory

**Direct Python Usage:**

```bash
python gradcamviz.py \
    --model-name=icarl \
    --dataset=seq-cifar100 \
    --dataset-tag=cifar100 \
    --seed=0 \
    --target-class=0 \
    --checkpoint-dir=checkpoints \
    --split=test \
    --sample-index=0 \
    --layer-name=layer4 \
    --baseline-tag=original \
    --sack-tag=sack \
    --output-dir=gradcam_outputs/icarl_apple
```

---

## Model-Specific Configurations

### Supported Models

- **iCaRL**: `--model=icarl --buffer_size=2000 --model_config=best`
- **DER**: `--model=der --buffer_size=2000 --model_config=best`
- **DER++**: `--model=derpp --buffer_size=2000 --model_config=best`
- **LwF**: `--model=lwf --lr=0.003` (or `--lr=0.03` for ImageNet-R)
- **CODA-Prompt**: `--model=coda_prompt --model_config=best`
- **BiC**: `--model=bic`
- **GSS**: `--model=gss`
- **HAL**: `--model=hal`
- **MER**: `--model=mer`
- **DualPrompt**: `--model=dualprompt`
- **DAP**: `--model=dap`
- **ER**: `--model=er`
- **GEM**: `--model=gem`

### Supported Datasets

- `seq-cifar100`: CIFAR-100 (32x32)
- `seq-cifar100-224`: CIFAR-100 (224x224, for ViT models)
- `seq-cifar100-c`: CIFAR-100-C (corrupted)
- `seq-cub200`: CUB-200
- `seq-imagenet-r`: ImageNet-R
- `seq-inaturalist-300`: iNaturalist (300 classes)
- `seq-inaturalist-1000`: iNaturalist (1000 classes)
- `seq-inaturalist`: Full iNaturalist
- `seq-cifar10`: CIFAR-10
- `seq-pmnist`: Permuted MNIST

### Model-Specific Scripts

Scripts are organized by dataset and model:

```
SACK_Scripts/
├── modelcogcl/          # Scripts with SACK enabled
│   ├── cifar100/
│   ├── cub200/
│   ├── imagenet-r/
│   └── pmnist/
└── modelOG/             # Baseline scripts (no SACK)
    ├── cifar100/
    ├── cub200/
    ├── imagenet-r/
    └── pmnist/
```

**Example: Run iCaRL on CIFAR-100 with SACK**
```bash
./SACK_Scripts/modelcogcl/cifar100/icarl.sh
```

**Example: Run iCaRL on CIFAR-100 baseline**
```bash
./SACK_Scripts/modelOG/cifar100/icarl.sh
```

---

## Inference-Only Mode

Evaluate trained checkpoints without training:

```bash
python main.py \
    --dataset=seq-cifar100 \
    --model=icarl \
    --inference_only=1 \
    --loadcheck=/path/to/checkpoint.pt \
    --cog_cl=1 \
    --sack_scores_type=0 \
    --enable_other_metrics=True \
    --permute_classes=True \
    --seed=0 \
    --buffer_size=2000 \
    --model_config=best
```

### CIFAR-100-C Inference

```bash
python main.py \
    --dataset=seq-cifar100-c \
    --model=icarl \
    --inference_only=1 \
    --loadcheck=/path/to/checkpoint.pt \
    --cifar_c_corruption=gaussian_noise \
    --cifar_c_severity=1 \
    --cog_cl=1 \
    --sack_scores_type=0 \
    --seed=0 \
    --buffer_size=2000 \
    --model_config=best
```

---

## Checkpoint Management

### Saving Checkpoints

```bash
# Save after each task
--savecheck=task

# Save at end
--savecheck=end

# Custom checkpoint name
--ckpt_name=my-experiment-name
```

### Loading Checkpoints

```bash
# Load specific checkpoint
--loadcheck=/path/to/checkpoint.pt

# Load from checkpoint directory
--checkpoint_dir=checkpoints \
--checkpoint_prefix=icarl-cifar100-sack-seed-0
```

---

## WandB Integration

### Basic Configuration

```bash
--wandb_entity=your-entity \
--wandb_project=your-project \
--wandb_name=experiment-name
```

### Offline Mode

```bash
export WANDB_MODE=offline
```

### Disable WandB

```bash
export WANDB_DISABLE=true
```

---

## Performance Metrics

### Enable Performance Logging

```bash
--log_perf_metrics=1
```

### Enable Additional Metrics

```bash
--enable_other_metrics=True
```

This computes:
- Forward Transfer
- Backward Transfer
- Forgetting

---

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**
   - Verify checkpoint path is correct
   - Check `--checkpoint_dir` and `--checkpoint_prefix` settings
   - Ensure checkpoint naming matches expected pattern

2. **CUDA Out of Memory**
   - Reduce `--batch_size`
   - Reduce `--buffer_size` for rehearsal methods
   - Use gradient accumulation

3. **Dataset Not Found**
   - Run dataset download scripts
   - Check dataset paths in configuration
   - Verify dataset format matches expected structure

4. **Model Configuration Errors**
   - Ensure `--model_config=best` matches your model
   - Check model-specific requirements (e.g., buffer_size for rehearsal methods)

### Debug Mode

Enable verbose logging:
```bash
python main.py ... --verbose
```

### Check System Requirements

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Script Organization

### Main Scripts

- `main.py`: Core training and evaluation script
- `SACK_ECE.py`: ECE computation utility
- `gradcamviz.py`: Grad-CAM visualization tool
- `metriccc.py`: Additional metrics computation

### Script Directories

- `SACK_Scripts/`: High-level execution scripts
- `scripts/`: Utility scripts (download, preprocessing, etc.)
- `SACK_Scripts/modelcogcl/`: SACK-enabled model scripts
- `SACK_Scripts/modelOG/`: Baseline model scripts

---

## Example Workflows

### Complete CIFAR-100 Evaluation

```bash
# 1. Train baseline
./SACK_Scripts/OG_CIFAR-100_RUNTIME.sh

# 2. Train with SACK
./SACK_Scripts/SACK_CIFAR-100_RUNTIME.sh

# 3. Compute ECE
./SACK_Scripts/run_ece_comparison.sh 0

# 4. Evaluate robustness
export ICARL_BASELINE_CKPT=checkpoints/icarl-cifar100-original-seed-0_final.pt
export ICARL_SACK_CKPT=checkpoints/icarl-cifar100-sack-seed-0_final.pt
./SACK_Scripts/sack_cifar100c_robustness.sh

# 5. Generate visualizations
./rungradcamviz.sh "apple"
```

### Multi-Dataset Comparison

```bash
# CIFAR-100
./SACK_Scripts/SACK_CIFAR-100_RUNTIME.sh

# CUB-200
./SACK_Scripts/SACK_CUB-200_RUNTIME.sh

# ImageNet-R
./SACK_Scripts/SACK_IMAGENET-R_RUNTIME.sh

# iNaturalist-300
./SACK_Scripts/SACK_inaturalist300.sh
```

### Hyperparameter Search

```bash
# Threshold sweep
./SACK_Scripts/sack_cifar100_threshold_sweep.sh

# Multiple seeds
for seed in 0 1 2 3 4; do
    SEED=$seed ./SACK_Scripts/SACK_CIFAR-100_RUNTIME.sh
done
```

---

## Additional Resources

- `REPRODUCIBILITY.md`: Model reproduction status
- `docs/`: Detailed documentation
- `requirements.txt`: Python dependencies
- `requirements-optional.txt`: Optional dependencies

---

## Support

For issues or questions:
1. Check existing documentation in `docs/`
2. Review script comments for usage examples
3. Examine existing checkpoint naming patterns
4. Verify dataset and model compatibility

---

## Notes

- All scripts use `set -euo pipefail` for error handling
- Default WandB entity: `abcxyz8431-cl` (modify as needed)
- Checkpoint naming convention: `{model}-{dataset}-{variant}-seed-{seed}`
- Most scripts support environment variable overrides
- GPU assignments in parallel scripts may need adjustment for your system

---

## Acknowledgments

This repository is built upon [Mammoth](https://github.com/aimagelab/mammoth), a benchmark Continual Learning framework for PyTorch. We gratefully acknowledge the Mammoth team for providing the foundational framework that enabled the development of SACK.

**Mammoth Contributors:**
- Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara
- Original Mammoth repository: https://github.com/aimagelab/mammoth

