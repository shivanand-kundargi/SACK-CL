## 🛠 Setup

```bash
cd sack
pip install -r requirements.txt      # PyTorch ≥ 2.1 required

# Prepare iNaturalist (one-time, before running seq-inaturalist)
python scripts/download_inaturalist.py

# Without SACK (baseline)
python main.py \
        --dataset=seq-cifar100-224 \
        --model=coda_prompt \
        --model_config=best\
        --sack 0 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
        --wandb_name=SACK-$seed \
        --enable_other_metrics=True \
        --permute_classes=True \
        --seed=$seed

# With SACK
python main.py \
        --dataset=seq-cifar100-224 \
        --model=coda_prompt \
        --model_config=best\
        --sack 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
        --wandb_name=SACK-$seed \
        --enable_other_metrics=True \
        --permute_classes=True \
        --seed=$seed
