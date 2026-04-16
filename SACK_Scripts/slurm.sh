#!/bin/bash
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=shivank2@umbc.edu                          # Where to send mail
#SBATCH -D .
#SBATCH --job-name=blabla
#SBATCH --output=blabla.out
#SBATCH --error=blabla.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5000
#SBATCH --time=2:00:00


conda activate clip_dissect
cd ./mammoth
bash PUT_YOUR_PATH
