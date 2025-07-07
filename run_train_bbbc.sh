#!/bin/bash

#SBATCH --account=pasteur
#SBATCH --partition=pasteur
#SBATCH --exclude=pasteur1,pasteur2,pasteur3,pasteur4,pasteur5,pasteur6,pasteur7,pasteur-hgx-1,pasteur-hgx-2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --job-name=bbbc
#SBATCH --output=/pasteur2/u/shiye/UNSB/outputs/%x_%j.out
#SBATCH --error=/pasteur2/u/shiye/UNSB/outputs/%x_%j.err
#SBATCH --time=72:00:00

set -ex



source ~/.bashrc_user
conda activate unsb_python310

cd /pasteur2/u/shiye/UNSB


python -u train.py \
    --cell_dataset_name bbbc021 \
    --dataset_mode unaligned_cell \
    --name bbbc_SB \
    --mode sb \
    --lambda_SB 1.0 \
    --lambda_NCE 1.0 \
    --direction AtoB \
    --gpu_ids 0 \
    --cond_dim 1024 \
    --display_id -1 \
    --n_epochs 3 \
    --n_epochs_decay 3 \
    --save_latest_freq 20000 \
    --save_epoch_freq 1 \
    --save_by_iter \