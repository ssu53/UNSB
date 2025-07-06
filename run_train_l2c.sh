#!/bin/bash

#SBATCH --account=pasteur
#SBATCH --partition=pasteur
#SBATCH --exclude=pasteur1,pasteur2,pasteur3,pasteur4,pasteur8,pasteur9
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --job-name=l2c
#SBATCH --output=/pasteur2/u/shiye/UNSB/outputs/%x_%j.out
#SBATCH --error=/pasteur2/u/shiye/UNSB/outputs/%x_%j.err
#SBATCH --time=48:00:00

set -ex



source ~/.bashrc_user
conda activate unsb_python310

cd /pasteur2/u/shiye/UNSB

python train.py \
    --dataroot ./datasets/cityscapes \
    --name city_SB \
    --mode sb \
    --lambda_SB 1.0 \
    --lambda_NCE 1.0 \
    --direction B2A \
    --gpu_ids 0 \
