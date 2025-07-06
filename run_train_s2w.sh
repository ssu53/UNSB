#!/bin/bash

#SBATCH --account=pasteur
#SBATCH --partition=pasteur
#SBATCH --exclude=pasteur1,pasteur2,pasteur3,pasteur4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --job-name=sum2win
#SBATCH --output=/pasteur2/u/shiye/UNSB/outputs/%x_%j.out
#SBATCH --error=/pasteur2/u/shiye/UNSB/outputs/%x_%j.err
#SBATCH --time=48:00:00

set -ex



source ~/.bashrc_user
conda activate unsb_python310

cd /pasteur2/u/shiye/UNSB/vgg_sb

bash scripts/train_sc_sum2win_main.sh
