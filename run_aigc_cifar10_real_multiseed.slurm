#!/bin/bash
#SBATCH --job-name=BAQP_C10
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --account=a_xmwang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%3
#SBATCH --output=logs/cifar10_%A_%a.out
#SBATCH --error=logs/cifar10_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export DATASETS=cifar10
bash run_baqp_real.slurm
