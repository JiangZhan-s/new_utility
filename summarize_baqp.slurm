#!/bin/bash
#SBATCH --job-name=BAQP_SUMMARY
#SBATCH --partition=cpuXeon6458
#SBATCH --account=a_xmwang
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/summary_%j.out
#SBATCH --error=logs/summary_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
/share/home/202521044325/miniconda3/envs/aigc39/bin/python -m baqp.summarize "${OUTPUT_ROOT:-outputs/real_baqp}"
