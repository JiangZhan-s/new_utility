#!/bin/bash
#SBATCH --job-name=BAQP_REAL
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --account=a_xmwang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-14%3
#SBATCH --output=logs/baqp_%A_%a.out
#SBATCH --error=logs/baqp_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON=/share/home/202521044325/miniconda3/envs/aigc39/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Slurm must own GPU visibility. Do not assign CUDA_VISIBLE_DEVICES manually in
# this script. Submission should preferably unset a stale login-shell value.
echo "[cuda preflight] job=${SLURM_JOB_ID:-unset} node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"$PYTHON" - <<'PY'
import os, torch
print('[cuda preflight] torch=',torch.__version__,'compiled_cuda=',torch.version.cuda)
print('[cuda preflight] CUDA_VISIBLE_DEVICES=',os.environ.get('CUDA_VISIBLE_DEVICES'))
print('[cuda preflight] available=',torch.cuda.is_available(),'count=',torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count()<1:
    raise SystemExit('CUDA preflight failed')
x=torch.zeros(1,device='cuda')
print('[cuda preflight] device=',torch.cuda.current_device(),torch.cuda.get_device_name(torch.cuda.current_device()))
PY

read -r -a datasets <<< "${DATASETS:-cifar10 cifar100 fmnist}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4}"
index=${SLURM_ARRAY_TASK_ID:-0}
seed_count=${#seeds[@]}
dataset=${datasets[$((index / seed_count))]}
seed=${seeds[$((index % seed_count))]}
args=(--dataset "$dataset" --seed "$seed" --model "${MODEL:-cnn}"
      --rounds "${ROUNDS:-100}" --steps "${STEPS:-5}" --local-epochs "${LOCAL_EPOCHS:-2}"
      --batch "${BATCH_SIZE:-32}"
      --clients "${CLIENTS:-6}" --alpha "${ALPHA:-0.3}" --budget-fraction "${BUDGET_FRACTION:-0.4}"
      --output "${OUTPUT_ROOT:-outputs/real_baqp_two_part_epochs}" --device cuda --resume)
if [[ -n "${METHODS:-}" ]]; then
    read -r -a methods <<< "$METHODS"
    args+=(--methods "${methods[@]}")
fi
if [[ "${RESIDUAL:-0}" == 1 ]]; then args+=(--residual); fi
"$PYTHON" -m baqp.experiment "${args[@]}"
