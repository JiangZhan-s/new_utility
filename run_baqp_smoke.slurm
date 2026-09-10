#!/bin/bash
#SBATCH --job-name=BAQP_SMOKE
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --account=a_xmwang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baqp_smoke_%j.out
#SBATCH --error=logs/baqp_smoke_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

PYTHON=/share/home/202521044325/miniconda3/envs/aigc39/bin/python
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Do not rewrite CUDA_VISIBLE_DEVICES here. Slurm owns GPU visibility.  The
# recommended submission command unsets any stale login-shell value before
# sbatch, after which Slurm injects the allocation-specific value.
echo "=== BAQP GPU smoke preflight ==="
echo "job=${SLURM_JOB_ID:-unset} node=$(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

"$PYTHON" - <<'PY'
import os
import torch
print('python cuda preflight:')
print('  torch=', torch.__version__)
print('  torch.version.cuda=', torch.version.cuda)
print('  CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('  cuda.is_available=', torch.cuda.is_available())
print('  cuda.device_count=', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('CUDA preflight failed before BAQP training')
x = torch.zeros(1, device='cuda')
print('  selected_device=', torch.cuda.current_device())
print('  device_name=', torch.cuda.get_device_name(torch.cuda.current_device()))
print('  allocation_test=', x.item())
PY

echo "=== mechanism/unit checks ==="
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m baqp.check_kkt

echo "=== single experiment ==="
DATASET=${DATASET:-cifar10}
SEED=${SEED:-0}
ROUNDS=${ROUNDS:-100}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
CLIENTS=${CLIENTS:-6}
ALPHA=${ALPHA:-0.3}
BUDGET_FRACTION=${BUDGET_FRACTION:-0.4}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/real_baqp_two_part_epochs_smoke}

"$PYTHON" -m baqp.experiment \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --model cnn \
  --rounds "$ROUNDS" \
  --steps 5 \
  --local-epochs "$LOCAL_EPOCHS" \
  --batch "$BATCH_SIZE" \
  --clients "$CLIENTS" \
  --alpha "$ALPHA" \
  --budget-fraction "$BUDGET_FRACTION" \
  --methods continuous three_state \
  --output "$OUTPUT_ROOT" \
  --device cuda \
  --resume

echo "=== BAQP smoke completed ==="
