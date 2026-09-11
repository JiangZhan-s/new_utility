#!/bin/bash
#SBATCH --job-name=QTRAJ_C10
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --account=a_xmwang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qtraj_c10_%j.out
#SBATCH --error=logs/qtraj_c10_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

PYTHON=/share/home/202521044325/miniconda3/envs/aigc39/bin/python
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Slurm owns GPU visibility. Do not set CUDA_VISIBLE_DEVICES manually.
echo "[job] id=${SLURM_JOB_ID:-unset} node=$(hostname) start=$(date -Is)"
echo "[job] cwd=$PWD"
echo "[job] python=$PYTHON"
echo "[cuda] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

"$PYTHON" - <<'PY'
import os, torch
print('[preflight] torch=', torch.__version__, 'compiled_cuda=', torch.version.cuda)
print('[preflight] CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('[preflight] available=', torch.cuda.is_available(), 'count=', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('CUDA preflight failed')
x = torch.zeros(1, device='cuda')
print('[preflight] device=', torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
PY

# This resumes the prepare-only run already stored in outputs/quality_trajectory_v2.
# Only the three non-redundant learning regimes are trained:
#   continuous        : trajectory-calibrated interior mixture
#   selected_no_aigc  : same six participants, u=0
#   selected_full     : same six participants, u=1
# In the current prepare-only solution, fedavg_all / no_aigc_budget / three_state
# have the same all-six-client u=0 learning contract as selected_no_aigc, so
# retraining all of them would be exactly redundant under deterministic seeds.
"$PYTHON" -m baqp.experiment_quality \
  --dataset cifar10 \
  --data-root data \
  --aigc-root aigc_imgs \
  --clients 6 \
  --alpha 0.3 \
  --split-seed 2026 \
  --budget-fraction 0.4 \
  --model cnn \
  --rounds 100 \
  --local-epochs 2 \
  --steps 5 \
  --batch 32 \
  --device cuda \
  --eval-every 10 \
  --generator-gap-source trajectory \
  --quality-device cpu \
  --quality-seed 2026 \
  --quality-samples-per-client 512 \
  --quality-batch 256 \
  --quality-rounds 20 \
  --quality-local-steps 5 \
  --quality-probe-every 1 \
  --output outputs/quality_trajectory_v2 \
  --methods continuous selected_no_aigc selected_full \
  --resume

# Print the decisive comparison and verify the fixed-compute invariant.
"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path('outputs/quality_trajectory_v2/cifar10/cnn/alpha_0.3_budget_0.4/seed_0')
methods = ['continuous', 'selected_no_aigc', 'selected_full']
rows = []
for method in methods:
    path = root / method / 'result.json'
    if not path.exists():
        raise SystemExit(f'missing result: {path}')
    r = json.loads(path.read_text())
    rows.append((
        method,
        float(r['test']['acc']),
        float(r['test']['ce']),
        int(r['gradient_steps']),
        int(r['actual_real_draws']),
        int(r['actual_synthetic_draws']),
    ))

print('\n=== DECISIVE FIXED-COMPUTE COMPARISON ===')
print('method\ttest_acc\ttest_ce\tgradient_steps\treal_draws\tsynthetic_draws')
for row in rows:
    print('%s\t%.6f\t%.6f\t%d\t%d\t%d' % row)

steps = {row[3] for row in rows}
if len(steps) != 1:
    raise SystemExit(f'fixed-compute invariant FAILED: gradient_steps={sorted(steps)}')

acc = {row[0]: row[1] for row in rows}
print('ACC_continuous - ACC_selected_full = %.6f' % (acc['continuous'] - acc['selected_full']))
print('ACC_continuous - ACC_selected_no_aigc = %.6f' % (acc['continuous'] - acc['selected_no_aigc']))
print('fixed-compute invariant: PASS (%d gradient steps each)' % next(iter(steps)))
PY

echo "[job] end=$(date -Is)"
