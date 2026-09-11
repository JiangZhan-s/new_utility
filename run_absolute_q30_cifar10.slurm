#!/bin/bash
#SBATCH --job-name=ABSQ30_C10
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --account=a_xmwang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/absq30_c10_%j.out
#SBATCH --error=logs/absq30_c10_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON=/share/home/202521044325/miniconda3/envs/aigc39/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[job] id=${SLURM_JOB_ID:-unset} node=$(hostname) start=$(date -Is)"
echo "[cuda] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"$PYTHON" - <<'PY'
import torch, os
print('[preflight] torch=',torch.__version__,'cuda=',torch.version.cuda)
print('[preflight] visible=',os.environ.get('CUDA_VISIBLE_DEVICES'),'available=',torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit('CUDA preflight failed')
print('[preflight] gpu=',torch.cuda.get_device_name(0))
PY

COMMON=(
  --dataset cifar10
  --data-root data
  --aigc-root aigc_imgs
  --clients 30
  --alpha 0.3
  --split-seed 2026
  --seed 0
  --budget-fraction 0.4
  --model cnn
  --rounds 100
  --local-epochs 2
  --steps 5
  --batch 32
  --device cuda
  --eval-every 10
  --base-cost-total 1.0
  --aigc-unit-cost 1.0
  --aigc-convex-cost 0.1
  --generator-gap-source value
  --generator-gap-ratio 0.06180083094278078
  --heuristic-restarts 4
  --heuristic-maxiter 180
  --output outputs/absolute_q_30_v1
  --methods personalized uniform_budget no_aigc repair_reference
)

# First materialize and print contracts. The training call reuses those exact
# contracts, so the heuristic is not solved twice.
"$PYTHON" -m baqp.experiment_absolute "${COMMON[@]}" --prepare-only
"$PYTHON" -m baqp.experiment_absolute "${COMMON[@]}" --resume

"$PYTHON" - <<'PY'
import json, numpy as np
from pathlib import Path
root=Path('outputs/absolute_q_30_v1/cifar10/cnn/clients_30_alpha_0.3_budget_0.4/seed_0')
methods=['personalized','uniform_budget','no_aigc','repair_reference']
print('\n=== ABSOLUTE-q 30-CLIENT FIXED-COMPUTE COMPARISON ===')
print('method\ttest_acc\ttest_ce\tobjective\tpayment\tmean_q\tmax_q\tmean_syn_frac\tmax_syn_frac\tsteps')
steps=[]
for m in methods:
    r=json.loads((root/m/'result.json').read_text()); c=r['contract']
    q=np.asarray(c['q_synthetic'],float); f=np.asarray(c['synthetic_fraction'],float)
    print(f"{m}\t{r['test']['acc']:.6f}\t{r['test']['ce']:.6f}\t{c['objective_normalized']:.9f}\t"
          f"{c['total_payment']:.6f}\t{q.mean():.2f}\t{q.max():.2f}\t{f.mean():.4f}\t{f.max():.4f}\t{r['gradient_steps']}")
    steps.append(int(r['gradient_steps']))
if len(set(steps))!=1: raise SystemExit(f'fixed-compute invariant FAILED: {steps}')
meta=json.loads((root/'metadata.json').read_text())
print('client size min/max =',meta['fixed_client_size_min'],meta['fixed_client_size_max'])
print('budget =',meta['instance']['budget'],'all_base_cost =',meta['instance']['all_base_cost'],
      'reference_repair_budget =',meta['instance']['reference_repair_budget'])
print('fixed-compute invariant: PASS (%d gradient steps each)' % steps[0])
PY

echo "[job] end=$(date -Is)"
