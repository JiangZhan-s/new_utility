#!/bin/bash
#SBATCH --job-name=QSWEEP_C10
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --account=a_xmwang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-24%3
#SBATCH --output=logs/qsweep_c10_%A_%a.out
#SBATCH --error=logs/qsweep_c10_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

PYTHON=/share/home/202521044325/miniconda3/envs/aigc39/bin/python
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Freeze the generator-quality calibration measured on the 20-round real-only
# FedAvg probe trajectory from outputs/quality_trajectory_v2:
#   g_data = 7.028657035286398
#   g_diff = 0.4343768451925214
#   chi    = g_diff / g_data = 0.06180083094278078
# The sweep therefore studies budget/seed sensitivity without recalibrating chi
# separately for each run.
GAP_RATIO="${GAP_RATIO:-0.06180083094278078}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/quality_budget_multiseed_v1}"

budgets=(0.1 0.2 0.3 0.4 0.5)
seeds=(0 1 2 3 4)
idx=${SLURM_ARRAY_TASK_ID:?}
seed_count=${#seeds[@]}
budget=${budgets[$((idx / seed_count))]}
seed=${seeds[$((idx % seed_count))]}

mkdir -p logs

echo "[job] id=${SLURM_JOB_ID:-unset} array=${SLURM_ARRAY_TASK_ID:-unset} node=$(hostname) start=$(date -Is)"
echo "[setting] budget_fraction=$budget seed=$seed gap_ratio=$GAP_RATIO output=$OUTPUT_ROOT"
echo "[cuda] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

"$PYTHON" - <<'PY'
import os, torch
print('[preflight] torch=', torch.__version__, 'compiled_cuda=', torch.version.cuda)
print('[preflight] CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('[preflight] available=', torch.cuda.is_available(), 'count=', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('CUDA preflight failed')
x=torch.zeros(1, device='cuda')
print('[preflight] device=', torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
PY

# Four regimes:
#   continuous      : budget-feasible personalized continuous enhancement
#   three_state     : budget-feasible discrete raw/full enhancement
#   no_aigc_budget  : budget-feasible real-data-only participant selection
#   full_balance_all: all clients fully enhanced; unconstrained oracle/reference
#
# Fixed original-data compute is used by experiment_quality for every regime.
"$PYTHON" -m baqp.experiment_quality \
  --dataset cifar10 \
  --data-root data \
  --aigc-root aigc_imgs \
  --seed "$seed" \
  --clients 6 \
  --alpha 0.3 \
  --split-seed 2026 \
  --budget-fraction "$budget" \
  --model cnn \
  --rounds 100 \
  --local-epochs 2 \
  --steps 5 \
  --batch 32 \
  --device cuda \
  --eval-every 10 \
  --generator-gap-source value \
  --generator-gap-ratio "$GAP_RATIO" \
  --output "$OUTPUT_ROOT" \
  --methods continuous three_state no_aigc_budget full_balance_all \
  --resume

# Compact per-task audit. Do not assert equal total gradient steps across methods:
# budget-feasible methods may choose different participant sets. The fixed-compute
# invariant is per selected client: local steps depend only on original n_k.
"$PYTHON" - "$OUTPUT_ROOT" "$budget" "$seed" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])/'cifar10'/'cnn'/f'alpha_0.3_budget_{sys.argv[2]}'/f'seed_{sys.argv[3]}'
methods=['continuous','three_state','no_aigc_budget','full_balance_all']
print('\n=== TASK SUMMARY ===')
print('method\ttest_acc\ttest_ce\tobjective\tactive\tmean_u_active\tgradient_steps\tsynthetic_draws')
for method in methods:
    p=root/method/'result.json'
    if not p.exists():
        print(method+'\tMISSING')
        continue
    r=json.loads(p.read_text())
    c=r['contract']; active=[i for i,a in enumerate(c['active']) if a]
    u=[float(c['u'][i]) for i in active]
    mean_u=sum(u)/len(u) if u else 0.0
    print('%s\t%.6f\t%.6f\t%.9f\t%s\t%.6f\t%d\t%d' % (
        method,float(r['test']['acc']),float(r['test']['ce']),
        float(c['objective_normalized']),active,mean_u,
        int(r['gradient_steps']),int(r['actual_synthetic_draws'])))
PY

echo "[job] end=$(date -Is)"
