"""Summarize the generator-aware budget x seed sweep.

Reads result.json files produced by run_quality_budget_multiseed_cifar10.slurm
and reports per-budget mean/std plus paired seed-wise deltas against continuous.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np


DEFAULT_METHODS = ['continuous','three_state','no_aigc_budget','full_balance_all']


def mean_std(values):
    x=np.asarray(values,dtype=float)
    if len(x)==0: return float('nan'),float('nan')
    return float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else 0.0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='outputs/quality_budget_multiseed_v1')
    p.add_argument('--dataset',default='cifar10')
    p.add_argument('--model',default='cnn')
    p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--budgets',type=float,nargs='+',default=[.1,.2,.3,.4,.5])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--methods',nargs='+',default=DEFAULT_METHODS)
    p.add_argument('--output',default=None,help='CSV output; default <root>/summary.csv')
    args=p.parse_args()

    root=Path(args.root)
    rows=[]; missing=[]
    for budget in args.budgets:
        bstr=str(float(budget))
        for seed in args.seeds:
            base=root/args.dataset/args.model/f'alpha_{args.alpha}_budget_{bstr}'/f'seed_{seed}'
            for method in args.methods:
                path=base/method/'result.json'
                if not path.exists():
                    missing.append(str(path)); continue
                r=json.loads(path.read_text()); c=r['contract']
                active=np.asarray(c['active'],dtype=bool); u=np.asarray(c['u'],dtype=float)
                active_u=u[active]
                total_payment=c.get('total_payment')
                budget_value=c.get('budget')
                rows.append(dict(
                    budget_fraction=float(budget),seed=int(seed),method=method,
                    test_acc=float(r['test']['acc']),test_ce=float(r['test']['ce']),
                    val_acc=float(r['validation']['acc']),val_ce=float(r['validation']['ce']),
                    objective=float(c['objective_normalized']),active_clients=int(active.sum()),
                    raw_clients=int(np.sum(active_u<=1e-6)),
                    interior_clients=int(np.sum((active_u>1e-6)&(active_u<1-1e-6))),
                    full_clients=int(np.sum(active_u>=1-1e-6)),
                    mean_u_active=float(active_u.mean()) if len(active_u) else 0.0,
                    gradient_steps=int(r['gradient_steps']),
                    real_draws=int(r['actual_real_draws']),synthetic_draws=int(r['actual_synthetic_draws']),
                    total_payment=(None if total_payment is None else float(total_payment)),
                    budget=(None if budget_value is None else float(budget_value)),
                ))

    if not rows:
        raise SystemExit('No result.json files found')
    out=Path(args.output) if args.output else root/'summary.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print('=== PER-BUDGET ACCURACY (mean ± sample std) ===')
    for budget in args.budgets:
        print(f'\nBudget fraction {budget:.1f}')
        for method in args.methods:
            vals=[r['test_acc'] for r in rows if r['budget_fraction']==float(budget) and r['method']==method]
            m,s=mean_std(vals)
            print(f'  {method:18s} n={len(vals)}  acc={m:.4f} ± {s:.4f}')

        # Paired comparisons use only seeds for which both methods completed.
        by={(r['seed'],r['method']):r for r in rows if r['budget_fraction']==float(budget)}
        if any((seed,'continuous') in by for seed in args.seeds):
            for baseline in [m for m in args.methods if m!='continuous']:
                diffs=[]
                for seed in args.seeds:
                    if (seed,'continuous') in by and (seed,baseline) in by:
                        diffs.append(by[(seed,'continuous')]['test_acc']-by[(seed,baseline)]['test_acc'])
                dm,ds=mean_std(diffs)
                wins=sum(d>0 for d in diffs)
                print(f'    Δ continuous-{baseline:14s}: {dm:+.4f} ± {ds:.4f}  wins={wins}/{len(diffs)}')

    # Mechanism structure for continuous.
    print('\n=== CONTINUOUS ALLOCATION STRUCTURE ===')
    for budget in args.budgets:
        q=[r for r in rows if r['budget_fraction']==float(budget) and r['method']=='continuous']
        if not q: continue
        a,asdev=mean_std([r['active_clients'] for r in q])
        i,isdev=mean_std([r['interior_clients'] for r in q])
        f,fsdev=mean_std([r['full_clients'] for r in q])
        u,usdev=mean_std([r['mean_u_active'] for r in q])
        print(f'  B={budget:.1f}: active={a:.2f}±{asdev:.2f}, interior={i:.2f}±{isdev:.2f}, '
              f'full={f:.2f}±{fsdev:.2f}, mean_u_active={u:.3f}±{usdev:.3f}')

    # A machine-readable aggregate for plotting/table generation later.
    aggregate=[]
    for budget in args.budgets:
        for method in args.methods:
            q=[r for r in rows if r['budget_fraction']==float(budget) and r['method']==method]
            if not q: continue
            am,asd=mean_std([r['test_acc'] for r in q])
            cm,csd=mean_std([r['test_ce'] for r in q])
            aggregate.append(dict(budget_fraction=float(budget),method=method,n=len(q),
                                  test_acc_mean=am,test_acc_std=asd,
                                  test_ce_mean=cm,test_ce_std=csd))
    (root/'summary.json').write_text(json.dumps(dict(rows=rows,aggregate=aggregate,missing=missing),indent=2))
    print(f'\nWrote {out} and {root/"summary.json"}')
    if missing:
        print(f'WARNING: {len(missing)} result files are missing; rerun incomplete Slurm tasks before final reporting.')


if __name__=='__main__':
    main()
