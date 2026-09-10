"""Pure CPU sweep for the generator-aware BAQP mechanism."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .data import load_real, split_train
from .mechanism_quality import instance, solve
from .quality import IMFL_CIFAR10_GAP_RATIO


def synthetic_count_proxy(z, counts, u):
    n=np.asarray(counts,dtype=float).sum(1)
    gamma=np.asarray(z['gamma'],dtype=float)
    u=np.asarray(u,dtype=float)
    den=1.0-gamma*u
    return np.divide(n*gamma*u,den,out=np.zeros_like(n),where=den>0)


def summarize(z,counts,result,method,budget,gap):
    if result.get('status')!='ok':
        return dict(method=method,budget_fraction=float(budget),generator_gap_ratio=float(gap),
                    status=result.get('status','unknown'))
    active=np.asarray(result['active'],dtype=bool); u=np.asarray(result['u'],dtype=float)
    n=np.asarray(counts,dtype=float).sum(1); syn=synthetic_count_proxy(z,counts,u)
    interior=active&(u>1e-6)&(u<1-1e-6); full=active&(u>=1-1e-6); raw=active&(u<=1e-6)
    return dict(method=method,budget_fraction=float(budget),generator_gap_ratio=float(gap),status='ok',
                objective_normalized=float(result['objective_normalized']),
                label_component=float(result['objective_components']['label']),
                generator_component=float(result['objective_components']['generator']),
                noise_component=float(result['objective_components']['noise']),
                active_clients=int(active.sum()),raw_clients=int(raw.sum()),
                interior_clients=int(interior.sum()),full_clients=int(full.sum()),
                mean_u_active=float(u[active].mean()) if active.any() else 0.,
                budget=float(z['budget']),full_budget=float(z['full_budget']),
                total_payment=float(result['total_payment']),
                budget_utilization=float(result['total_payment']/z['budget']) if z['budget']>0 else 0.,
                active_original_samples=int(n[active].sum()),
                synthetic_samples_proxy=float(syn[active].sum()),
                active_indices=json.dumps(np.flatnonzero(active).tolist()),
                u=json.dumps([float(x) for x in u]),
                normalized_absolute_gap=float(result.get('normalized_absolute_gap',np.nan)),
                solver_failures=int(result.get('solver_failures',0)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],default='cifar10')
    p.add_argument('--data-root',default='data')
    p.add_argument('--clients',type=int,default=6)
    p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--rounds',type=int,default=100); p.add_argument('--steps',type=int,default=5)
    p.add_argument('--batch',type=int,default=32); p.add_argument('--mu',type=float,default=.05)
    p.add_argument('--smooth',type=float,default=.55); p.add_argument('--lr',type=float,default=None)
    p.add_argument('--base-cost-total',type=float,default=1.0)
    p.add_argument('--aigc-unit-cost',type=float,default=1.0)
    p.add_argument('--generator-gap-ratios',type=float,nargs='+',
                   default=[0.0,IMFL_CIFAR10_GAP_RATIO,.4],
                   help='chi=g_diff/g_data; 0 is perfect-AIGC ablation, 0.54/1.75 is IMFL CIFAR-10 reference')
    p.add_argument('--budgets',type=float,nargs='+',default=[.02,.04,.06,.08,.10,.15,.20,.30,.40,.50])
    p.add_argument('--output',default='outputs/generator_quality_sweep/cifar10_seed2026.csv')
    args=p.parse_args()
    if args.clients<1 or args.alpha<=0 or any(x<0 for x in args.generator_gap_ratios):
        p.error('invalid clients/alpha/generator gap')
    _,y,_,_=load_real(args.data_root,args.dataset); classes=int(y.max())+1
    _,_,counts=split_train(y,classes,args.clients,args.alpha,args.split_seed)
    rows=[]
    for gap in args.generator_gap_ratios:
        for budget in args.budgets:
            z=instance(counts,seed=args.split_seed,budget_fraction=budget,
                       rounds=args.rounds,steps=args.steps,batch=args.batch,
                       mu=args.mu,smooth=args.smooth,lr=args.lr,
                       base_cost_total=args.base_cost_total,aigc_unit_cost=args.aigc_unit_cost,
                       generator_gap_ratio=gap,
                       generator_gap_source=('perfect_aigc_ablation' if gap==0 else 'sensitivity_or_reference'))
            for method,kwargs in [('continuous',{}),('three_state',{'discrete':True}),('no_aigc',{'raw_only':True})]:
                result=solve(z,**kwargs)
                row=summarize(z,counts,result,method,budget,gap); rows.append(row)
                if row['status']=='ok':
                    print(json.dumps({k:row[k] for k in ('method','budget_fraction','generator_gap_ratio',
                          'objective_normalized','active_clients','interior_clients','full_clients',
                          'mean_u_active','budget_utilization')},separators=(',',':')),flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps(dict(config=vars(args),counts=counts.tolist(),rows=rows),indent=2,allow_nan=False))
    print(f'Wrote {out} and {out.with_suffix(".json")}')


if __name__=='__main__':
    main()
