"""Pure mechanism sweep for data-calibrated BAQP client types.

No GPU and no synthetic cache are required.  The script reconstructs the real
label partition, calibrates client types from that partition, and solves the
existing server objective over a grid of budget fractions and base/AIGC cost
ratios.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .data import load_real, split_train
from .mechanism import instance, solve


def synthetic_count_proxy(z, counts, u):
    """Continuous-count proxy m_k(u)=n_k*gamma*u/(1-gamma*u)."""
    n=np.asarray(counts,dtype=float).sum(1)
    gamma=np.asarray(z['gamma'],dtype=float)
    u=np.asarray(u,dtype=float)
    den=1.0-gamma*u
    return np.divide(n*gamma*u,den,out=np.zeros_like(n),where=den>0)


def summarize(z, counts, result, method, budget_fraction, ratio, base_scale, aigc_scale):
    if result.get('status')!='ok':
        return dict(method=method,budget_fraction=budget_fraction,cost_ratio=ratio,
                    base_cost_total=base_scale,aigc_unit_cost=aigc_scale,status=result.get('status','unknown'))
    active=np.asarray(result['active'],dtype=bool)
    u=np.asarray(result['u'],dtype=float)
    n=np.asarray(counts,dtype=float).sum(1)
    syn=synthetic_count_proxy(z,counts,u)
    interior=active&(u>1e-8)&(u<1-1e-8)
    full=active&(u>=1-1e-8)
    raw=active&(u<=1e-8)
    return dict(
        method=method,
        budget_fraction=float(budget_fraction),
        cost_ratio=float(ratio),
        base_cost_total=float(base_scale),
        aigc_unit_cost=float(aigc_scale),
        status='ok',
        objective_normalized=float(result['objective_normalized']),
        budget=float(z['budget']),
        full_budget=float(z['full_budget']),
        total_payment=float(result['total_payment']),
        budget_utilization=float(result['total_payment']/z['budget']) if z['budget']>0 else 0.0,
        active_clients=int(active.sum()),
        raw_clients=int(raw.sum()),
        interior_clients=int(interior.sum()),
        full_clients=int(full.sum()),
        active_original_samples=int(n[active].sum()),
        synthetic_samples_proxy=float(syn[active].sum()),
        total_augmented_samples_proxy=float((n[active]+syn[active]).sum()),
        mean_u_active=float(u[active].mean()) if active.any() else 0.0,
        active_indices=json.dumps(np.flatnonzero(active).tolist()),
        u=json.dumps([float(x) for x in u]),
        missing_ratio=json.dumps([float(x) for x in np.asarray(z['missing_ratio'])]),
        full_synthetic_n=json.dumps([float(x) for x in np.asarray(z['full_synthetic_n'])]),
        normalized_absolute_gap=float(result.get('normalized_absolute_gap',np.nan)),
        solver_failures=int(result.get('solver_failures',0)),
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],default='cifar10')
    p.add_argument('--data-root',default='data')
    p.add_argument('--clients',type=int,default=6)
    p.add_argument('--alpha',type=float,default=.3,help='Dirichlet partition concentration')
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--mu',type=float,default=.05)
    p.add_argument('--smooth',type=float,default=.55)
    p.add_argument('--lr',type=float,default=None)
    p.add_argument('--budgets',type=float,nargs='+',default=[.02,.04,.06,.08,.10,.15,.20,.30,.40,.50])
    p.add_argument('--cost-ratios',type=float,nargs='+',default=[.25,.5,1.,2.,4.],
                   help='base_cost_total / aigc_unit_cost; aigc_unit_cost is fixed to 1')
    p.add_argument('--output',default='outputs/calibrated_cost_sweep/cifar10_seed2026.csv')
    args=p.parse_args()
    if args.clients<1 or args.alpha<=0 or any(b<=0 for b in args.budgets) or any(r<=0 for r in args.cost_ratios):
        p.error('clients/alpha/budgets/cost-ratios must be positive')

    _,y,_,_=load_real(args.data_root,args.dataset)
    classes=int(y.max())+1
    _,_,counts=split_train(y,classes,args.clients,args.alpha,args.split_seed)

    rows=[]
    for ratio in args.cost_ratios:
        base_scale=float(ratio); aigc_scale=1.0
        for budget in args.budgets:
            z=instance(counts,seed=args.split_seed,budget_fraction=budget,
                       rounds=args.rounds,steps=args.steps,batch=args.batch,
                       mu=args.mu,smooth=args.smooth,lr=args.lr,residual=False,
                       base_cost_total=base_scale,aigc_unit_cost=aigc_scale)
            for method,kwargs in [('continuous',{}),('three_state',{'discrete':True}),('no_aigc',{'raw_only':True})]:
                result=solve(z,**kwargs)
                row=summarize(z,counts,result,method,budget,ratio,base_scale,aigc_scale)
                rows.append(row)
                if row['status']=='ok':
                    print(json.dumps({k:row[k] for k in (
                        'method','budget_fraction','cost_ratio','objective_normalized',
                        'active_clients','interior_clients','full_clients','mean_u_active',
                        'active_original_samples','synthetic_samples_proxy','budget_utilization')},
                        separators=(',',':')),flush=True)

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    json_out=out.with_suffix('.json')
    json_out.write_text(json.dumps(dict(config=vars(args),counts=counts.tolist(),rows=rows),indent=2,allow_nan=False))
    print(f'Wrote {out} and {json_out}')


if __name__=='__main__':
    main()
