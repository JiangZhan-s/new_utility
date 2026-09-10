"""Scalable heuristic for generator-aware BAQP with many clients.

This is not a global-optimality certificate.  It solves each fixed participant
set as a convex continuous problem and uses multi-start add/drop/swap local
search over participant sets.  It is intended to diagnose K=20/50 endpoint
behavior where exact mode enumeration is exponential.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from .data import load_real,split_train
from .mechanism import implementation_payment
from .mechanism_quality import instance,quadratic,package
from .quality import IMFL_CIFAR10_GAP_RATIO


def solve_subset(z,active,tol=1e-9,maxiter=400):
    active=np.asarray(active,dtype=bool)
    if not active.any(): return None
    low=np.zeros(len(active)); high=active.astype(float)
    if implementation_payment(z,active,low).sum()>z['budget']+1e-12: return None
    Ql,Qg,const=quadratic(z,active)
    fun=lambda x: float((1-x)@Ql@(1-x)+x@Qg@x+const)
    jac=lambda x: -2*Ql@(1-x)+2*Qg@x
    cost=lambda x: float(implementation_payment(z,active,x).sum())
    al=z['alpha']*z['lam']; be=z['beta']*z['lam']**2
    res=minimize(fun,low,jac=jac,bounds=list(zip(low,high)),method='SLSQP',
                 constraints=[dict(type='ineq',fun=lambda x:z['budget']-cost(x),
                                   jac=lambda x:-z['d']*active*(al+2*be*x))],
                 options=dict(ftol=tol,maxiter=maxiter))
    x=np.clip(res.x,low,high)
    if cost(x)>z['budget']+1e-7: return None
    return dict(value=fun(x),active=active,u=x,cost=cost(x),success=bool(res.success))


def search(z,restarts=12,seed=2026,max_rounds=30):
    rng=np.random.default_rng(seed); K=len(z['d'])
    base=z['d']*z['s']; full=implementation_payment(z,np.ones(K,dtype=bool),np.ones(K))
    starts=[]
    for order in (np.argsort(base),np.argsort(full)):
        active=np.zeros(K,dtype=bool)
        for k in order:
            trial=active.copy(); trial[k]=True
            if base[trial].sum()<=z['budget']+1e-12: active=trial
        if active.any(): starts.append(active)
    for _ in range(restarts):
        active=rng.random(K)<rng.uniform(.05,.55)
        if not active.any(): active[rng.integers(K)]=True
        while base[active].sum()>z['budget'] and active.any():
            ids=np.flatnonzero(active); active[rng.choice(ids)]=False
        if active.any(): starts.append(active)
    # Strong singleton seeds protect against poor random starts.
    for k in np.argsort(full)[:min(K,20)]:
        active=np.zeros(K,dtype=bool); active[k]=True
        if base[k]<=z['budget']+1e-12: starts.append(active)

    best=None
    for active in starts:
        cur=solve_subset(z,active)
        if cur is None: continue
        for _ in range(max_rounds):
            incumbent=cur; candidates=[]
            on=np.flatnonzero(cur['active']); off=np.flatnonzero(~cur['active'])
            # Prefer cheap additions plus random exploration.
            if len(off):
                cheap=off[np.argsort(base[off])[:min(12,len(off))]]
                rand=rng.choice(off,size=min(6,len(off)),replace=False)
                for j in np.unique(np.r_[cheap,rand]):
                    a=cur['active'].copy(); a[j]=True; candidates.append(a)
            for i in rng.permutation(on)[:min(10,len(on))]:
                a=cur['active'].copy(); a[i]=False
                if a.any(): candidates.append(a)
            if len(on) and len(off):
                for i in rng.permutation(on)[:min(6,len(on))]:
                    for j in rng.permutation(off)[:min(6,len(off))]:
                        a=cur['active'].copy(); a[i]=False; a[j]=True; candidates.append(a)
            for a in candidates:
                trial=solve_subset(z,a)
                if trial is not None and trial['value']<incumbent['value']-1e-9:
                    incumbent=trial
            if incumbent is cur: break
            cur=incumbent
        if best is None or cur['value']<best['value']:
            best=cur
    if best is None: return dict(status='infeasible',budget=z['budget'])
    result=package(z,best['active'],best['u'])
    result.update(status='ok',heuristic=True,
                  heuristic_note='multi-start add/drop/swap participant search; no global optimality certificate',
                  fixed_subset_solver_success=best['success'])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],default='cifar10')
    p.add_argument('--data-root',default='data'); p.add_argument('--clients',type=int,nargs='+',default=[20,50])
    p.add_argument('--alpha',type=float,default=.3); p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--budgets',type=float,nargs='+',default=[.05,.10,.20,.40])
    p.add_argument('--generator-gap-ratio',type=float,default=IMFL_CIFAR10_GAP_RATIO)
    p.add_argument('--restarts',type=int,default=12)
    p.add_argument('--output',default='outputs/generator_quality_heuristic.json')
    args=p.parse_args()
    _,y,_,_=load_real(args.data_root,args.dataset); classes=int(y.max())+1
    rows=[]
    for K in args.clients:
        _,_,counts=split_train(y,classes,K,args.alpha,args.split_seed)
        for b in args.budgets:
            z=instance(counts,seed=args.split_seed,budget_fraction=b,
                       generator_gap_ratio=args.generator_gap_ratio,
                       generator_gap_source='heuristic_sensitivity_or_reference')
            r=search(z,args.restarts,args.split_seed)
            if r['status']=='ok':
                active=np.asarray(r['active'],bool); u=np.asarray(r['u'],float)
                interior=active&(u>1e-3)&(u<1-1e-3); full=active&(u>=1-1e-3)
                row=dict(clients=K,budget_fraction=b,status='ok',
                         objective_normalized=r['objective_normalized'],active_clients=int(active.sum()),
                         interior_clients=int(interior.sum()),full_clients=int(full.sum()),
                         mean_u_active=float(u[active].mean()),u=u.tolist(),active=r['active'],
                         total_payment=r['total_payment'],budget=r['budget'],
                         budget_utilization=float(r['total_payment']/r['budget']))
            else: row=dict(clients=K,budget_fraction=b,status=r['status'])
            rows.append(row); print(json.dumps(row,separators=(',',':')),flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(config=vars(args),rows=rows),indent=2,allow_nan=False))
    print(f'Wrote {out}')


if __name__=='__main__':
    main()
