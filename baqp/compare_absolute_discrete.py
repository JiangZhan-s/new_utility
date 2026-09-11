"""Fair continuous-vs-discrete allocation comparison for the 30-client path.

This is a mechanism-only diagnostic; it does not train a neural network.
It solves the continuous all-active absolute-q allocation first, then gives the
all-active discrete repair baseline exactly the continuous solution's realized
payment as its monetary cap.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from .data import load_real
from .data_absolute import split_train_fixed_size
from .mechanism_absolute import instance, solve_all_clients_heuristic, package
from .discrete_absolute import solve_discrete_repair_matched


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],default='cifar10')
    p.add_argument('--data-root',default='data')
    p.add_argument('--clients',type=int,default=30)
    p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--budget-fraction',type=float,default=.4)
    p.add_argument('--generator-gap-ratio',type=float,default=.06180083094278078)
    p.add_argument('--base-cost-total',type=float,default=1.0)
    p.add_argument('--aigc-unit-cost',type=float,default=1.0)
    p.add_argument('--aigc-convex-cost',type=float,default=.1)
    p.add_argument('--continuous-restarts',type=int,default=8)
    p.add_argument('--discrete-restarts',type=int,default=128)
    p.add_argument('--output',default='outputs/absolute_q_30_fair_compare.json')
    args=p.parse_args()

    _,y,_,_=load_real(args.data_root,args.dataset)
    classes=int(y.max())+1
    _,_,counts=split_train_fixed_size(y,classes,args.clients,args.alpha,args.split_seed)
    z=instance(counts,budget_fraction=args.budget_fraction,
               base_cost_total=args.base_cost_total,
               aigc_unit_cost=args.aigc_unit_cost,
               aigc_convex_cost=args.aigc_convex_cost,
               generator_gap_ratio=args.generator_gap_ratio,
               generator_gap_source='fixed_measured_reference')

    con=solve_all_clients_heuristic(z,restarts=args.continuous_restarts,
                                    seed=args.split_seed,maxiter=300)
    if con.get('status')!='ok': raise SystemExit('continuous solve failed')
    cap=float(con['total_payment'])
    disc=solve_discrete_repair_matched(z,cap,restarts=args.discrete_restarts,
                                       seed=args.split_seed,max_rounds=80)
    if disc.get('status')!='ok': raise SystemExit('discrete solve failed')

    active=np.ones(args.clients,dtype=bool)
    raw=package(z,active,np.zeros(args.clients))
    rows=[]
    for name,r in [('continuous',con),('discrete_repair_matched',disc),('no_aigc',raw)]:
        q=np.asarray(r['q_synthetic'],float)
        f=np.asarray(r['synthetic_fraction'],float)
        rows.append(dict(method=name,objective=float(r['objective_normalized']),
                         payment=float(r['total_payment']),q_total=float(q.sum()),
                         q_mean=float(q.mean()),q_max=float(q.max()),
                         syn_fraction_mean=float(f.mean()),syn_fraction_max=float(f.max()),
                         components=r['objective_components']))

    print('\n=== FAIR SPEND-MATCHED ABSOLUTE-q COMPARISON ===')
    print(f'original budget cap = {z["budget"]:.9f}')
    print(f'continuous realized spend = matched discrete cap = {cap:.9f}')
    print('method\tJ\tpayment\tq_total\tmean_q\tmax_q\tmean_syn_frac\tmax_syn_frac')
    for r in rows:
        print(f"{r['method']}\t{r['objective']:.9f}\t{r['payment']:.9f}\t"
              f"{r['q_total']:.2f}\t{r['q_mean']:.2f}\t{r['q_max']:.2f}\t"
              f"{r['syn_fraction_mean']:.4f}\t{r['syn_fraction_max']:.4f}")
    print(f"discrete repaired clients = {disc['repaired_clients']} / {args.clients}")
    print(f"discrete unused matched cap = {disc['unused_matched_budget']:.9f}")
    print(f"J_discrete - J_continuous = {disc['objective_normalized']-con['objective_normalized']:.9f}")
    print('\nclient\tq_cont\tq_discrete\tdiscrete_state\trepair_q')
    cq=np.asarray(con['q_synthetic'],float); dq=np.asarray(disc['q_synthetic'],float)
    rq=np.asarray(z['repair_q'],float)
    for k in range(args.clients):
        print(f'{k}\t{cq[k]:.2f}\t{dq[k]:.2f}\t{disc["discrete_state"][k]}\t{rq[k]:.2f}')

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(config=vars(args),instance_budget=float(z['budget']),
                                   continuous=con,discrete_repair_matched=disc,
                                   no_aigc=raw,summary=rows),indent=2,allow_nan=False))
    print(f'Wrote {out}')


if __name__=='__main__': main()
