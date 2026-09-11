"""CPU diagnostic for the joint participation + absolute-q model."""
import argparse
import json
import numpy as np

from .data import load_real
from .data_absolute import split_train_fixed_size
from .mechanism_joint_absolute import (instance,solve_joint_heuristic,
                                       solve_three_state_heuristic)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],default='cifar10')
    p.add_argument('--data-root',default='data')
    p.add_argument('--clients',type=int,default=30)
    p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--budget-fraction',type=float,default=.4)
    p.add_argument('--generator-gap-ratio',type=float,default=.06180083094278078)
    p.add_argument('--joint-restarts',type=int,default=6)
    p.add_argument('--three-restarts',type=int,default=30)
    args=p.parse_args()

    _,y,_,_=load_real(args.data_root,args.dataset)
    classes=int(y.max())+1
    _,_,counts=split_train_fixed_size(y,classes,args.clients,args.alpha,args.split_seed)
    z=instance(counts,budget_fraction=args.budget_fraction,
               generator_gap_ratio=args.generator_gap_ratio)
    con=solve_joint_heuristic(z,restarts=args.joint_restarts,seed=args.split_seed)
    if con.get('status')!='ok': raise SystemExit(json.dumps(con))
    matched=float(con['total_payment'])
    tri=solve_three_state_heuristic(z,restarts=args.three_restarts,seed=args.split_seed)
    tri_match=solve_three_state_heuristic(z,budget=matched,restarts=args.three_restarts,seed=args.split_seed+1)

    def summary(name,r):
        if r.get('status')!='ok': return dict(method=name,status=r.get('status'))
        a=np.asarray(r['active'],bool); q=np.asarray(r['q_synthetic'],float)
        return dict(method=name,status='ok',active_clients=int(a.sum()),
                    total_q=float(q.sum()),mean_q_active=float(q[a].mean()) if a.any() else 0.,
                    max_q=float(q.max()),payment=float(r['total_payment']),budget=float(r['budget']),
                    objective=float(r['objective_normalized']),components=r['objective_components'])

    print(json.dumps(dict(config=vars(args),all_base_cost=z['all_base_cost'],
                          reference_repair_budget=z['reference_repair_budget'],budget=z['budget']),indent=2))
    for name,r in [('continuous',con),('three_state_same_cap',tri),('three_state_spend_matched',tri_match)]:
        print(json.dumps(summary(name,r),indent=2))

    print('\nclient\tn\tcontinuous_active\tq_cont\ttri_state\tq_tri\ttri_match_state\tq_tri_match\trepair_q')
    s=np.asarray(tri.get('state',np.zeros(args.clients)),int) if tri.get('status')=='ok' else np.zeros(args.clients,int)
    sm=np.asarray(tri_match.get('state',np.zeros(args.clients)),int) if tri_match.get('status')=='ok' else np.zeros(args.clients,int)
    qc=np.asarray(con['q_synthetic'],float)
    qt=np.asarray(tri.get('q_synthetic',np.zeros(args.clients)),float)
    qtm=np.asarray(tri_match.get('q_synthetic',np.zeros(args.clients)),float)
    ac=np.asarray(con['active'],bool)
    label={0:'inactive',1:'raw',2:'repair'}
    for k in range(args.clients):
        print(f'{k}\t{int(z["n"][k])}\t{int(ac[k])}\t{qc[k]:.3f}\t{label[int(s[k])]}\t{qt[k]:.3f}\t'
              f'{label[int(sm[k])]}\t{qtm[k]:.3f}\t{z["repair_q"][k]:.3f}')

    if tri_match.get('status')=='ok':
        print('\n=== SET-INCLUSION / SPEND-MATCHED CHECK ===')
        print(f'continuous spend = {matched:.9f}')
        print(f'three-state matched spend = {tri_match["total_payment"]:.9f}')
        print(f'three-state unused cap = {matched-tri_match["total_payment"]:.9f}')
        print(f'J_continuous = {con["objective_normalized"]:.9f}')
        print(f'J_three_state_matched = {tri_match["objective_normalized"]:.9f}')
        print(f'gap = {tri_match["objective_normalized"]-con["objective_normalized"]:.9f}')


if __name__=='__main__':
    main()
