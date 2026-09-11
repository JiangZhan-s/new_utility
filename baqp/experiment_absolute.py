"""Thirty-client absolute-count AIGC procurement experiment.

Primary design choices:
- equal original sample mass per client (CIFAR-10: 45,000/30 = 1,500);
- all clients are fixed active, eliminating participant-set renormalization jumps;
- server decision is absolute synthetic sample count q_k >= 0;
- no synthetic/real ratio cap and no q<=repair constraint;
- only the monetary budget restricts purchased quantities;
- local CNN compute is fixed from original real-data size, so methods use the
  same number of optimizer updates.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

from .data import load_real,load_synthetic,audit
from .data_absolute import split_train_fixed_size,absolute_target_counts
from .experiment import save,model_for,preprocess,evaluate,draw_additive_batch,local_epoch_steps
from .mechanism_absolute import (instance,solve_all_clients_heuristic,uniform_budget,
                                 reference_all_repair,package)
from .quality import estimate_gradient_gap_trajectory

DEFAULT_GAP=.06180083094278078


def build_absolute_augmented_indices(parts,counts,pstar,contract,sy,seed):
    classes=len(pstar); pools=[torch.where(sy==c)[0].numpy() for c in range(classes)]
    plans={}; qvec=np.asarray(contract['q_synthetic'],dtype=float)
    for k in np.flatnonzero(contract['active']):
        # Continuous procurement is implemented conservatively by flooring to
        # an integer number of images; this can only reduce actual generation
        # cost relative to the continuous contract.
        q_int=max(0,int(np.floor(qvec[k]+1e-9)))
        target,final_counts,add_counts=absolute_target_counts(counts[k],pstar,q_int)
        rng=np.random.default_rng(np.random.SeedSequence([seed,1700,k]))
        synthetic=[]
        for c,m in enumerate(add_counts):
            if m<=0: continue
            if len(pools[c])==0: raise ValueError(f'No synthetic samples for class {c}')
            synthetic.append(rng.choice(pools[c],size=int(m),replace=(m>len(pools[c]))))
        synthetic_ids=np.concatenate(synthetic).astype(np.int64) if synthetic else np.empty(0,dtype=np.int64)
        plans[k]=dict(real_ids=np.asarray(parts[k],dtype=np.int64),synthetic_ids=synthetic_ids,
                      target_distribution=target.tolist(),final_counts=final_counts.tolist(),
                      added_counts=add_counts.tolist(),original_n=int(len(parts[k])),
                      augmented_n=int(final_counts.sum()),synthetic_n=int(add_counts.sum()),
                      q_requested_continuous=float(qvec[k]),q_train_integer=q_int,
                      q_floor_rounding=float(qvec[k]-q_int))
    return plans


def train_fixed_original_compute(args,contract,parts,counts,pstar,x,y,sx,sy,vx,vy,tx,ty,out):
    device=torch.device(args.device); active=np.flatnonzero(contract['active'])
    if not len(active): raise ValueError('No active clients')
    torch.manual_seed(args.seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    model=model_for(args.model,x.shape[1:],len(pstar)).to(device); local=copy.deepcopy(model)
    weights=np.array([len(parts[k]) for k in active],dtype=float); weights/=weights.sum()
    rngs={k:np.random.default_rng(np.random.SeedSequence([args.seed,100,k])) for k in active}
    plans=build_absolute_augmented_indices(parts,counts,pstar,contract,sy,args.seed)
    save(out/'augmentation_plan.json',{str(k):{kk:vv for kk,vv in v.items() if kk not in ('real_ids','synthetic_ids')}
                                       for k,v in plans.items()})
    steps_per_client={int(k):local_epoch_steps(len(parts[k]),args.batch,args.local_epochs) for k in active}
    history=[]; start=time.time(); initial=evaluate(model,vx,vy,device,args.model)
    synthetic_draws=0; real_draws=0; gradient_steps=0
    for t in range(args.rounds):
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        aggregate={k:torch.zeros_like(v) for k,v in state.items()}
        for k,weight in zip(active,weights):
            local.load_state_dict(state); local.train()
            optimizer=torch.optim.SGD(local.parameters(),lr=args.lr,weight_decay=args.mu)
            rng=rngs[k]
            for _ in range(steps_per_client[int(k)]):
                bx,by,drawn=draw_additive_batch(plans[k],x,y,sx,sy,args.batch,rng)
                synthetic_draws+=drawn; real_draws+=len(by)-drawn; gradient_steps+=1
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(local(preprocess(bx.to(device),args.model)),by.to(device))
                loss.backward(); optimizer.step()
            for name,value in local.state_dict().items(): aggregate[name].add_(value,alpha=float(weight))
        model.load_state_dict(aggregate)
        if (t+1)%args.eval_every==0 or t+1==args.rounds:
            val=evaluate(model,vx,vy,device,args.model)
            history.append(dict(round=t+1,validation=val,seconds=time.time()-start,
                                gradient_steps=int(gradient_steps)))
            save(out/'history.json',history)
            print(json.dumps(dict(method=out.name,round=t+1,val_acc=val['acc'],gradient_steps=gradient_steps)),flush=True)
    test=evaluate(model,tx,ty,device,args.model); torch.save(model.state_dict(),out/'final_model.pt')
    expected=int(args.rounds*sum(steps_per_client.values()))
    if expected!=gradient_steps: raise AssertionError(f'fixed-compute mismatch {expected} != {gradient_steps}')
    return dict(test=test,validation=history[-1]['validation'],initial_validation=initial,history=history,
                seconds=time.time()-start,training_mode='fixed_original_data_compute_absolute_q_mixture',
                local_epochs_reference=int(args.local_epochs),fixed_steps_per_client_per_round=steps_per_client,
                gradient_steps=int(gradient_steps),expected_gradient_steps=expected,
                actual_real_draws=int(real_draws),actual_synthetic_draws=int(synthetic_draws),
                augmentation={str(k):{kk:vv for kk,vv in v.items() if kk not in ('real_ids','synthetic_ids')}
                              for k,v in plans.items()})


def resolve_gap(args,x,y,sx,sy,parts,classes):
    if args.generator_gap_source=='value':
        return float(args.generator_gap_ratio),dict(method='fixed_measured_reference',
            generator_gap_ratio=float(args.generator_gap_ratio),
            note='Frozen from the measured CIFAR-10 EDM trajectory calibration for controlled mechanism comparison.')
    torch.manual_seed(args.quality_seed)
    ref=model_for(args.model,x.shape[1:],classes)
    est=estimate_gradient_gap_trajectory(ref,x,y,sx,sy,parts,args.model,device=args.quality_device,
        classes=classes,rounds=args.quality_rounds,local_steps=args.quality_local_steps,
        train_batch=args.batch,probe_batch=args.quality_batch,
        max_probe_samples_per_client=args.quality_samples_per_client,lr=args.lr,
        weight_decay=args.mu,seed=args.quality_seed,probe_every=args.quality_probe_every)
    return float(est['generator_gap_ratio']),est


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],default='cifar10')
    p.add_argument('--data-root',default='data'); p.add_argument('--aigc-root',default='aigc_imgs')
    p.add_argument('--output',default='outputs/absolute_q_30_v1')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--clients',type=int,default=30); p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--budget-fraction',type=float,default=.4)
    p.add_argument('--rounds',type=int,default=100); p.add_argument('--local-epochs',type=int,default=2)
    p.add_argument('--steps',type=int,default=5); p.add_argument('--batch',type=int,default=32)
    p.add_argument('--mu',type=float,default=.0005); p.add_argument('--lr',type=float,default=.05)
    p.add_argument('--model',choices=['cnn','softmax'],default='cnn'); p.add_argument('--device',default='cuda')
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--base-cost-total',type=float,default=1.0); p.add_argument('--aigc-unit-cost',type=float,default=1.0)
    p.add_argument('--aigc-convex-cost',type=float,default=.1,
                   help='Weak increasing-marginal-effort cost for unique Stackelberg response; not a quantity cap.')
    p.add_argument('--generator-gap-source',choices=['value','trajectory'],default='value')
    p.add_argument('--generator-gap-ratio',type=float,default=DEFAULT_GAP)
    p.add_argument('--quality-device',default='cpu'); p.add_argument('--quality-seed',type=int,default=2026)
    p.add_argument('--quality-samples-per-client',type=int,default=256); p.add_argument('--quality-batch',type=int,default=256)
    p.add_argument('--quality-rounds',type=int,default=20); p.add_argument('--quality-local-steps',type=int,default=5)
    p.add_argument('--quality-probe-every',type=int,default=1)
    p.add_argument('--heuristic-restarts',type=int,default=4); p.add_argument('--heuristic-maxiter',type=int,default=180)
    p.add_argument('--methods',nargs='+',default=['personalized','uniform_budget','no_aigc','repair_reference'])
    p.add_argument('--prepare-only',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.clients<2 or args.alpha<=0 or args.budget_fraction<=0: p.error('invalid clients/alpha/budget')
    if min(args.rounds,args.local_epochs,args.steps,args.batch,args.eval_every)<1: p.error('positive training controls required')
    if args.aigc_convex_cost<=0: p.error('--aigc-convex-cost must be >0 for a unique continuous follower response')
    if args.generator_gap_ratio<0: p.error('--generator-gap-ratio must be >=0')
    if args.device.startswith('cuda') and not args.prepare_only and 'SLURM_JOB_ID' not in os.environ:
        p.error('GPU training must run inside a Slurm allocation')
    torch.set_num_threads(2)
    x,y,tx,ty=load_real(args.data_root,args.dataset); classes=int(y.max())+1
    sx,sy,cache=load_synthetic(args.aigc_root,args.dataset,classes)
    parts,valid,counts=split_train_fixed_size(y,classes,args.clients,args.alpha,args.split_seed)
    sizes=np.array([len(v) for v in parts])
    if sizes.max()-sizes.min()>1: raise AssertionError('fixed-size partition invariant violated')
    gap,quality=resolve_gap(args,x,y,sx,sy,parts,classes)
    z=instance(counts,args.budget_fraction,args.rounds,args.steps,args.batch,args.mu,.5+args.mu,args.lr,
               args.base_cost_total,args.aigc_unit_cost,args.aigc_convex_cost,gap,args.generator_gap_source)
    pstar=z['pstar']
    out=Path(args.output)/args.dataset/args.model/f'clients_{args.clients}_alpha_{args.alpha}_budget_{args.budget_fraction}'/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    setting={k:v for k,v in vars(args).items() if k not in ('resume','prepare_only','methods','output')}
    setting['resolved_generator_gap_ratio']=gap
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    setting['source_sha256']=hashes; fingerprint=hashlib.sha256(json.dumps(setting,sort_keys=True).encode()).hexdigest()
    if (out/'metadata.json').exists():
        old=json.loads((out/'metadata.json').read_text())
        if old['fingerprint']!=fingerprint: raise ValueError('Output configuration/code changed; choose a new --output')
        if not args.resume: raise ValueError('Output exists; use --resume or a new --output')
    np.savez_compressed(out/'partition.npz',validation=valid,counts=counts,**{f'client_{k}':v for k,v in enumerate(parts)})
    diagnostic=audit(x,y,sx,sy,parts,classes)
    save(out/'metadata.json',dict(config=vars(args),fingerprint=fingerprint,source_sha256=hashes,
        torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),slurm_job=os.getenv('SLURM_JOB_ID'),
        cache=str(cache.resolve()),partition_sha256=hashlib.sha256((out/'partition.npz').read_bytes()).hexdigest(),
        fixed_client_sizes=sizes.tolist(),fixed_client_size_min=int(sizes.min()),fixed_client_size_max=int(sizes.max()),
        generator_quality=quality,audit=diagnostic,
        instance={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in z.items()},
        decision_semantics='q_k is absolute purchased AIGC sample count; q_k>=0; no synthetic/real ratio cap',
        budget_semantics='only monetary budget limits q; repair_q is a normalization/reference quantity, never a cap',
        participation_semantics='all clients fixed active to remove participant-set renormalization jumps',
        training_semantics='fixed original-data local compute; batches sampled from real plus purchased synthetic multiset',
        aggregation_semantics='FedAvg weights use fixed original client sample counts',accuracy_certificate=None))

    all_active=np.ones(args.clients,dtype=bool); zero=np.zeros(args.clients)
    builders={
        'personalized':lambda:solve_all_clients_heuristic(z,args.heuristic_restarts,args.split_seed,
                                                          maxiter=args.heuristic_maxiter),
        'uniform_budget':lambda:uniform_budget(z),
        'no_aigc':lambda:package(z,all_active,zero),
        'repair_reference':lambda:reference_all_repair(z),
    }
    for method in args.methods:
        if method not in builders: raise ValueError('Unknown method '+method)
        dest=out/method; dest.mkdir(exist_ok=True)
        if args.resume and (dest/'result.json').exists(): continue
        cp=dest/'contract.json'
        if args.resume and cp.exists():
            contract=json.loads(cp.read_text())
        else:
            contract=builders[method](); save(cp,contract)
        print(json.dumps(dict(method=method,contract=contract)),flush=True)
        if args.prepare_only or contract.get('status')!='ok': continue
        metrics=train_fixed_original_compute(args,contract,parts,counts,pstar,x,y,sx,sy,x[valid],y[valid],tx,ty,dest)
        save(dest/'result.json',dict(dataset=args.dataset,model=args.model,seed=args.seed,method=method,
             clients=args.clients,alpha=args.alpha,budget_fraction=args.budget_fraction,
             generator_gap_ratio=gap,contract=contract,**metrics))


if __name__=='__main__': main()
