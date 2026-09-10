"""Generator-aware BAQP experiment with fixed original-data compute.

The experiment separates two questions:
  1. does the learning-aware Stackelberg mechanism choose genuine interior u?
  2. under equal local optimization compute, can that real/synthetic mixture
     outperform full enhancement on held-out accuracy?

Generator quality is measured by default on a short **real-data-only FedAvg
trajectory**, with matched-label real/synthetic gradient probes.  This avoids
using the unrealistically optimistic initial-model-only discrepancy as the
production calibration.
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

from .data import load_real,load_synthetic,split_train,audit
from .experiment import (save,model_for,preprocess,evaluate,build_augmented_indices,
                         draw_additive_batch,local_epoch_steps)
from .mechanism_quality import instance,solve,objective,public_discrete,random_feasible
from .quality import (estimate_gradient_gap,estimate_gradient_gap_trajectory,
                      IMFL_CIFAR10_GAP_RATIO,IMFL_CIFAR10_G_DATA,IMFL_CIFAR10_G_DIFF)


def train_fixed_original_compute(args,contract,parts,counts,pstar,x,y,sx,sy,vx,vy,tx,ty,out):
    device=torch.device(args.device); active=np.flatnonzero(contract['active'])
    if not len(active): raise ValueError('No active clients')
    torch.manual_seed(args.seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    model=model_for(args.model,x.shape[1:],len(pstar)).to(device)
    local=copy.deepcopy(model)
    weights=np.array([len(parts[k]) for k in active],dtype=float); weights/=weights.sum()
    rngs={k:np.random.default_rng(np.random.SeedSequence([args.seed,100,k])) for k in active}
    plans=build_augmented_indices(parts,counts,pstar,contract,sy,args.seed)
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
            for name,value in local.state_dict().items():
                aggregate[name].add_(value,alpha=float(weight))
        model.load_state_dict(aggregate)
        if (t+1)%args.eval_every==0 or t+1==args.rounds:
            val=evaluate(model,vx,vy,device,args.model)
            history.append(dict(round=t+1,validation=val,seconds=time.time()-start,
                                gradient_steps=int(gradient_steps)))
            save(out/'history.json',history)
            print(json.dumps(dict(method=out.name,round=t+1,val_acc=val['acc'],
                                  gradient_steps=gradient_steps)),flush=True)
    test=evaluate(model,tx,ty,device,args.model)
    torch.save(model.state_dict(),out/'final_model.pt')
    expected=int(args.rounds*sum(steps_per_client.values()))
    if expected!=gradient_steps:
        raise AssertionError(f'fixed-compute step count mismatch: expected {expected}, got {gradient_steps}')
    return dict(test=test,validation=history[-1]['validation'],initial_validation=initial,
                history=history,seconds=time.time()-start,
                training_mode='fixed_original_data_compute_sampling_from_augmented_mixture',
                local_epochs_reference=int(args.local_epochs),fixed_steps_per_client_per_round=steps_per_client,
                gradient_steps=int(gradient_steps),expected_gradient_steps=expected,
                actual_real_draws=int(real_draws),actual_synthetic_draws=int(synthetic_draws),
                augmentation={str(k):{kk:vv for kk,vv in v.items() if kk not in ('real_ids','synthetic_ids')}
                              for k,v in plans.items()})


def resolve_gap(args,x,y,sx,sy,parts,classes):
    if args.generator_gap_source=='imfl':
        return IMFL_CIFAR10_GAP_RATIO,dict(
            method='external_imfl_cifar10_reference',g_data=IMFL_CIFAR10_G_DATA,
            g_diff=IMFL_CIFAR10_G_DIFF,generator_gap_ratio=IMFL_CIFAR10_GAP_RATIO,
            note='External CIFAR-10 reference; not measured on this cache.')
    if args.generator_gap_source=='value':
        if args.generator_gap_ratio is None or args.generator_gap_ratio<0:
            raise ValueError('--generator-gap-ratio >=0 is required with --generator-gap-source value')
        return float(args.generator_gap_ratio),dict(method='user_supplied_sensitivity',
                                                     generator_gap_ratio=float(args.generator_gap_ratio))
    torch.manual_seed(args.quality_seed)
    ref=model_for(args.model,x.shape[1:],classes)
    if args.generator_gap_source=='initial':
        train_ids=np.concatenate(parts)
        estimate=estimate_gradient_gap(ref,x[train_ids],y[train_ids],sx,sy,args.model,
                                       device=args.quality_device,classes=classes,
                                       max_samples_per_class=args.quality_samples_per_class,
                                       seed=args.quality_seed,batch=args.quality_batch)
    else:
        estimate=estimate_gradient_gap_trajectory(
            ref,x,y,sx,sy,parts,args.model,device=args.quality_device,classes=classes,
            rounds=args.quality_rounds,local_steps=args.quality_local_steps,
            train_batch=args.batch,probe_batch=args.quality_batch,
            max_probe_samples_per_client=args.quality_samples_per_client,
            lr=args.lr,weight_decay=args.mu,seed=args.quality_seed,
            probe_every=args.quality_probe_every)
    return float(estimate['generator_gap_ratio']),estimate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],required=True)
    p.add_argument('--data-root',default='data'); p.add_argument('--aigc-root',default='aigc_imgs')
    p.add_argument('--output',default='outputs/real_baqp_generator_aware')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--clients',type=int,default=6); p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--budget-fraction',type=float,default=.4)
    p.add_argument('--rounds',type=int,default=100); p.add_argument('--local-epochs',type=int,default=2)
    p.add_argument('--steps',type=int,default=5,help='Theory coefficient h; training compute is fixed from original local epochs')
    p.add_argument('--batch',type=int,default=32); p.add_argument('--mu',type=float,default=None)
    p.add_argument('--lr',type=float,default=None); p.add_argument('--model',choices=['cnn','softmax'],default='cnn')
    p.add_argument('--device',default='cuda'); p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--base-cost-total',type=float,default=1.0); p.add_argument('--aigc-unit-cost',type=float,default=1.0)
    p.add_argument('--generator-gap-source',choices=['trajectory','initial','imfl','value'],default='trajectory')
    p.add_argument('--generator-gap-ratio',type=float,default=None)
    p.add_argument('--quality-device',default='cpu'); p.add_argument('--quality-seed',type=int,default=2026)
    p.add_argument('--quality-samples-per-class',type=int,default=512,
                   help='Only for the legacy initial-model diagnostic')
    p.add_argument('--quality-samples-per-client',type=int,default=512,
                   help='Real samples per client in each trajectory quality probe')
    p.add_argument('--quality-batch',type=int,default=256)
    p.add_argument('--quality-rounds',type=int,default=20)
    p.add_argument('--quality-local-steps',type=int,default=5)
    p.add_argument('--quality-probe-every',type=int,default=1)
    p.add_argument('--methods',nargs='+',default=['fedavg_all','no_aigc_budget','three_state','continuous',
                                                  'selected_no_aigc','selected_full'])
    p.add_argument('--prepare-only',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    args.mu=(.05 if args.model=='softmax' else .0005) if args.mu is None else args.mu
    args.lr=(.1/(.5+args.mu) if args.model=='softmax' else .05) if args.lr is None else args.lr
    if min(args.rounds,args.steps,args.local_epochs,args.batch,args.eval_every,args.clients,
           args.quality_rounds,args.quality_local_steps,args.quality_batch,args.quality_probe_every)<1 or args.alpha<=0:
        p.error('positive counts/alpha/quality controls required')
    if args.device.startswith('cuda') and not args.prepare_only and 'SLURM_JOB_ID' not in os.environ:
        p.error('GPU training must run inside a Slurm allocation')
    torch.set_num_threads(2)
    x,y,tx,ty=load_real(args.data_root,args.dataset); classes=int(y.max())+1
    sx,sy,cache=load_synthetic(args.aigc_root,args.dataset,classes)
    if sx.shape[1:]!=x.shape[1:]: raise ValueError('Real/synthetic shape mismatch')
    parts,valid,counts=split_train(y,classes,args.clients,args.alpha,args.split_seed)
    gap,quality=resolve_gap(args,x,y,sx,sy,parts,classes)
    z=instance(counts,args.split_seed,args.budget_fraction,args.rounds,args.steps,args.batch,
               args.mu,.5+args.mu,args.lr,False,args.base_cost_total,args.aigc_unit_cost,
               generator_gap_ratio=gap,generator_gap_source=args.generator_gap_source)
    pstar=counts.sum(0)/counts.sum()
    out=Path(args.output)/args.dataset/args.model/f'alpha_{args.alpha}_budget_{args.budget_fraction}'/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    setting={k:v for k,v in vars(args).items() if k not in ('resume','prepare_only','methods','output')}
    setting['resolved_generator_gap_ratio']=gap
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    setting['source_sha256']=hashes
    fingerprint=hashlib.sha256(json.dumps(setting,sort_keys=True).encode()).hexdigest()
    if (out/'metadata.json').exists():
        old=json.loads((out/'metadata.json').read_text())
        if old['fingerprint']!=fingerprint: raise ValueError('Output configuration/code changed; choose a new --output')
        if not args.resume: raise ValueError('Output exists; use --resume or a new --output')
    diagnostic=audit(x,y,sx,sy,parts,classes)
    np.savez_compressed(out/'partition.npz',validation=valid,counts=counts,**{f'client_{k}':v for k,v in enumerate(parts)})
    save(out/'metadata.json',dict(config=vars(args),fingerprint=fingerprint,source_sha256=hashes,
        torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),slurm_job=os.getenv('SLURM_JOB_ID'),
        cache=str(cache.resolve()),cache_bytes=cache.stat().st_size,cache_mtime_ns=cache.stat().st_mtime_ns,
        partition_sha256=hashlib.sha256((out/'partition.npz').read_bytes()).hexdigest(),
        instance={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in z.items()},
        generator_quality=quality,audit=diagnostic,
        augmentation_semantics='addition_only: retain every original sample and add class-targeted AIGC samples',
        training_semantics='fixed original-data local compute; sample from the real+synthetic mixture with replacement',
        aggregation_semantics='FedAvg weights use original client sample counts',
        interpretation=('Generator-aware empirical experiment. Trajectory quality is a train-only empirical envelope; '
                        'actual CNN accuracy remains an empirical outcome.'),accuracy_certificate=None))
    builders={'continuous':lambda:solve(z),'three_state':lambda:solve(z,discrete=True),
              'no_aigc_budget':lambda:solve(z,raw_only=True),'public_price':lambda:public_discrete(z),
              'random_budget':lambda:random_feasible(z,args.split_seed)}
    continuous_cache=None
    for method in args.methods:
        dest=out/method; dest.mkdir(exist_ok=True)
        if args.resume and (dest/'result.json').exists(): continue
        if method in ('selected_no_aigc','selected_full'):
            if continuous_cache is None:
                cp=out/'continuous'/'contract.json'
                continuous_cache=json.loads(cp.read_text()) if cp.exists() else solve(z)
            if continuous_cache['status']!='ok': contract=dict(status='infeasible')
            else:
                level=0.0 if method=='selected_no_aigc' else 1.0
                uu=np.array([level if a else 0.0 for a in continuous_cache['active']])
                aa=np.asarray(continuous_cache['active'],bool)
                contract=dict(status='ok',active=continuous_cache['active'],u=uu.tolist(),
                              objective_normalized=objective(z,aa,uu),total_payment=None,reference_only=True,
                              note='Same participants as continuous; fixed-compute learning ablation, not necessarily budget-feasible.')
        elif method=='fedavg_all':
            active=np.ones(args.clients,dtype=bool); u=np.zeros(args.clients)
            contract=dict(status='ok',active=active.tolist(),u=u.tolist(),objective_normalized=objective(z,active,u),
                          total_payment=None,reference_only=True,note='All-client real-data reference')
        elif method=='full_balance_all':
            active=np.ones(args.clients,dtype=bool); u=np.ones(args.clients)
            contract=dict(status='ok',active=active.tolist(),u=u.tolist(),objective_normalized=objective(z,active,u),
                          total_payment=None,reference_only=True,note='All-client full AIGC reference')
        elif method in builders:
            contract=builders[method]()
            if method=='continuous': continuous_cache=contract
        else: raise ValueError('Unknown method '+method)
        save(dest/'contract.json',contract)
        print(json.dumps(dict(method=method,contract=contract)),flush=True)
        if args.prepare_only or contract['status']!='ok': continue
        metrics=train_fixed_original_compute(args,contract,parts,counts,pstar,x,y,sx,sy,x[valid],y[valid],tx,ty,dest)
        save(dest/'result.json',dict(dataset=args.dataset,model=args.model,seed=args.seed,method=method,
                                  alpha=args.alpha,budget_fraction=args.budget_fraction,
                                  generator_gap_ratio=gap,contract=contract,**metrics))


if __name__=='__main__': main()
