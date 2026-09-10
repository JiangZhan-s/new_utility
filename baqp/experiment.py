import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .data import load_real,load_synthetic,split_train,audit,additive_target_counts
from .mechanism import instance,solve,public_discrete,random_feasible,objective


def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)); tmp.replace(path)


def model_for(kind,shape,classes):
    if kind=='softmax': return nn.Sequential(nn.Flatten(),nn.Linear(int(np.prod(shape)),classes,bias=False))
    return nn.Sequential(nn.Conv2d(shape[0],32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AvgPool2d(shape[-1]//8),
                         nn.Flatten(),nn.Linear(512,classes))


def preprocess(x,kind):
    x=x.float()/255
    if kind=='softmax':
        x=x/torch.linalg.vector_norm(x.flatten(1),dim=1).clamp_min(1)[:,None,None,None]
    else: x=(x-.5)/.5
    return x


@torch.no_grad()
def evaluate(model,x,y,device,kind):
    model.eval(); correct=0; loss=0
    for i in range(0,len(y),512):
        target=y[i:i+512].to(device); logits=model(preprocess(x[i:i+512].to(device),kind))
        correct+=(logits.argmax(1)==target).sum().item()
        loss+=nn.functional.cross_entropy(logits,target,reduction='sum').item()
    return dict(acc=correct/len(y),ce=loss/len(y),n=len(y))


def build_augmented_indices(parts,counts,pstar,contract,sy,seed):
    """Return per-client real ids plus synthetic ids; never remove real samples."""
    classes=len(pstar)
    pools=[torch.where(sy==c)[0].numpy() for c in range(classes)]
    plans={}
    for k in np.flatnonzero(contract['active']):
        u=float(contract['u'][k])
        target,final_counts,add_counts=additive_target_counts(counts[k],pstar,u)
        rng=np.random.default_rng(np.random.SeedSequence([seed,700,k]))
        synthetic=[]
        for c,m in enumerate(add_counts):
            if m<=0: continue
            if len(pools[c])==0: raise ValueError(f'No synthetic samples for class {c}')
            synthetic.append(rng.choice(pools[c],size=int(m),replace=(m>len(pools[c]))))
        synthetic_ids=np.concatenate(synthetic).astype(np.int64) if synthetic else np.empty(0,dtype=np.int64)
        plans[k]=dict(real_ids=np.asarray(parts[k],dtype=np.int64),synthetic_ids=synthetic_ids,
                      target_distribution=target.tolist(),final_counts=final_counts.tolist(),
                      added_counts=add_counts.tolist(),original_n=int(len(parts[k])),
                      augmented_n=int(final_counts.sum()),synthetic_n=int(add_counts.sum()))
    return plans


def draw_additive_batch(plan,x,y,sx,sy,batch,rng):
    """Fixed-step sampler for the theorem-oriented softmax path."""
    nr=plan['original_n']; ns=plan['synthetic_n']; total=nr+ns
    choose=rng.integers(total,size=batch)
    real_mask=choose<nr
    bx=torch.empty((batch,*x.shape[1:]),dtype=x.dtype)
    by=torch.empty(batch,dtype=y.dtype)
    if real_mask.any():
        pos=torch.from_numpy(np.flatnonzero(real_mask))
        rid=torch.from_numpy(np.asarray(plan['real_ids'])[choose[real_mask]])
        bx[pos]=x[rid]; by[pos]=y[rid]
    if (~real_mask).any():
        pos=torch.from_numpy(np.flatnonzero(~real_mask))
        sid=torch.from_numpy(np.asarray(plan['synthetic_ids'])[choose[~real_mask]-nr])
        bx[pos]=sx[sid]; by[pos]=sy[sid]
    return bx,by,int((~real_mask).sum())


def local_epoch_steps(augmented_n,batch,local_epochs):
    """Number of optimizer updates for complete local-epoch training."""
    if min(int(augmented_n),int(batch),int(local_epochs))<1:
        raise ValueError('augmented_n, batch and local_epochs must be positive')
    return int(local_epochs)*int(math.ceil(int(augmented_n)/int(batch)))


def iter_additive_epoch_batches(plan,x,y,sx,sy,batch,rng):
    """Yield one shuffled pass over D_real union D_syn without sample replacement.

    A synthetic cache entry may itself appear multiple times in synthetic_ids when
    the required class addition exceeds the offline cache size. Those repeated
    entries are part of the augmented multiset and are therefore visited once per
    epoch, exactly like duplicated training examples in an ordinary dataset.
    """
    nr=int(plan['original_n']); ns=int(plan['synthetic_n']); total=nr+ns
    order=rng.permutation(total)
    real_ids=np.asarray(plan['real_ids'],dtype=np.int64)
    synthetic_ids=np.asarray(plan['synthetic_ids'],dtype=np.int64)
    for start in range(0,total,batch):
        choose=order[start:start+batch]
        real_mask=choose<nr
        size=len(choose)
        bx=torch.empty((size,*x.shape[1:]),dtype=x.dtype)
        by=torch.empty(size,dtype=y.dtype)
        if real_mask.any():
            pos=torch.from_numpy(np.flatnonzero(real_mask))
            rid=torch.from_numpy(real_ids[choose[real_mask]])
            bx[pos]=x[rid]; by[pos]=y[rid]
        if (~real_mask).any():
            pos=torch.from_numpy(np.flatnonzero(~real_mask))
            sid=torch.from_numpy(synthetic_ids[choose[~real_mask]-nr])
            bx[pos]=sx[sid]; by[pos]=sy[sid]
        yield bx,by,int((~real_mask).sum())


def train(args,contract,parts,counts,pstar,x,y,sx,sy,vx,vy,tx,ty,out):
    device=torch.device(args.device); active=np.flatnonzero(contract['active'])
    if not len(active): raise ValueError('No active clients')
    torch.manual_seed(args.seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    model=model_for(args.model,x.shape[1:],len(pstar)).to(device)
    local=copy.deepcopy(model)
    # Keep the paper's fixed original-data FedAvg weights. AIGC changes local
    # training data and computation, but not a client's voting power.
    weights=np.array([len(parts[k]) for k in active],dtype=float); weights/=weights.sum()
    rngs={k:np.random.default_rng(np.random.SeedSequence([args.seed,100,k])) for k in active}
    plans=build_augmented_indices(parts,counts,pstar,contract,sy,args.seed)
    save(out/'augmentation_plan.json',{str(k):{kk:vv for kk,vv in v.items() if kk not in ('real_ids','synthetic_ids')}
                                       for k,v in plans.items()})
    history=[]; start=time.time(); initial=evaluate(model,vx,vy,device,args.model)
    synthetic_draws=0; real_draws=0; gradient_steps=0
    epoch_training=(args.model=='cnn')
    for t in range(args.rounds):
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        aggregate={k:torch.zeros_like(v) for k,v in state.items()}
        for k,weight in zip(active,weights):
            local.load_state_dict(state); local.train()
            optimizer=torch.optim.SGD(local.parameters(),lr=args.lr,weight_decay=args.mu)
            rng=rngs[k]
            if epoch_training:
                # Practical CNN path: every augmented sample is used once in each
                # local epoch. More AIGC data therefore produces proportionally
                # more local optimizer updates, matching standard data augmentation.
                for _ in range(args.local_epochs):
                    for bx,by,drawn in iter_additive_epoch_batches(plans[k],x,y,sx,sy,args.batch,rng):
                        synthetic_draws+=drawn; real_draws+=len(by)-drawn; gradient_steps+=1
                        optimizer.zero_grad(set_to_none=True)
                        loss=nn.functional.cross_entropy(local(preprocess(bx.to(device),args.model)),by.to(device))
                        loss.backward(); optimizer.step()
            else:
                # The strongly-convex softmax certificate assumes a fixed number h
                # of local stochastic-gradient steps, so preserve that path.
                for _ in range(args.steps):
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
            print(json.dumps(dict(method=out.name,round=t+1,val_acc=val['acc'],
                                  gradient_steps=gradient_steps)),flush=True)
    test=evaluate(model,tx,ty,device,args.model)
    torch.save(model.state_dict(),out/'final_model.pt')
    expected_steps=None
    if epoch_training:
        expected_steps=int(args.rounds*sum(local_epoch_steps(plans[k]['augmented_n'],args.batch,args.local_epochs)
                                           for k in active))
        if expected_steps!=gradient_steps:
            raise AssertionError(f'epoch step count mismatch: expected {expected_steps}, got {gradient_steps}')
    return dict(test=test,validation=history[-1]['validation'],initial_validation=initial,
                history=history,seconds=time.time()-start,
                training_mode=('local_epochs_over_augmented_dataset' if epoch_training else 'fixed_local_steps'),
                local_epochs=(int(args.local_epochs) if epoch_training else None),
                theory_local_steps=int(args.steps),gradient_steps=int(gradient_steps),
                expected_gradient_steps=expected_steps,
                actual_real_draws=int(real_draws),actual_synthetic_draws=int(synthetic_draws),
                augmentation={str(k):{kk:vv for kk,vv in v.items() if kk not in ('real_ids','synthetic_ids')}
                              for k,v in plans.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','cifar100','fmnist'],required=True)
    p.add_argument('--data-root',default='data'); p.add_argument('--aigc-root',default='aigc_imgs')
    p.add_argument('--output',default='outputs/real_baqp_additive_epochs')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--clients',type=int,default=6); p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--budget-fraction',type=float,default=.4)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--steps',type=int,default=5,
                   help='Fixed local-step count h for the certificate/softmax path; CNN training uses --local-epochs')
    p.add_argument('--local-epochs',type=int,default=2,
                   help='CNN local epochs over the full augmented dataset per communication round')
    p.add_argument('--batch',type=int,default=32); p.add_argument('--mu',type=float,default=None)
    p.add_argument('--lr',type=float,default=None); p.add_argument('--model',choices=['cnn','softmax'],default='cnn')
    p.add_argument('--residual',action='store_true',help='Conservative finite-real-data residual objective; excludes generator error')
    p.add_argument('--device',default='cuda'); p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--methods',nargs='+',default=['fedavg_all','no_aigc_budget','random_budget','three_state','public_price','continuous','selected_no_aigc','full_balance_all'])
    p.add_argument('--prepare-only',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    args.mu=(.05 if args.model=='softmax' else .0005) if args.mu is None else args.mu
    args.lr=(.1/(.5+args.mu) if args.model=='softmax' else .05) if args.lr is None else args.lr
    if min(args.rounds,args.steps,args.local_epochs,args.batch,args.eval_every,args.clients)<1 or args.alpha<=0:
        p.error('Positive counts/alpha required')
    if args.model=='cnn' and args.residual: p.error('The analytic residual bound is only implemented for unit-ball softmax')
    if args.device.startswith('cuda') and not args.prepare_only and 'SLURM_JOB_ID' not in os.environ:
        p.error('GPU training must run inside a Slurm allocation')
    torch.set_num_threads(2)
    x,y,tx,ty=load_real(args.data_root,args.dataset); classes=int(y.max())+1
    sx,sy,cache=load_synthetic(args.aigc_root,args.dataset,classes)
    if sx.shape[1:]!=x.shape[1:]: raise ValueError('Real/synthetic shape mismatch')
    parts,valid,counts=split_train(y,classes,args.clients,args.alpha,args.split_seed)
    z=instance(counts,args.split_seed,args.budget_fraction,args.rounds,args.steps,args.batch,
               args.mu,.5+args.mu,args.lr,args.residual)
    pstar=counts.sum(0)/counts.sum()
    out=Path(args.output)/args.dataset/args.model/f'alpha_{args.alpha}_budget_{args.budget_fraction}'/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    setting={k:v for k,v in vars(args).items() if k not in ('resume','prepare_only','methods','output')}
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
        instance={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in z.items()},audit=diagnostic,
        augmentation_semantics='addition_only: retain every original sample and add class-targeted AIGC samples',
        training_semantics=('CNN: complete local epochs over each augmented client multiset; softmax: fixed h local steps'),
        aggregation_semantics='FedAvg weights use original client sample counts, not augmented counts',
        interpretation='Empirical AIGC experiment; J is a proxy for CNN, generator mismatch is not certified; original samples are never replaced',
        accuracy_certificate=None))
    builders={'continuous':lambda:solve(z),'three_state':lambda:solve(z,discrete=True),
              'no_aigc_budget':lambda:solve(z,raw_only=True),'public_price':lambda:public_discrete(z),
              'random_budget':lambda:random_feasible(z,args.split_seed)}
    for method in args.methods:
        dest=out/method; dest.mkdir(exist_ok=True)
        if args.resume and (dest/'result.json').exists(): continue
        if method=='selected_no_aigc':
            cp=out/'continuous'/'contract.json'
            selected=json.loads(cp.read_text()) if cp.exists() else solve(z)
            if selected['status']!='ok': contract=dict(status='infeasible')
            else:
                contract=dict(status='ok',active=selected['active'],u=[0.]*args.clients,
                              total_payment=None,reference_only=True,
                              note='Same participants as continuous, no augmentation; learning ablation, not an implementable contract')
        elif method=='fedavg_all':
            contract=dict(status='ok',active=[True]*args.clients,u=[0.]*args.clients,
                          objective_normalized=objective(z,np.ones(args.clients,dtype=bool),np.zeros(args.clients)),
                          total_payment=None,reference_only=True,
                          note='All-client real-data reference; not a budget-feasible economic mechanism')
        elif method=='full_balance_all':
            contract=dict(status='ok',active=[True]*args.clients,u=[1.]*args.clients,
                          objective_normalized=objective(z,np.ones(args.clients,dtype=bool),np.ones(args.clients)),
                          total_payment=None,reference_only=True,
                          note='All clients retain all original data and add AIGC samples until their label distribution reaches p_star')
        elif method in builders:
            if args.resume and (dest/'contract.json').exists(): contract=json.loads((dest/'contract.json').read_text())
            else: contract=builders[method]()
        else: raise ValueError('Unknown method '+method)
        save(dest/'contract.json',contract)
        print(json.dumps(dict(method=method,contract=contract)),flush=True)
        if args.prepare_only or contract['status']!='ok': continue
        metrics=train(args,contract,parts,counts,pstar,x,y,sx,sy,x[valid],y[valid],tx,ty,dest)
        save(dest/'result.json',dict(dataset=args.dataset,model=args.model,seed=args.seed,method=method,
                                  alpha=args.alpha,budget_fraction=args.budget_fraction,contract=contract,**metrics))

if __name__=='__main__': main()
