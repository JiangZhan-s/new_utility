"""AIGC quality calibration utilities.

Two calibration modes are provided.

1. ``estimate_gradient_gap`` is the cheap pointwise diagnostic used in the
   first generator-aware prototype.
2. ``estimate_gradient_gap_trajectory`` follows the empirical procedure used
   by IMFL-AIGC more closely: run a short *real-data-only* FL probe trajectory
   and record maxima of the real gradient norm and the real-vs-generated
   gradient discrepancy along that trajectory.

The trajectory estimator also records a cross-entropy distribution-gap proxy.
Neither empirical estimator is a mathematical supremum over the complete
optimization region, so the output metadata deliberately calls them
calibrations/proxies rather than certified uniform constants.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn


IMFL_CIFAR10_G_DATA = 1.75
IMFL_CIFAR10_G_DIFF = 0.54
IMFL_CIFAR10_THETA = IMFL_CIFAR10_G_DIFF / (2.0 * IMFL_CIFAR10_G_DATA)
IMFL_CIFAR10_GAP_RATIO = IMFL_CIFAR10_G_DIFF / IMFL_CIFAR10_G_DATA


def _preprocess(x, kind):
    x=x.float()/255
    if kind=='softmax':
        x=x/torch.linalg.vector_norm(x.flatten(1),dim=1).clamp_min(1)[:,None,None,None]
    else:
        x=(x-.5)/.5
    return x


def _gradient_vector(model):
    pieces=[]
    for p in model.parameters():
        if p.grad is not None:
            pieces.append(p.grad.detach().reshape(-1).cpu().double())
    if not pieces:
        raise ValueError('Reference model has no differentiable parameters')
    return torch.cat(pieces)


def _mean_gradient_from_ids(model, x, y, ids, device, kind, batch=256):
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)==0: raise ValueError('Cannot estimate a gradient from zero samples')
    model.zero_grad(set_to_none=True); model.eval(); total=len(ids)
    for start in range(0,total,batch):
        take=torch.from_numpy(ids[start:start+batch])
        bx=_preprocess(x[take].to(device),kind); by=y[take].to(device)
        loss=nn.functional.cross_entropy(model(bx),by,reduction='sum')/float(total)
        loss.backward()
    return _gradient_vector(model)


@torch.no_grad()
def _mean_ce_from_ids(model, x, y, ids, device, kind, batch=256):
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)==0: raise ValueError('Cannot estimate a loss from zero samples')
    model.eval(); total=0.0
    for start in range(0,len(ids),batch):
        take=torch.from_numpy(ids[start:start+batch])
        bx=_preprocess(x[take].to(device),kind); by=y[take].to(device)
        total+=float(nn.functional.cross_entropy(model(bx),by,reduction='sum'))
    return total/len(ids)


def _class_mean_gradient(model, x, y, class_id, device, kind, max_samples, rng, batch=256):
    ids=torch.where(y==int(class_id))[0].cpu().numpy()
    if len(ids)==0:
        raise ValueError(f'No samples for class {class_id}')
    if max_samples is not None and len(ids)>int(max_samples):
        ids=rng.choice(ids,size=int(max_samples),replace=False)
    ids=np.asarray(ids,dtype=np.int64)
    return _mean_gradient_from_ids(model,x,y,ids,device,kind,batch),int(len(ids))


def estimate_gradient_gap(model, real_x, real_y, synthetic_x, synthetic_y,
                          kind, device='cpu', classes=None, max_samples_per_class=512,
                          seed=2026, batch=256):
    """Pointwise class-conditional real/synthetic gradient discrepancy."""
    device=torch.device(device); model=model.to(device)
    classes=int(real_y.max())+1 if classes is None else int(classes)
    if int(synthetic_y.max())+1 < classes:
        raise ValueError('Synthetic data are missing required classes')
    real_norm=[]; diff_norm=[]; real_used=[]; synthetic_used=[]
    for c in range(classes):
        rr=np.random.default_rng(np.random.SeedSequence([seed,901,c]))
        sr=np.random.default_rng(np.random.SeedSequence([seed,902,c]))
        gr,nr=_class_mean_gradient(model,real_x,real_y,c,device,kind,
                                   max_samples_per_class,rr,batch)
        gs,ns=_class_mean_gradient(model,synthetic_x,synthetic_y,c,device,kind,
                                   max_samples_per_class,sr,batch)
        real_norm.append(float(torch.linalg.vector_norm(gr)))
        diff_norm.append(float(torch.linalg.vector_norm(gs-gr)))
        real_used.append(nr); synthetic_used.append(ns)
    g_data=float(max(real_norm)); g_diff=float(max(diff_norm))
    if not np.isfinite(g_data) or g_data<=0:
        raise ValueError('Estimated g_data must be positive and finite')
    ratio=float(g_diff/g_data)
    return dict(
        method='initial_model_class_conditional_gradient_gap',
        g_data=g_data,g_diff=g_diff,generator_gap_ratio=ratio,
        imfl_theta_equivalent=ratio/2.0,
        per_class_real_gradient_norm=real_norm,
        per_class_gradient_gap_norm=diff_norm,
        real_samples_per_class=real_used,synthetic_samples_per_class=synthetic_used,
        max_samples_per_class=(None if max_samples_per_class is None else int(max_samples_per_class)),
        reference_model_kind=str(kind),
        note=('Pointwise initial-model calibration. It is useful as a cheap diagnostic but '
              'can substantially understate a trajectory envelope.'),
    )


def _probe_real_ids(parts, max_samples, seed):
    out=[]
    for k,p in enumerate(parts):
        ids=np.asarray(p,dtype=np.int64)
        if max_samples is not None and len(ids)>int(max_samples):
            rng=np.random.default_rng(np.random.SeedSequence([seed,930,k]))
            ids=rng.choice(ids,size=int(max_samples),replace=False)
        out.append(np.asarray(ids,dtype=np.int64))
    return out


def _matched_synthetic_ids(real_ids, real_y, synthetic_y, classes, seed, client):
    """Synthetic probe with exactly the same empirical label histogram as real_ids."""
    labels=np.asarray(real_y[torch.from_numpy(np.asarray(real_ids,dtype=np.int64))])
    counts=np.bincount(labels,minlength=classes)
    chosen=[]
    rng=np.random.default_rng(np.random.SeedSequence([seed,931,client]))
    sy=np.asarray(synthetic_y)
    for c,m in enumerate(counts):
        if m<=0: continue
        pool=np.flatnonzero(sy==c)
        if len(pool)==0: raise ValueError(f'No synthetic samples for class {c}')
        chosen.append(rng.choice(pool,size=int(m),replace=(m>len(pool))))
    return np.concatenate(chosen).astype(np.int64) if chosen else np.empty(0,dtype=np.int64)


def estimate_gradient_gap_trajectory(model, real_x, real_y, synthetic_x, synthetic_y,
                                     parts, kind, device='cpu', classes=None,
                                     rounds=20, local_steps=5, train_batch=32,
                                     probe_batch=256, max_probe_samples_per_client=512,
                                     lr=.05, weight_decay=0.0, seed=2026,
                                     probe_every=1):
    """IMFL-style max-over-trajectory generator-quality calibration.

    A short FedAvg trajectory is trained **only on real client data**.  At every
    probe checkpoint and for every client, the routine compares the gradient of
    a fixed real probe subset with a synthetic probe subset having exactly the
    same label histogram.  Consequently ``g_diff`` isolates conditional
    real/synthetic mismatch instead of confounding it with label skew.

    We report

        g_data = max_{t,k} ||grad F^real_k(w_t)||,
        g_diff = max_{t,k} ||grad F^syn,k-matched_k(w_t)-grad F^real_k(w_t)||,

    as well as

        ce_gap = max_{t,k} |CE_syn,k-matched(w_t)-CE_real,k(w_t)|.

    The latter is a train-only empirical risk-discrepancy proxy useful for
    auditing generalization-shift terms.  No validation/test data are used.
    """
    if min(int(rounds),int(local_steps),int(train_batch),int(probe_batch),int(probe_every))<1:
        raise ValueError('rounds/local_steps/batches/probe_every must be positive')
    if lr<=0 or weight_decay<0: raise ValueError('Require lr>0 and weight_decay>=0')
    device=torch.device(device); model=model.to(device)
    classes=int(real_y.max())+1 if classes is None else int(classes)
    if len(parts)<1: raise ValueError('At least one client is required')
    if int(synthetic_y.max())+1 < classes: raise ValueError('Synthetic data are missing required classes')

    probe_real=_probe_real_ids(parts,max_probe_samples_per_client,seed)
    probe_syn=[_matched_synthetic_ids(ids,real_y,synthetic_y,classes,seed,k)
               for k,ids in enumerate(probe_real)]
    if any(len(x)==0 for x in probe_real) or any(len(x)==0 for x in probe_syn):
        raise ValueError('Every client must have non-empty real and matched-synthetic probes')

    weights=np.array([len(p) for p in parts],dtype=float); weights/=weights.sum()
    train_rng=[np.random.default_rng(np.random.SeedSequence([seed,932,k])) for k in range(len(parts))]
    local=copy.deepcopy(model).to(device)
    checkpoints=[]

    def probe(round_index):
        rows=[]
        for k,(rid,sid) in enumerate(zip(probe_real,probe_syn)):
            gr=_mean_gradient_from_ids(model,real_x,real_y,rid,device,kind,probe_batch)
            gs=_mean_gradient_from_ids(model,synthetic_x,synthetic_y,sid,device,kind,probe_batch)
            rn=float(torch.linalg.vector_norm(gr)); dn=float(torch.linalg.vector_norm(gs-gr))
            rce=_mean_ce_from_ids(model,real_x,real_y,rid,device,kind,probe_batch)
            sce=_mean_ce_from_ids(model,synthetic_x,synthetic_y,sid,device,kind,probe_batch)
            rows.append(dict(client=int(k),real_gradient_norm=rn,gradient_gap_norm=dn,
                             real_ce=float(rce),synthetic_ce=float(sce),ce_gap=abs(float(sce-rce)),
                             real_probe_n=int(len(rid)),synthetic_probe_n=int(len(sid))))
        checkpoints.append(dict(round=int(round_index),clients=rows,
                                max_real_gradient_norm=float(max(r['real_gradient_norm'] for r in rows)),
                                max_gradient_gap_norm=float(max(r['gradient_gap_norm'] for r in rows)),
                                max_ce_gap=float(max(r['ce_gap'] for r in rows))))

    torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    probe(0)
    for t in range(1,int(rounds)+1):
        state={name:value.detach().clone() for name,value in model.state_dict().items()}
        aggregate={name:torch.zeros_like(value) for name,value in state.items()}
        for k,(part,w) in enumerate(zip(parts,weights)):
            part=np.asarray(part,dtype=np.int64)
            local.load_state_dict(state); local.train()
            opt=torch.optim.SGD(local.parameters(),lr=float(lr),weight_decay=float(weight_decay))
            rng=train_rng[k]
            for _ in range(int(local_steps)):
                ids=rng.choice(part,size=int(train_batch),replace=(len(part)<int(train_batch)))
                take=torch.from_numpy(np.asarray(ids,dtype=np.int64))
                bx=_preprocess(real_x[take].to(device),kind); by=real_y[take].to(device)
                opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(local(bx),by)
                loss.backward(); opt.step()
            for name,value in local.state_dict().items():
                aggregate[name].add_(value,alpha=float(w))
        model.load_state_dict(aggregate)
        if t%int(probe_every)==0 or t==int(rounds): probe(t)

    g_data=max(c['max_real_gradient_norm'] for c in checkpoints)
    g_diff=max(c['max_gradient_gap_norm'] for c in checkpoints)
    ce_gap=max(c['max_ce_gap'] for c in checkpoints)
    if not np.isfinite(g_data) or g_data<=0: raise ValueError('Estimated g_data must be positive and finite')
    ratio=float(g_diff/g_data)
    max_gdata_at=max(checkpoints,key=lambda c:c['max_real_gradient_norm'])['round']
    max_gdiff_at=max(checkpoints,key=lambda c:c['max_gradient_gap_norm'])['round']
    max_ce_at=max(checkpoints,key=lambda c:c['max_ce_gap'])['round']
    return dict(
        method='real_fedavg_trajectory_matched_label_gradient_gap',
        g_data=float(g_data),g_diff=float(g_diff),generator_gap_ratio=ratio,
        imfl_theta_equivalent=ratio/2.0,
        ce_gap=float(ce_gap),
        max_g_data_round=int(max_gdata_at),max_g_diff_round=int(max_gdiff_at),
        max_ce_gap_round=int(max_ce_at),
        rounds=int(rounds),local_steps=int(local_steps),train_batch=int(train_batch),
        probe_batch=int(probe_batch),probe_every=int(probe_every),
        max_probe_samples_per_client=(None if max_probe_samples_per_client is None else int(max_probe_samples_per_client)),
        lr=float(lr),weight_decay=float(weight_decay),reference_model_kind=str(kind),
        checkpoints=checkpoints,
        note=('Train-only trajectory-envelope calibration. It mirrors IMFL-AIGC\'s empirical '
              'max-over-short-FL-training strategy more closely than the initial-model proxy; '
              'it is still an empirical envelope, not a formal supremum over all w.'),
    )
