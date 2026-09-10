"""AIGC quality calibration utilities.

The production mixed-data mechanism needs a dimensionless generator mismatch
ratio rather than assuming that label balancing makes synthetic data identical
to real data.  Following the practical calibration idea in IMFL-AIGC, we
measure class-conditional gradients on a small real reference set and a
synthetic set at a common reference model.

For a fixed reference model w_ref, define

    g_data = max_y || E_real[grad ell(w_ref; x,y) | y] ||,
    g_diff = max_y || E_syn [grad ell(w_ref; x,y) | y]
                         - E_real[grad ell(w_ref; x,y) | y] ||.

We expose chi = g_diff / g_data.  IMFL-AIGC writes
    theta = g_diff / (2 g_data),
so chi = 2 theta.  Our additive-data bound uses g_diff directly, hence chi is
the natural dimensionless quantity to transfer between gradient scales.
"""
from __future__ import annotations

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


def _class_mean_gradient(model, x, y, class_id, device, kind, max_samples, rng, batch=256):
    ids=torch.where(y==int(class_id))[0].cpu().numpy()
    if len(ids)==0:
        raise ValueError(f'No samples for class {class_id}')
    if max_samples is not None and len(ids)>int(max_samples):
        ids=rng.choice(ids,size=int(max_samples),replace=False)
    ids=np.asarray(ids,dtype=np.int64)
    model.zero_grad(set_to_none=True)
    model.eval()
    total=len(ids)
    for start in range(0,total,batch):
        take=torch.from_numpy(ids[start:start+batch])
        bx=_preprocess(x[take].to(device),kind)
        by=y[take].to(device)
        loss=nn.functional.cross_entropy(model(bx),by,reduction='sum')/float(total)
        loss.backward()
    pieces=[]
    for p in model.parameters():
        if p.grad is not None:
            pieces.append(p.grad.detach().reshape(-1).cpu().double())
    if not pieces:
        raise ValueError('Reference model has no differentiable parameters')
    return torch.cat(pieces),int(total)


def estimate_gradient_gap(model, real_x, real_y, synthetic_x, synthetic_y,
                          kind, device='cpu', classes=None, max_samples_per_class=512,
                          seed=2026, batch=256):
    """Estimate class-conditional real/synthetic gradient discrepancy.

    This is a calibration diagnostic, not a uniform-in-w certificate.  It uses
    the same initial/reference model for real and generated data and returns
    enough per-class information to audit the resulting scalar ratio.
    """
    device=torch.device(device)
    model=model.to(device)
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
        g_data=g_data,
        g_diff=g_diff,
        generator_gap_ratio=ratio,
        imfl_theta_equivalent=ratio/2.0,
        per_class_real_gradient_norm=real_norm,
        per_class_gradient_gap_norm=diff_norm,
        real_samples_per_class=real_used,
        synthetic_samples_per_class=synthetic_used,
        max_samples_per_class=(None if max_samples_per_class is None else int(max_samples_per_class)),
        reference_model_kind=str(kind),
        note=('Pointwise initial-model calibration following the practical IMFL-AIGC quality '
              'estimation idea; it is not a uniform gradient bound over the whole trajectory.'),
    )
