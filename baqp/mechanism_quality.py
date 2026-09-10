"""Generator-aware BAQP mechanism.

This module is the production candidate that fixes the ideal-AIGC endpoint
artifact without changing the two-part Stackelberg tariff or the data-driven
cost calibration.

Learning model
--------------
For client k and enhancement u_k in [0,1], addition-only label repair gives
synthetic fraction gamma_k u_k.  We separate two effects:

    Delta_k(u) = (1-u_k) M e_k + g^A_k(u),

where the first term is residual label skew and the second is generator
mismatch.  Assume

    ||M||_op <= G_cls,
    ||g^A_k(u)|| <= chi G_cls gamma_k u_k.

Here chi is a dimensionless real/synthetic gradient-gap ratio.  The default
chi=0.54/1.75 is the CIFAR-10 empirical ratio reported by IMFL-AIGC; experiments
should also estimate chi from the actual synthetic cache using baqp.quality.

Using ||x+y||^2 <= 2||x||^2+2||y||^2 and the weighted-variance identity gives

 B^2 <= 2 G_cls^2 ||sum a_k(1-u_k)e_k||^2
        +2 chi^2 G_cls^2 (sum a_k gamma_k u_k)^2,

 H^2 <= 2 G_cls^2 V_label(u)
        +2 chi^2 G_cls^2 sum a_k gamma_k^2 u_k^2.

The second terms are zero at u=0 and strictly penalize excessive synthetic
content whenever chi>0.  Therefore full enhancement is no longer identified
with the true population objective.  The resulting fixed-participant problem
remains a convex quadratic program with a convex quadratic budget.
"""
import itertools
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .mechanism import (coefficients, calibrated_types, response, tariffs, prices,
                        implementation_payment, modes)
from .quality import IMFL_CIFAR10_GAP_RATIO


def instance(counts, seed=2026, budget_fraction=.4, rounds=100, steps=5,
             batch=32, mu=.05, smooth=.55, lr=None, residual=False,
             base_cost_total=1.0, aigc_unit_cost=1.0,
             generator_gap_ratio=IMFL_CIFAR10_GAP_RATIO,
             generator_gap_source='imfl_cifar10_reference'):
    """Construct a generator-aware BAQP instance.

    generator_gap_ratio = chi = g_diff/g_data.  For the paper's main empirical
    results this should be replaced by a value measured on the exact AIGC
    cache; the IMFL CIFAR-10 value is a reproducible external reference only.
    """
    if generator_gap_ratio < 0 or not np.isfinite(generator_gap_ratio):
        raise ValueError('generator_gap_ratio must be finite and nonnegative')
    counts=np.asarray(counts,dtype=float)
    n=counts.sum(1); d=n/n.sum(); p=counts/n[:,None]
    e=p-d@p
    eps=np.sqrt(2)*(2+np.abs(e).sum(1)) if residual else np.zeros(len(n))
    lam=np.sqrt(2)*np.abs(e).sum(1)+eps
    bar=max(1.1*lam.max(),1e-12)
    types=calibrated_types(counts,lam,base_cost_total,aigc_unit_cost)
    s,alpha,beta=types['s'],types['alpha'],types['beta']
    z=dict(d=d,e=e,lam=lam,bar=bar,s=s,alpha=alpha,beta=beta,eps=eps,
           coef=coefficients(rounds,steps,mu,smooth,lr),
           g2=2*counts.shape[1],noise=np.full(len(n),2/batch),
           tariff='two_part_signed_fixed_transfer',
           learning_profile='generator_aware_additive_mixture_bound',
           generator_gap_ratio=float(generator_gap_ratio),
           generator_gap_source=str(generator_gap_source),
           type_calibration='deterministic_additive_missing_workload_quadratic_upper_bound',
           R=types['R'],gamma=types['gamma'],missing_ratio=types['missing_ratio'],
           full_synthetic_n=types['full_synthetic_n'],
           effective_fixed_cost=types['effective_fixed_cost'],
           base_cost_total=types['base_cost_total'],aigc_unit_cost=types['aigc_unit_cost'])
    full_cost=s+alpha*lam+beta*lam**2
    z['full_budget']=float(d@full_cost)
    z['budget']=float(budget_fraction)*z['full_budget']
    return z


def quadratic(z, active):
    """Return label matrix, generator matrix and optimization-noise constant.

    objective(u) = (1-u)' Q_label (1-u) + u' Q_gen u + constant.
    All quantities are normalized by C_B as in the original mechanism.
    """
    active=np.asarray(active,dtype=bool)
    a=z['d']*active
    if a.sum()<=0: raise ValueError('At least one active client is required')
    a=a/a.sum()
    E=z['e']; gram=E@E.T
    bias=np.outer(a,a)*gram
    variance=np.diag(a*np.diag(gram))-bias
    cb,ch,cl,ca=z['coef']/z['coef'][0]

    # Split the real-label and generator residuals with Young's inequality.
    Q_label=2*z['g2']*(cb*bias+ch*variance)
    chi=float(z.get('generator_gap_ratio',0.0))
    ag=a*z['gamma']
    Q_gen=2*z['g2']*chi*chi*(cb*np.outer(ag,ag)+ch*np.diag(a*z['gamma']**2))

    # Optional finite-real-data residual from the earlier robust profile.  It
    # remains attached to the residual real-data direction and therefore does
    # not stand in for generator mismatch.
    if np.any(z['eps']):
        ep=z['eps']
        Q_label=Q_label+2*cb*np.outer(a*ep,a*ep)+2*ch*np.diag(a*ep**2)

    constant=cl*(a@z['noise'])+ca*(a*a@z['noise'])
    return Q_label,Q_gen,float(constant)


def objective_components(z, active, u):
    u=np.asarray(u,dtype=float)
    Ql,Qg,c=quadratic(z,active)
    rho=1-u
    return dict(label=float(rho@Ql@rho),generator=float(u@Qg@u),noise=float(c),
                total=float(rho@Ql@rho+u@Qg@u+c))


def objective(z, active, u):
    return objective_components(z,active,u)['total']


def package(z, active, u, discrete=False, fixed_transfer=None, price=None):
    active=np.asarray(active,dtype=bool); u=np.asarray(u,dtype=float)
    if fixed_transfer is None or price is None:
        fixed,rate=tariffs(z,active,u,discrete)
    else:
        fixed=np.asarray(fixed_transfer,dtype=float); rate=np.asarray(price,dtype=float)
    actual,ru,util=response(z,fixed,rate,discrete)
    if not np.array_equal(actual,active) or not np.allclose(ru,u,atol=2e-6):
        raise AssertionError(('Client response mismatch',active,u,actual,ru,fixed,rate,util))
    q=u*z['lam']; pay=z['d']*(fixed+rate*q)*active
    if pay.sum()>z['budget']+1e-8 or np.any(pay[active]<-1e-10) or np.any(util[active]<-1e-8):
        raise AssertionError('Budget/payment/individual rationality violated')
    minimum=implementation_payment(z,active,u)
    minimum_tariff=(fixed_transfer is None or price is None)
    if minimum_tariff and not np.allclose(pay,minimum,atol=1e-9,rtol=1e-9):
        raise AssertionError(('Minimum tariff does not equal true client cost',pay,minimum))
    pieces=objective_components(z,active,u)
    return dict(active=active.tolist(),u=u.tolist(),fixed_transfer=fixed.tolist(),
                marginal_price=rate.tolist(),price=rate.tolist(),payment=pay.tolist(),
                minimum_cost_payment=minimum.tolist(),total_payment=float(pay.sum()),
                client_utility=np.where(active,util,0).tolist(),
                objective_normalized=pieces['total'],objective_components=pieces,
                budget=z['budget'],tariff='two_part_signed_fixed_transfer',
                learning_profile=z.get('learning_profile'))


def solve(z, discrete=False, raw_only=False, tol=1e-7, maxiter=800):
    """Exact mode enumeration; each fixed-mode problem is convex."""
    opts=modes(z,discrete,raw_only)
    combinations=int(np.prod([len(o) for o in opts]))
    if combinations>100000:
        raise ValueError('More than 100,000 modes: reduce clients or use a heuristic solver')
    best=None; global_lb=np.inf; feasible=0; failed=0
    for mode in itertools.product(*opts):
        m=np.array(mode); active=m[:,0].astype(bool)
        if not active.any(): continue
        low,high,A,B,C=m[:,1:].T
        cost=lambda x: float((A*x*x+B*x+C).sum())
        if cost(low)>z['budget']+1e-12: continue
        feasible+=1
        Ql,Qg,const=quadratic(z,active)
        fun=lambda x: float((1-x)@Ql@(1-x)+x@Qg@x+const)
        jac=lambda x: -2*Ql@(1-x)+2*Qg@x
        free=high>low
        if free.any():
            res=minimize(fun,low,jac=jac,bounds=list(zip(low,high)),method='SLSQP',
                         constraints=[dict(type='ineq',fun=lambda x:z['budget']-cost(x),
                                           jac=lambda x:-(2*A*x+B))],
                         options=dict(ftol=max(1e-14,tol**2),maxiter=maxiter))
            failed+=int(not res.success)
            x=np.clip(res.x,low,high)
            if cost(x)>z['budget']:
                left,right=0.,1.
                for _ in range(55):
                    mid=(left+right)/2
                    if cost(low+mid*(x-low))<=z['budget']: left=mid
                    else: right=mid
                x=low+left*(x-low)
            # Independent convex dual lower bound from the tangent plane at x.
            g=jac(x)
            def lower(mult):
                v=np.where(g+mult*B>=0,low,high).copy()
                curved=mult*A>0
                v[curved]=np.clip(-(g[curved]+mult*B[curved])/(2*mult*A[curved]),
                                   low[curved],high[curved])
                return fun(x)-g@x+g@v+mult*(cost(v)-z['budget'])
            dual=minimize_scalar(lambda t:-lower(np.expm1(t)),bounds=(0,30),method='bounded')
            lb=max(lower(0),lower(np.expm1(dual.x)))
        else:
            x=low; lb=fun(x)
        global_lb=min(global_lb,lb)
        if best is None or fun(x)<best['objective_normalized']-1e-15:
            best=package(z,active,x,discrete)
    if best is None:
        return dict(status='infeasible',budget=z['budget'])
    best.update(status='ok',normalized_lower_bound=float(global_lb),
                normalized_absolute_gap=float(max(0,best['objective_normalized']-global_lb)),
                enumerated_modes=combinations,feasible_modes=feasible,solver_failures=failed,
                converged_to_requested_gap=bool(best['objective_normalized']-global_lb<=tol),
                requested_normalized_gap=tol,
                solver='mode enumeration + convex SLSQP + independent dual lower bounds')
    return best


def public_discrete(z):
    """Legacy common-marginal-price discrete baseline under the new J."""
    l,s,al,be=[z[k] for k in ('lam','s','alpha','beta')]
    switches=al+be*l
    candidates=[0.]
    for t in switches[l>0]:
        candidates.extend([np.nextafter(t,-np.inf),t])
    best=None
    for r0 in np.unique(np.asarray(candidates)):
        full=(r0>=switches-8*np.finfo(float).eps*np.maximum(1,np.abs(switches)))&(l>0)
        u=full.astype(float)
        q=u*l
        surplus=r0*q-al*q-be*q*q
        min_fixed=s-surplus
        for bits in itertools.product([False,True],repeat=len(l)):
            active=np.asarray(bits,dtype=bool)
            if not active.any(): continue
            fixed=np.where(active,min_fixed,0.); rate=np.where(active,r0,0.)
            try: result=package(z,active,u*active,True,fixed,rate)
            except AssertionError: continue
            if best is None or result['objective_normalized']<best['objective_normalized']:
                best=result
    return dict(status='infeasible') if best is None else dict(status='ok',**best)


def random_feasible(z, seed=31415, samples=2000):
    rng=np.random.default_rng(seed); candidates=[]
    for _ in range(samples):
        active=rng.random(len(z['lam']))<.7
        if not active.any(): continue
        u=np.where(active,rng.random(len(active)),0.)
        if implementation_payment(z,active,u).sum()<=z['budget']:
            candidates.append((active,u))
    if not candidates: return dict(status='infeasible')
    active,u=candidates[rng.integers(len(candidates))]
    return dict(status='ok',**package(z,active,u))
