"""BAQP Stackelberg mechanism with data-calibrated client types.

The game and server objective are unchanged.  The important change in this
revision is type construction: s_k, alpha_k and beta_k are no longer drawn at
random.  They are deterministically calibrated from each client's original
sample mass and the amount of addition-only AIGC data required to repair its
label skew.

The server offers each client a fixed transfer b_k and a marginal enhancement
price r_k. Conditional on participation, client k solves

    max_{0<=q<=lambda} b + r q - s - alpha q - beta q^2.

With u=q/lambda, the enhancement-cost bracket is A_k u+B_k u^2, where
A_k=alpha_k lambda_k and B_k=beta_k lambda_k^2.  Let

    R_k=max_y p_k(y)/p_*(y),  gamma_k=1-1/R_k.

Addition-only label repair requires, in the continuous-count relaxation,

    m_k(u)/n_k = gamma_k u/(1-gamma_k u).

We calibrate

    A_k = kappa_A gamma_k,
    B_k = kappa_A gamma_k^2/(1-gamma_k).

This matches the true initial marginal synthetic workload and the exact full
workload m_k(1)=n_k(R_k-1).  The resulting quadratic is also a conservative
upper bound on the exact normalized workload for every u in [0,1].

The base coefficient s_k is a common per-unit-original-data cost.  Since the
model multiplies the bracket by d_k=n_k/N, the effective base payment d_k s_k
is proportional to original data mass.  Thus both base and enhancement costs
are tied to observable workload rather than accidental random draws.
"""
import itertools
import numpy as np
from scipy.optimize import minimize, minimize_scalar


def coefficients(rounds=100, steps=5, mu=.05, smooth=.55, lr=None):
    lr = .1 / smooth if lr is None else lr
    if not (mu > 0 and smooth >= mu and 0 < lr < 1 / smooth):
        raise ValueError('Require 0 < lr < 1/L and 0 < mu <= L')
    tau = 1 - mu * lr
    e = np.arange(steps)
    w = tau ** (steps - 1 - e); w /= w.sum()
    a = 1 - tau ** (rounds * steps)
    return np.array([smooth*a/mu**2, 2*smooth**3*lr**2*a/mu**2*(w*e**2).sum(),
                     2*smooth**3*lr**2*a/mu**2*(w*e).sum(), smooth*lr*a/(2*mu)])


def calibrated_types(counts, lam, base_cost_total=1.0, aigc_unit_cost=1.0):
    """Deterministically calibrate (s, alpha, beta) from client data.

    `base_cost_total` is the total base-data cost if all original data are
    purchased.  Because sum_k d_k=1, setting every bracket coefficient
    s_k=base_cost_total makes the effective payment d_k*s_k proportional to
    client k's original sample mass.

    `aigc_unit_cost` is the normalized cost of adding one original-dataset
    equivalent mass of synthetic samples.  The original quadratic Stackelberg
    cost is retained; only its coefficients are calibrated from the
    addition-only workload.
    """
    counts=np.asarray(counts,dtype=float)
    if counts.ndim!=2 or np.any(counts<0):
        raise ValueError('counts must be a nonnegative client-by-class matrix')
    n=counts.sum(1)
    if np.any(n<=0): raise ValueError('every client must have at least one sample')
    if base_cost_total<=0 or aigc_unit_cost<=0:
        raise ValueError('cost scales must be positive')
    total=float(n.sum()); d=n/total; p=counts/n[:,None]; pstar=d@p
    ratios=np.divide(p,pstar[None,:],out=np.zeros_like(p),where=pstar[None,:]>0)
    R=np.maximum(1.0,ratios.max(1))
    gamma=1.0-1.0/R
    missing_ratio=R-1.0
    full_synthetic_n=n*missing_ratio

    # The outer d_k already equals n_k/N.  A common s therefore makes the
    # effective base payment d_k*s proportional to the amount of real data.
    s=np.full(len(n),float(base_cost_total))
    effective_fixed=d*s

    # In u=q/lambda coordinates, d_k(A u+B u^2) is the enhancement payment.
    # A matches the exact workload derivative at zero; A+B matches the exact
    # full missing-data workload.  B = gamma^2/(1-gamma) >= 0.
    A=float(aigc_unit_cost)*gamma
    B=float(aigc_unit_cost)*np.divide(gamma*gamma,1.0-gamma,
                                      out=np.zeros_like(gamma),where=(1.0-gamma)>0)
    lam=np.asarray(lam,dtype=float)
    safe=np.where(lam>0,lam,1.0)
    alpha=A/safe
    beta=B/(safe*safe)
    # q is fixed at zero when lambda=0, but response() still evaluates the
    # closed form before clipping. A harmless positive beta avoids 0/0 there.
    alpha=np.where(lam>0,alpha,0.0)
    beta=np.where(lam>0,np.maximum(beta,np.finfo(float).eps),1.0)
    return dict(s=s,alpha=alpha,beta=beta,R=R,gamma=gamma,
                missing_ratio=missing_ratio,full_synthetic_n=full_synthetic_n,
                effective_fixed_cost=effective_fixed,
                base_cost_total=float(base_cost_total),
                aigc_unit_cost=float(aigc_unit_cost))


def instance(counts, seed=2026, budget_fraction=.4, rounds=100, steps=5,
             batch=32, mu=.05, smooth=.55, lr=None, residual=False,
             base_cost_total=1.0, aigc_unit_cost=1.0):
    n = counts.sum(1); d = n/n.sum(); p = counts/n[:, None]
    e = p - d @ p
    eps = np.sqrt(2)*(2+np.abs(e).sum(1)) if residual else np.zeros(len(n))
    lam = np.sqrt(2)*np.abs(e).sum(1)+eps
    # `seed` is retained for API compatibility; client cost types are now
    # deterministic functions of the data rather than random draws.
    bar = max(1.1*lam.max(), 1e-12)
    types=calibrated_types(counts,lam,base_cost_total,aigc_unit_cost)
    s,alpha,beta=types['s'],types['alpha'],types['beta']
    z = dict(d=d,e=e,lam=lam,bar=bar,s=s,alpha=alpha,beta=beta,eps=eps,
             coef=coefficients(rounds,steps,mu,smooth,lr),g2=2*counts.shape[1],noise=np.full(len(n),2/batch),
             tariff='two_part_signed_fixed_transfer',
             type_calibration='deterministic_additive_missing_workload_quadratic_upper_bound',
             R=types['R'],gamma=types['gamma'],missing_ratio=types['missing_ratio'],
             full_synthetic_n=types['full_synthetic_n'],
             effective_fixed_cost=types['effective_fixed_cost'],
             base_cost_total=types['base_cost_total'],aigc_unit_cost=types['aigc_unit_cost'])
    full_cost = s + alpha*lam + beta*lam**2
    z['full_budget'] = float(d@full_cost); z['budget'] = budget_fraction*z['full_budget']
    return z


def response(z, fixed_transfer, marginal_price, discrete=False):
    """Follower response to a two-part tariff (b_k,r_k)."""
    l,s,al,be = [z[k] for k in ('lam','s','alpha','beta')]
    b=np.asarray(fixed_transfer,dtype=float); r=np.asarray(marginal_price,dtype=float)
    if discrete:
        threshold=al+be*l
        tied_or_above=r >= threshold-8*np.finfo(float).eps*np.maximum(1,np.abs(threshold))
        q=np.where(tied_or_above&(l>0),l,0.)
    else:
        q=np.clip((r-al)/(2*be),0,l)
    utility=b+r*q-s-al*q-be*q*q
    active=utility>=-1e-10
    return active,np.where(active,q/np.where(l>0,l,1),0),z['d']*utility


def tariffs(z, active, u, discrete=False):
    """Minimum personalized two-part tariffs implementing an allocation."""
    l,s,al,be=[z[k] for k in ('lam','s','alpha','beta')]
    q=np.asarray(u,dtype=float)*l
    fixed=np.zeros(len(l)); price=np.zeros(len(l))
    for k in np.flatnonzero(active):
        if l[k]==0 or u[k]<=1e-14:
            price[k]=0.; fixed[k]=s[k]
        elif discrete and u[k]>=1-1e-14:
            price[k]=al[k]+be[k]*l[k]
            fixed[k]=s[k]
        else:
            price[k]=al[k]+2*be[k]*q[k]
            fixed[k]=s[k]-be[k]*q[k]**2
    return fixed,price


def prices(z, active, u, discrete=False):
    """Backward-compatible accessor returning only marginal prices."""
    return tariffs(z,active,u,discrete)[1]


def implementation_payment(z, active, u):
    """Minimum total payment under the complete-information two-part tariff."""
    l,s,al,be=[z[k] for k in ('lam','s','alpha','beta')]
    q=np.asarray(u,dtype=float)*l
    return z['d']*np.asarray(active,dtype=float)*(s+al*q+be*q*q)


def quadratic(z, active):
    a=z['d']*active; a=a/a.sum(); E=z['e']; gram=E@E.T
    bias=np.outer(a,a)*gram
    variance=np.diag(a*np.diag(gram))-bias
    cb,ch,cl,ca=z['coef']/z['coef'][0]
    Q=z['g2']*(cb*bias+ch*variance)
    if np.any(z['eps']):
        ep=z['eps']; Q=2*Q+2*cb*np.outer(a*ep,a*ep)+2*ch*np.diag(a*ep**2)
    constant=cl*(a@z['noise'])+ca*(a*a@z['noise'])
    return Q, float(constant)


def objective(z, active, u):
    Q,c=quadratic(z,active); rho=1-np.asarray(u,dtype=float)
    return float(rho@Q@rho+c)


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
    return dict(active=active.tolist(),u=u.tolist(),fixed_transfer=fixed.tolist(),
                marginal_price=rate.tolist(),price=rate.tolist(),payment=pay.tolist(),
                minimum_cost_payment=minimum.tolist(),total_payment=float(pay.sum()),
                client_utility=np.where(active,util,0).tolist(),
                objective_normalized=objective(z,active,u),budget=z['budget'],
                tariff='two_part_signed_fixed_transfer')


def modes(z, discrete=False, raw_only=False):
    """Allocation modes with one common implementation-cost technology."""
    out=[]
    for k,l in enumerate(z['lam']):
        d,s,al,be=[z[x][k] for x in ('d','s','alpha','beta')]
        inactive=(False,0.,0.,0.,0.,0.)
        raw=(True,0.,0.,0.,0.,d*s)
        if raw_only or l<=0:
            out.append([inactive,raw]); continue
        if discrete:
            full=(True,1.,1.,0.,0.,d*(s+al*l+be*l*l))
            out.append([inactive,raw,full])
        else:
            continuous=(True,0.,1.,d*be*l*l,d*al*l,d*s)
            out.append([inactive,continuous])
    return out


def solve(z, discrete=False, raw_only=False, tol=1e-7, maxiter=800):
    opts=modes(z,discrete,raw_only)
    combinations=int(np.prod([len(o) for o in opts]))
    if combinations>100000:
        raise ValueError('More than 100,000 modes: reduce clients (default 6)')
    best=None; global_lb=np.inf; feasible=0; failed=0
    for mode in itertools.product(*opts):
        m=np.array(mode); active=m[:,0].astype(bool)
        if not active.any(): continue
        low,high,A,B,C=m[:,1:].T
        cost=lambda x: float((A*x*x+B*x+C).sum())
        if cost(low)>z['budget']+1e-12: continue
        feasible+=1; Q,const=quadratic(z,active)
        fun=lambda x: float((1-x)@Q@(1-x)+const)
        jac=lambda x: -2*Q@(1-x)
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
            g=jac(x)
            def lower(mult):
                v=np.where(g+mult*B>=0,low,high).copy()
                curved=mult*A>0
                v[curved]=np.clip(-(g[curved]+mult*B[curved])/(2*mult*A[curved]),low[curved],high[curved])
                return fun(x)-g@x+g@v+mult*(cost(v)-z['budget'])
            dual=minimize_scalar(lambda t:-lower(np.expm1(t)),bounds=(0,30),method='bounded')
            lb=max(lower(0),lower(np.expm1(dual.x)))
        else:
            x=low; lb=fun(x)
        global_lb=min(global_lb,lb)
        if best is None or fun(x)<best['objective_normalized']-1e-15:
            best=package(z,active,x,discrete)
    if best is None: return dict(status='infeasible',budget=z['budget'])
    best.update(status='ok',normalized_lower_bound=float(global_lb),
                normalized_absolute_gap=float(max(0,best['objective_normalized']-global_lb)),
                enumerated_modes=combinations,feasible_modes=feasible,solver_failures=failed,
                converged_to_requested_gap=bool(best['objective_normalized']-global_lb<=tol),
                requested_normalized_gap=tol,
                solver='mode enumeration + convex SLSQP + independent dual lower bounds')
    return best


def public_discrete(z):
    """Common marginal-price two-part-tariff baseline."""
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
            fixed=np.where(active,min_fixed,0.)
            rate=np.where(active,r0,0.)
            try: result=package(z,active,u*active,True,fixed,rate)
            except AssertionError: continue
            if best is None or result['objective_normalized']<best['objective_normalized']:
                best=result
    return dict(status='infeasible') if best is None else dict(status='ok',**best)


def random_feasible(z, seed=31415, samples=2000):
    """Feasible random continuous allocation, without selecting on J or accuracy."""
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
