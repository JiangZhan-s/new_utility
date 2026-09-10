"""Equations 17--46 of FINAL_GUIDE.md; numerical, not interval certificates."""
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


def instance(counts, seed=2026, budget_fraction=.4, rounds=100, steps=5,
             batch=32, mu=.05, smooth=.55, lr=None, residual=False):
    n = counts.sum(1); d = n/n.sum(); p = counts/n[:, None]
    e = p - d @ p
    eps = np.sqrt(2)*(2+np.abs(e).sum(1)) if residual else np.zeros(len(n))
    lam = np.sqrt(2)*np.abs(e).sum(1)+eps
    bar = max(1.1*lam.max(), 1e-12)
    rng = np.random.default_rng(seed)
    s = rng.uniform(.01,.03,len(n)); ac = rng.uniform(.02,.08,len(n)); bc = rng.uniform(.04,.12,len(n))
    alpha = ac/np.where(lam>0,lam,1); beta = bc/np.where(lam>0,lam**2,1)
    z = dict(d=d,e=e,lam=lam,bar=bar,s=s,alpha=alpha,beta=beta,eps=eps,
             coef=coefficients(rounds,steps,mu,smooth,lr),g2=2*counts.shape[1],noise=np.full(len(n),2/batch))
    full = np.maximum(bar*(alpha+2*beta*lam),s+alpha*lam+beta*lam**2)
    full = np.where(lam>0,full,s)
    z['full_budget'] = float(d@full); z['budget'] = budget_fraction*z['full_budget']
    return z


def response(z, price, discrete=False):
    l,b,s,al,be = [z[k] for k in ('lam','bar','s','alpha','beta')]
    c = 1-l/b
    if discrete:
        raw = price*c-s; full = price-s-al*l-be*l*l
        # Compare the analytically equivalent switching price, avoiding
        # cancellation between two nearly equal utilities at a tie.
        threshold=b*(al+be*l)
        tied_or_above=price >= threshold-8*np.finfo(float).eps*np.maximum(1,np.abs(threshold))
        q = np.where(tied_or_above&(l>0),l,0)
    else:
        q = np.clip((price/b-al)/(2*be),0,l)
    utility = price*(c+q/b)-s-al*q-be*q*q
    active = utility >= -1e-10  # documented zero-utility participation convention
    return active, np.where(active,q/np.where(l>0,l,1),0), z['d']*utility


def prices(z, active, u, discrete=False):
    l,b,s,al,be = [z[k] for k in ('lam','bar','s','alpha','beta')]
    q=u*l; c=1-l/b
    r=np.zeros(len(l))
    for k in np.flatnonzero(active):
        if l[k]==0 or u[k]==0:
            r[k]=s[k]/c[k]
        elif u[k]==1:
            r[k]=max(b*(al[k]+(1 if discrete else 2)*be[k]*l[k]),s[k]+al[k]*l[k]+be[k]*l[k]**2)
        else:
            r[k]=b*(al[k]+2*be[k]*q[k])
    return r


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
    Q,c=quadratic(z,active); rho=1-u
    return float(rho@Q@rho+c)


def package(z, active, u, discrete=False, price=None):
    r=prices(z,active,u,discrete) if price is None else price
    actual, ru, util=response(z,r,discrete)
    if not np.array_equal(actual,active) or not np.allclose(ru,u,atol=2e-6):
        raise AssertionError(('Client response mismatch',active,u,actual,ru,r,util))
    pay=z['d']*r*(1-z['lam']/z['bar']+u*z['lam']/z['bar'])
    if pay.sum()>z['budget']+1e-8 or np.any(util[active]<-1e-8):
        raise AssertionError('Budget/individual rationality violated')
    return dict(active=active.tolist(),u=u.tolist(),price=r.tolist(),payment=pay.tolist(),
                total_payment=float(pay.sum()),client_utility=np.where(active,util,0).tolist(),
                objective_normalized=objective(z,active,u),budget=z['budget'])


def modes(z, discrete=False, raw_only=False):
    out=[]
    for k,l in enumerate(z['lam']):
        d,s,al,be=[z[x][k] for x in ('d','s','alpha','beta')]; t=z['bar']-l
        opts=[(False,0.,0.,0.,0.,0.)]
        raw_ok = l==0 or (s<t*(al+be*l) if discrete else s<=al*t)
        if raw_ok:
            opts.append((True,0.,0.,0.,0.,d*s))
        if l>0 and not raw_only:
            if not discrete:
                low=0 if s<=al*t else (s-al*t)/(be*(np.sqrt(t*t+(s-al*t)/be)+t))
                if low<=l:
                    opts.append((True,low/l,1.,2*d*be*l*l,d*(al*l+2*be*t*l),d*al*t))
            full=max(z['bar']*(al+(1 if discrete else 2)*be*l),s+al*l+be*l*l)
            opts.append((True,1.,1.,0.,0.,d*full))
        out.append(opts)
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
            # Every multiplier yields a valid convex supporting-hyperplane bound.
            dual=minimize_scalar(lambda t:-lower(np.expm1(t)),bounds=(0,30),method='bounded')
            lb=max(lower(0),lower(np.expm1(dual.x)))
        else:
            x=low; lb=fun(x)
        global_lb=min(global_lb,lb)
        # Closed continuous mode at zero can be implemented more cheaply as raw.
        if best is None or fun(x)<best['objective_normalized']:
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
    l,b,s,al,be=[z[k] for k in ('lam','bar','s','alpha','beta')]
    candidates=np.unique(np.r_[0,s/(1-l/b),s+al*l+be*l*l,b*(al+be*l)])
    best=None
    for price in candidates:
        r=np.full(len(l),price); active,u,_=response(z,r,True)
        r=np.where(active,r,0)
        if not active.any(): continue
        try: result=package(z,active,u,True,r)
        except AssertionError: continue
        if best is None or result['objective_normalized']<best['objective_normalized']: best=result
    return dict(status='infeasible') if best is None else dict(status='ok',**best)


def random_feasible(z, seed=31415, samples=2000):
    """Feasible random allocation, without selecting on J or accuracy."""
    rng=np.random.default_rng(seed); opts=modes(z); candidates=[]
    for _ in range(samples):
        m=np.array([o[rng.integers(len(o))] for o in opts]); active=m[:,0].astype(bool)
        u=rng.uniform(m[:,1],m[:,2]); A,B,C=m[:,3:].T
        if active.any() and (A*u*u+B*u+C).sum()<=z['budget']:
            candidates.append((active,u))
    if not candidates: return dict(status='infeasible')
    active,u=candidates[rng.integers(len(candidates))]
    return dict(status='ok',**package(z,active,u))
