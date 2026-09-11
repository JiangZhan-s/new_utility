"""Absolute-sample AIGC procurement prototype for many-client BAQP.

Decision semantics
------------------
The server procures q_k >= 0 synthetic samples for client k. There is no
synthetic/real ratio cap and no artificial upper bound such as u_k <= 1.
Numerical upper bounds are derived only from the monetary budget.

For a fixed active set, client k's augmented label distribution is the best
addition-only projection toward p_star after exactly q_k new samples. The
AIGC mismatch term depends on the actual synthetic fraction q_k/(n_k+q_k).

Economic implementation
-----------------------
Client cost is
    C_k(q)=c^R_k + c_A q/N + .5 c_Q q^2/(N n_k),
where c_Q>0 is a weak increasing-marginal-effort term that gives a unique
continuous follower response. It is a cost primitive, not a quantity cap.
The complete-information two-part tariff implements q with
    r_k=C'_k(q), b_k=c^R_k-.5 c2_k q^2,
and payment equals true cost. Signed fixed transfers inherit the caveat of
the legacy production mechanism.
"""
import numpy as np
from scipy.optimize import minimize

from .mechanism import coefficients
from .data_absolute import (optimal_augmented_distribution,
                            optimal_augmented_distribution_with_derivative,repair_quantity)


def instance(counts,budget_fraction=.4,rounds=100,steps=5,batch=32,
             mu=.0005,smooth=.5005,lr=.05,base_cost_total=1.0,
             aigc_unit_cost=1.0,aigc_convex_cost=.1,
             generator_gap_ratio=.06180083094278078,
             generator_gap_source='measured_cifar10_trajectory_reference'):
    counts=np.asarray(counts,dtype=float)
    if counts.ndim!=2 or np.any(counts<0) or np.any(counts.sum(1)<=0):
        raise ValueError('counts must be nonnegative with nonempty clients')
    if min(base_cost_total,aigc_unit_cost,aigc_convex_cost)<=0:
        raise ValueError('cost scales must be positive')
    if budget_fraction<=0 or generator_gap_ratio<0:
        raise ValueError('budget_fraction must be positive and gap nonnegative')
    n=counts.sum(1); N=float(n.sum()); d=n/N; p=counts/n[:,None]; pstar=d@p
    e=p-pstar
    base=base_cost_total*d
    c1=np.full(len(n),aigc_unit_cost/N)
    c2=aigc_convex_cost/(N*n)
    repair=np.array([repair_quantity(counts[k],pstar) for k in range(len(n))],dtype=float)
    if not np.all(np.isfinite(repair)): raise ValueError('non-finite repair quantity')
    z=dict(counts=counts,n=n,N=N,d=d,p=p,pstar=pstar,e=e,
           coef=coefficients(rounds,steps,mu,smooth,lr),g2=2*counts.shape[1],
           noise=np.full(len(n),2/batch),base_cost=base,c1=c1,c2=c2,
           base_cost_total=float(base_cost_total),aigc_unit_cost=float(aigc_unit_cost),
           aigc_convex_cost=float(aigc_convex_cost),
           generator_gap_ratio=float(generator_gap_ratio),
           generator_gap_source=str(generator_gap_source),repair_q=repair,
           tariff='two_part_signed_fixed_transfer_absolute_q',
           learning_profile='generator_aware_absolute_sample_addition')
    z['all_base_cost']=float(base.sum())
    z['reference_repair_budget']=float(client_cost(z,repair,np.ones(len(n),dtype=bool)).sum())
    z['budget']=float(budget_fraction)*z['reference_repair_budget']
    return z


def client_cost(z,q,active=None):
    q=np.asarray(q,dtype=float)
    if active is None: active=np.ones(len(q),dtype=bool)
    a=np.asarray(active,dtype=bool)
    if np.any(q<0): raise ValueError('q must be nonnegative')
    return a*(z['base_cost']+z['c1']*q+.5*z['c2']*q*q)


def budget_implied_upper(z,active):
    """Per-coordinate q upper implied solely by the total budget."""
    a=np.asarray(active,dtype=bool)
    rem=float(z['budget']-z['base_cost'][a].sum())
    out=np.zeros(len(a))
    if rem<0: return out
    for k in np.flatnonzero(a):
        c1=float(z['c1'][k]); c2=float(z['c2'][k])
        out[k]=(-c1+np.sqrt(c1*c1+2*c2*rem))/c2
    return out


def state(z,q,active):
    q=np.asarray(q,dtype=float); a=np.asarray(active,dtype=bool)
    residual=np.zeros_like(z['e']); synfrac=np.zeros(len(q))
    for k in np.flatnonzero(a):
        dist,_=optimal_augmented_distribution(z['counts'][k],z['pstar'],q[k])
        residual[k]=dist-z['pstar']
        synfrac[k]=q[k]/(z['n'][k]+q[k])
    return residual,synfrac


def objective_components(z,active,q):
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    if not active.any(): return dict(label=np.inf,generator=np.inf,noise=np.inf,total=np.inf)
    if np.any(q[~active]>1e-8) or np.any(q<0):
        return dict(label=np.inf,generator=np.inf,noise=np.inf,total=np.inf)
    a=z['d']*active; a=a/a.sum()
    residual,s=state(z,q,active)
    mean=a@residual
    b_label=float(mean@mean)
    second=float(np.sum(a*np.sum(residual*residual,axis=1)))
    v_label=max(0.0,second-b_label)
    cb,ch,cl,ca=z['coef']/z['coef'][0]
    label=2*z['g2']*(cb*b_label+ch*v_label)
    chi=float(z['generator_gap_ratio'])
    gen=2*z['g2']*chi*chi*(cb*float((a@s)**2)+ch*float(a@(s*s)))
    noise=cl*float(a@z['noise'])+ca*float((a*a)@z['noise'])
    return dict(label=float(label),generator=float(gen),noise=float(noise),
                total=float(label+gen+noise))


def objective(z,active,q):
    return objective_components(z,active,q)['total']


def objective_gradient(z,active,q):
    """Piecewise analytic gradient d objective / d q for active clients."""
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    a=z['d']*active; a=a/a.sum()
    residual=np.zeros_like(z['e']); deriv=np.zeros_like(z['e']); s=np.zeros(len(q)); sp=np.zeros(len(q))
    for k in np.flatnonzero(active):
        dist,dd=optimal_augmented_distribution_with_derivative(z['counts'][k],z['pstar'],q[k])
        residual[k]=dist-z['pstar']; deriv[k]=dd
        T=z['n'][k]+q[k]; s[k]=q[k]/T; sp[k]=z['n'][k]/(T*T)
    mean=a@residual; sm=float(a@s)
    cb,ch,_,_=z['coef']/z['coef'][0]
    chi=float(z['generator_gap_ratio']); g=np.zeros(len(q))
    for k in np.flatnonzero(active):
        label=2*z['g2']*((cb-ch)*2*a[k]*float(mean@deriv[k])
                         +ch*2*a[k]*float(residual[k]@deriv[k]))
        gen=2*z['g2']*chi*chi*(cb*2*sm*a[k]*sp[k]+ch*2*a[k]*s[k]*sp[k])
        g[k]=label+gen
    return g


def tariffs(z,active,q):
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    rate=np.zeros(len(q)); fixed=np.zeros(len(q))
    rate[active]=z['c1'][active]+z['c2'][active]*q[active]
    fixed[active]=z['base_cost'][active]-.5*z['c2'][active]*q[active]**2
    return fixed,rate


def response(z,fixed,rate):
    fixed=np.asarray(fixed,dtype=float); rate=np.asarray(rate,dtype=float)
    q=np.maximum((rate-z['c1'])/z['c2'],0.0)
    util=fixed+rate*q-(z['base_cost']+z['c1']*q+.5*z['c2']*q*q)
    active=util>=-1e-9
    return active,np.where(active,q,0.0),util


def package(z,active,q,heuristic_note=None):
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    fixed,rate=tariffs(z,active,q)
    ra,rq,util=response(z,fixed,rate)
    if not np.array_equal(ra,active) or not np.allclose(rq,q,atol=2e-4,rtol=1e-8):
        raise AssertionError(('response mismatch',active,q,ra,rq))
    payment=client_cost(z,q,active)
    if payment.sum()>z['budget']+1e-7: raise AssertionError('budget violated')
    pieces=objective_components(z,active,q)
    frac=np.divide(q,z['n']+q,out=np.zeros_like(q),where=(z['n']+q)>0)
    out=dict(status='ok',active=active.tolist(),q_synthetic=q.tolist(),
             synthetic_fraction=frac.tolist(),augmented_n=(z['n']+q).tolist(),
             fixed_transfer=fixed.tolist(),marginal_price=rate.tolist(),
             payment=payment.tolist(),total_payment=float(payment.sum()),
             client_utility=np.where(active,util,0).tolist(),
             objective_normalized=pieces['total'],objective_components=pieces,
             budget=float(z['budget']),reference_repair_q=z['repair_q'].tolist(),
             reference_repair_budget=float(z['reference_repair_budget']),
             all_base_cost=float(z['all_base_cost']),tariff=z['tariff'],
             learning_profile=z['learning_profile'])
    if heuristic_note is not None:
        out['heuristic']=True; out['heuristic_note']=heuristic_note
    return out


def solve_all_clients_heuristic(z,restarts=4,seed=2026,tol=1e-10,maxiter=180):
    """Multi-start local solver with all clients active and no q-ratio cap.

    x=q/n is only a numerical scaling coordinate. It has no exogenous upper
    bound; xhigh below is exactly the monetary-budget-implied q upper divided
    by n. Reported and implemented allocations are absolute q sample counts.
    """
    active=np.ones(len(z['n']),dtype=bool)
    base=float(z['base_cost'].sum())
    if base>z['budget']+1e-12:
        return dict(status='infeasible',budget=float(z['budget']),reason='budget below all-client real-data cost')
    qhigh=budget_implied_upper(z,active); xhigh=qhigh/z['n']
    bounds=[(0.0,float(v)) for v in xhigh]
    fun=lambda x: objective(z,active,np.asarray(x,float)*z['n'])
    jac=lambda x: objective_gradient(z,active,np.asarray(x,float)*z['n'])*z['n']
    con=lambda x: float(z['budget']-client_cost(z,np.asarray(x,float)*z['n'],active).sum())
    cjac=lambda x: -(z['c1']+z['c2']*(np.asarray(x,float)*z['n']))*z['n']
    rem=max(0.0,z['budget']-base); starts=[np.zeros(len(active))]

    def quantities_from_spend(spend):
        spend=np.asarray(spend,dtype=float); q=np.zeros(len(active))
        for k in range(len(active)):
            c1=float(z['c1'][k]); c2=float(z['c2'][k]); v=max(0.0,float(spend[k]))
            q[k]=(-c1+np.sqrt(c1*c1+2*c2*v))/c2
        return q

    if rem>0:
        equal=np.full(len(active),rem/len(active))
        for scale in (.35,.75,.98): starts.append(quantities_from_spend(scale*equal)/z['n'])
        rng=np.random.default_rng(seed)
        for _ in range(restarts):
            shares=rng.dirichlet(np.full(len(active),.8)); utilization=rng.uniform(.45,1.0)
            starts.append(quantities_from_spend(rem*utilization*shares)/z['n'])

    best=None; failures=0
    for x0 in starts:
        res=minimize(fun,x0,jac=jac,method='SLSQP',bounds=bounds,
                     constraints=[dict(type='ineq',fun=con,jac=cjac)],
                     options=dict(ftol=tol,maxiter=maxiter))
        failures+=int(not res.success)
        x=np.clip(res.x,0,xhigh); q=x*z['n']
        if client_cost(z,q,active).sum()>z['budget']+1e-7:
            lo,hi=0.0,1.0
            for _ in range(60):
                mid=(lo+hi)/2
                if client_cost(z,mid*q,active).sum()<=z['budget']: lo=mid
                else: hi=mid
            q*=lo
        cand=(float(objective(z,active,q)),q,bool(res.success))
        if best is None or cand[0]<best[0]: best=cand
    note=('all clients fixed active; multi-start SLSQP local heuristic over absolute synthetic sample counts; '
          'q>=0 and monetary budget are the only quantity restrictions. q/n appears internally only '
          'as numerical scaling and is not a model constraint or calibration cap.')
    result=package(z,active,best[1],note)
    result.update(heuristic_restarts=int(len(starts)),solver_failures=int(failures),
                  local_solver_success=bool(best[2]),solver='multi-start SLSQP local heuristic with piecewise analytic gradient')
    return result


def uniform_budget(z):
    """All clients receive the same absolute q, chosen only by the budget."""
    active=np.ones(len(z['n']),dtype=bool)
    if z['base_cost'].sum()>z['budget']+1e-12:
        return dict(status='infeasible',budget=float(z['budget']))
    lo=0.0; hi=float(np.max(budget_implied_upper(z,active)))
    for _ in range(80):
        mid=(lo+hi)/2
        q=np.full(len(active),mid)
        if client_cost(z,q,active).sum()<=z['budget']: lo=mid
        else: hi=mid
    q=np.full(len(active),lo)
    out=package(z,active,q)
    out['baseline']='uniform absolute AIGC count per client under the same budget'
    return out


def reference_all_repair(z):
    """All-client exact label-repair reference; intentionally not budget constrained."""
    active=np.ones(len(z['n']),dtype=bool); q=z['repair_q'].copy()
    pieces=objective_components(z,active,q)
    payment=client_cost(z,q,active)
    frac=q/(z['n']+q)
    return dict(status='ok',active=active.tolist(),q_synthetic=q.tolist(),
                synthetic_fraction=frac.tolist(),augmented_n=(z['n']+q).tolist(),
                payment=payment.tolist(),total_payment=float(payment.sum()),budget=float(z['budget']),
                objective_normalized=pieces['total'],objective_components=pieces,
                reference_only=True,budget_feasible=bool(payment.sum()<=z['budget']+1e-9),
                note='All-client exact label-repair reference; q_repair is not a cap on the continuous mechanism.')
