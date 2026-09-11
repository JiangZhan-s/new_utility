"""Joint client-selection + absolute-count AIGC procurement prototype.

This model removes the all-clients-active assumption. The server chooses

    x_k in {0,1}, q_k >= 0,

subject only to a monetary budget. There is no synthetic/real ratio cap and no
q<=repair constraint.

To avoid the participant-set renormalization artifact, aggregation keeps the
original global client weights d_k. A nonparticipant retains the current global
model in the weighted model average, so a one-round update has direction

    sum_k d_k x_k g_k(q_k).

Under the label-skew decomposition

    g_k(q) = g + M e_k(q) + r^A_k(q),

with ||g|| <= G, ||M|| <= G_cls and
||r^A_k(q)|| <= chi G_cls s_k(q), s_k=q/(n_k+q),

we have

    G_{x,q} - g
      = -(1-D_x) g + M sum_k d_k x_k e_k(q)
        + sum_k d_k x_k r^A_k(q),

where D_x=sum_k d_k x_k. Thus one selected, fully repaired client cannot make
selection bias vanish: the missing-mass term (1-D_x)^2 remains unless D_x=1.
No artificial participant-count reward is added.

The normalized objective below combines this fixed-mass aggregate-bias bound
with a local-drift second-moment bound. It is a mechanism prototype; the old
renormalized-participant Theorem 1 does not apply verbatim and needs a separate
fixed-mass convergence statement before theorem-level claims are made.
"""
import numpy as np
from scipy.optimize import minimize

from .mechanism import coefficients
from .mechanism_absolute import client_cost, budget_implied_upper, tariffs, response
from .data_absolute import (optimal_augmented_distribution,
                            optimal_augmented_distribution_with_derivative,
                            repair_quantity)


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
        raise ValueError('invalid budget/generator gap')
    n=counts.sum(1); N=float(n.sum()); d=n/N; p=counts/n[:,None]; pstar=d@p
    e=p-pstar
    base=base_cost_total*d
    c1=np.full(len(n),aigc_unit_cost/N)
    c2=aigc_convex_cost/(N*n)
    repair=np.array([repair_quantity(counts[k],pstar) for k in range(len(n))],dtype=float)
    z=dict(counts=counts,n=n,N=N,d=d,p=p,pstar=pstar,e=e,
           coef=coefficients(rounds,steps,mu,smooth,lr),g2=2*counts.shape[1],
           noise=np.full(len(n),2/batch),base_cost=base,c1=c1,c2=c2,
           base_cost_total=float(base_cost_total),aigc_unit_cost=float(aigc_unit_cost),
           aigc_convex_cost=float(aigc_convex_cost),
           generator_gap_ratio=float(generator_gap_ratio),
           generator_gap_source=str(generator_gap_source),repair_q=repair,
           tariff='two_part_signed_fixed_transfer_absolute_q',
           learning_profile='fixed_mass_joint_selection_absolute_q')
    z['all_base_cost']=float(base.sum())
    z['reference_repair_budget']=float(client_cost(z,repair,np.ones(len(n),dtype=bool)).sum())
    z['budget']=float(budget_fraction)*z['reference_repair_budget']
    return z


def _state(z,active,q,derivative=False):
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    residual=np.zeros_like(z['e']); syn=np.zeros(len(q)); deriv=np.zeros_like(z['e']); sp=np.zeros(len(q))
    for k in np.flatnonzero(active):
        if derivative:
            dist,dd=optimal_augmented_distribution_with_derivative(z['counts'][k],z['pstar'],q[k])
            deriv[k]=dd
        else:
            dist,_=optimal_augmented_distribution(z['counts'][k],z['pstar'],q[k])
        residual[k]=dist-z['pstar']
        T=z['n'][k]+q[k]; syn[k]=q[k]/T
        if derivative: sp[k]=z['n'][k]/(T*T)
    return residual,syn,deriv,sp


def objective_components(z,active,q):
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    if not active.any() or np.any(q<0) or np.any(q[~active]>1e-8):
        return dict(missing=np.inf,label=np.inf,generator=np.inf,local=np.inf,noise=np.inf,total=np.inf)
    d=z['d']; dx=d*active; D=float(dx.sum())
    residual,s,_,_=_state(z,active,q,False)
    mean=dx@residual
    sm=float(dx@s)
    local_label=float(np.sum(dx*np.sum(residual*residual,axis=1)))
    local_gen=float(dx@(s*s))
    cb,ch,cl,ca=z['coef']/z['coef'][0]
    g2=float(z['g2']); chi=float(z['generator_gap_ratio'])
    # Three-term Young bound for aggregate bias:
    # missing global mass + selected label residual + selected generator residual.
    missing=3*g2*cb*(1-D)**2
    label=3*g2*cb*float(mean@mean)
    generator=3*g2*cb*chi*chi*sm*sm
    # Local-drift second moment around the global reference.
    local=2*g2*ch*(local_label+chi*chi*local_gen)
    # Fixed-mass stochastic terms: no renormalization by the active set.
    noise=cl*float(dx@z['noise'])+ca*float((dx*dx)@z['noise'])
    total=missing+label+generator+local+noise
    return dict(missing=float(missing),label=float(label),generator=float(generator),
                local=float(local),noise=float(noise),total=float(total),active_mass=D)


def objective(z,active,q):
    return objective_components(z,active,q)['total']


def objective_gradient(z,active,q):
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    d=z['d']; dx=d*active
    residual,s,deriv,sp=_state(z,active,q,True)
    mean=dx@residual; sm=float(dx@s)
    cb,ch,_,_=z['coef']/z['coef'][0]
    g2=float(z['g2']); chi=float(z['generator_gap_ratio'])
    grad=np.zeros(len(q))
    for k in np.flatnonzero(active):
        aggregate_label=3*g2*cb*2*d[k]*float(mean@deriv[k])
        aggregate_gen=3*g2*cb*chi*chi*2*sm*d[k]*sp[k]
        local=2*g2*ch*(2*d[k]*float(residual[k]@deriv[k])
                       +chi*chi*2*d[k]*s[k]*sp[k])
        grad[k]=aggregate_label+aggregate_gen+local
    return grad


def package(z,active,q,note=None):
    active=np.asarray(active,dtype=bool); q=np.asarray(q,dtype=float)
    fixed,rate=tariffs(z,active,q)
    ra,rq,util=response(z,fixed,rate)
    # Inactive clients receive zero tariff; response() may mark zero-utility
    # inactive entries active under its weak IR tie rule, so validate only the
    # intended active coordinates and q=0 outside them.
    if not np.allclose(rq[active],q[active],atol=2e-4,rtol=1e-8) or np.any(q[~active]>1e-8):
        raise AssertionError('follower response mismatch on active coordinates')
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
    if note: out.update(heuristic=True,heuristic_note=note)
    return out


def solve_subset(z,active,restarts=2,seed=2026,maxiter=180,tol=1e-10):
    active=np.asarray(active,dtype=bool)
    if not active.any(): return None
    if z['base_cost'][active].sum()>z['budget']+1e-12: return None
    qhigh=budget_implied_upper(z,active); xhigh=np.divide(qhigh,z['n'],out=np.zeros_like(qhigh),where=z['n']>0)
    bounds=[(0.0,float(xhigh[k])) if active[k] else (0.0,0.0) for k in range(len(active))]
    fun=lambda x: objective(z,active,np.asarray(x)*z['n'])
    jac=lambda x: objective_gradient(z,active,np.asarray(x)*z['n'])*z['n']
    con=lambda x: float(z['budget']-client_cost(z,np.asarray(x)*z['n'],active).sum())
    cjac=lambda x: -(z['c1']+z['c2']*(np.asarray(x)*z['n']))*z['n']*active
    rem=max(0.0,z['budget']-z['base_cost'][active].sum())
    starts=[np.zeros(len(active))]
    if rem>0:
        rng=np.random.default_rng(seed+int(np.flatnonzero(active).sum()))
        ids=np.flatnonzero(active)
        for frac in (.35,.75,.98):
            spend=np.zeros(len(active)); spend[ids]=frac*rem/len(ids)
            q=np.zeros(len(active))
            for k in ids:
                c1=float(z['c1'][k]); c2=float(z['c2'][k]); v=spend[k]
                q[k]=(-c1+np.sqrt(c1*c1+2*c2*v))/c2
            starts.append(q/z['n'])
        for _ in range(restarts):
            shares=rng.dirichlet(np.full(len(ids),.8)); spend=np.zeros(len(active)); spend[ids]=rem*rng.uniform(.4,1.0)*shares
            q=np.zeros(len(active))
            for k in ids:
                c1=float(z['c1'][k]); c2=float(z['c2'][k]); v=spend[k]
                q[k]=(-c1+np.sqrt(c1*c1+2*c2*v))/c2
            starts.append(q/z['n'])
    best=None
    for x0 in starts:
        res=minimize(fun,x0,jac=jac,method='SLSQP',bounds=bounds,
                     constraints=[dict(type='ineq',fun=con,jac=cjac)],
                     options=dict(ftol=tol,maxiter=maxiter))
        x=np.minimum(np.maximum(res.x,0),xhigh); q=x*z['n']; q[~active]=0
        if client_cost(z,q,active).sum()>z['budget']+1e-7:
            lo,hi=0.,1.
            for _ in range(60):
                mid=(lo+hi)/2
                if client_cost(z,mid*q,active).sum()<=z['budget']: lo=mid
                else: hi=mid
            q*=lo
        cand=(objective(z,active,q),q,bool(res.success))
        if best is None or cand[0]<best[0]: best=cand
    return dict(value=float(best[0]),active=active.copy(),q=best[1].copy(),success=best[2])


def solve_joint_heuristic(z,restarts=8,seed=2026,max_rounds=20,subset_restarts=1,maxiter=140):
    """Joint heuristic over participant set x and absolute AIGC quantities q."""
    rng=np.random.default_rng(seed); K=len(z['n']); base=z['base_cost']
    starts=[]
    # All-client start when affordable, plus deterministic size ladder.
    all_active=np.ones(K,dtype=bool)
    if base.sum()<=z['budget']+1e-12: starts.append(all_active)
    order=np.argsort(base)
    for frac in (.2,.4,.6,.8,1.0):
        a=np.zeros(K,dtype=bool)
        for k in order[:max(1,int(round(frac*K)))]:
            a[k]=True
            if base[a].sum()>z['budget']+1e-12: a[k]=False
        if a.any(): starts.append(a)
    for _ in range(restarts):
        a=rng.random(K)<rng.uniform(.15,.9)
        if not a.any(): a[rng.integers(K)]=True
        while base[a].sum()>z['budget']+1e-12 and a.any():
            ids=np.flatnonzero(a); a[rng.choice(ids)]=False
        if a.any(): starts.append(a)

    best=None
    for si,a0 in enumerate(starts):
        cur=solve_subset(z,a0,subset_restarts,seed+1000*si,maxiter)
        if cur is None: continue
        for rr in range(max_rounds):
            incumbent=cur; candidates=[]
            on=np.flatnonzero(cur['active']); off=np.flatnonzero(~cur['active'])
            # Add/drop/swap neighborhood. Equal-size clients make exhaustive
            # single additions/drops cheap enough at K=30.
            for j in off:
                a=cur['active'].copy(); a[j]=True
                if base[a].sum()<=z['budget']+1e-12: candidates.append(a)
            for i in on:
                a=cur['active'].copy(); a[i]=False
                if a.any(): candidates.append(a)
            if len(on) and len(off):
                for i in rng.permutation(on)[:min(8,len(on))]:
                    for j in rng.permutation(off)[:min(8,len(off))]:
                        a=cur['active'].copy(); a[i]=False; a[j]=True; candidates.append(a)
            seen=set()
            for ci,a in enumerate(candidates):
                key=a.tobytes()
                if key in seen: continue
                seen.add(key)
                trial=solve_subset(z,a,subset_restarts,seed+100000+rr*997+ci,maxiter)
                if trial is not None and trial['value']<incumbent['value']-1e-9:
                    incumbent=trial
            if incumbent is cur: break
            cur=incumbent
        if best is None or cur['value']<best['value']: best=cur
    if best is None: return dict(status='infeasible',budget=float(z['budget']))
    note=('joint participant selection and absolute-q allocation; original global weights d_k are not '
          'renormalized over the active set; multi-start add/drop/swap heuristic, no global certificate')
    out=package(z,best['active'],best['q'],note)
    out.update(solver='joint add/drop/swap + fixed-subset multi-start SLSQP',local_solver_success=bool(best['success']))
    return out


def solve_three_state_heuristic(z,budget=None,restarts=20,seed=2026,max_rounds=40):
    """Three-state subset of the joint model: inactive / raw / repair."""
    rng=np.random.default_rng(seed); K=len(z['n'])
    original_budget=float(z['budget'])
    if budget is not None: z['budget']=float(budget)
    try:
        repair=z['repair_q']; base=z['base_cost']
        def decode(state):
            state=np.asarray(state,dtype=np.int8); active=state>0; q=np.where(state==2,repair,0.0)
            return active,q
        def evaluate(state):
            active,q=decode(state)
            if not active.any(): return None
            pay=client_cost(z,q,active).sum()
            if pay>z['budget']+1e-12: return None
            return float(objective(z,active,q)),float(pay),active,q

        starts=[]
        if base.sum()<=z['budget']+1e-12: starts.append(np.ones(K,dtype=np.int8))
        # Greedy raw set and random feasible state vectors.
        s=np.zeros(K,dtype=np.int8)
        for k in np.argsort(base):
            s[k]=1
            if evaluate(s) is None: s[k]=0
        if np.any(s): starts.append(s.copy())
        for _ in range(restarts):
            s=rng.integers(0,3,size=K,dtype=np.int8)
            while True:
                ev=evaluate(s)
                if ev is not None: break
                nz=np.flatnonzero(s>0)
                if not len(nz): s[rng.integers(K)]=1; continue
                j=int(rng.choice(nz)); s[j]=max(0,s[j]-1)
            starts.append(s.copy())

        best=None
        for s0 in starts:
            cur=s0.copy(); ev=evaluate(cur)
            if ev is None: continue
            for _ in range(max_rounds):
                incumbent=(ev[0],cur.copy(),ev)
                # Exhaustive one-client state changes plus random pair swaps.
                for k in range(K):
                    for v in (0,1,2):
                        if v==cur[k]: continue
                        t=cur.copy(); t[k]=v; tev=evaluate(t)
                        if tev is not None and tev[0]<incumbent[0]-1e-9: incumbent=(tev[0],t,tev)
                for _ in range(80):
                    i,j=rng.choice(K,size=2,replace=False)
                    t=cur.copy(); t[i],t[j]=t[j],t[i]; tev=evaluate(t)
                    if tev is not None and tev[0]<incumbent[0]-1e-9: incumbent=(tev[0],t,tev)
                if np.array_equal(incumbent[1],cur): break
                cur=incumbent[1]; ev=incumbent[2]
            if best is None or ev[0]<best[0]: best=(ev[0],cur.copy(),ev)
        if best is None: return dict(status='infeasible',budget=float(z['budget']))
        _,state,ev=best; _,pay,active,q=ev
        out=package(z,active,q,'three-state inactive/raw/repair local-search heuristic under the same fixed-mass model')
        out.update(state=state.tolist(),solver='three-state multi-start single-flip/swap heuristic',matched_budget=(budget is not None))
        return out
    finally:
        z['budget']=original_budget
