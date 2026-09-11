"""Equal-mass client partitions and absolute-count AIGC augmentation.

This experimental path deliberately avoids an enhancement-ratio decision
variable.  The economic decision is q_k >= 0, the absolute number of synthetic
samples purchased for client k.  Any finite numerical upper bound used by a
solver must be implied by the monetary budget, never imposed as a synthetic/
real ratio cap.
"""
import numpy as np


def split_train_fixed_size(labels, classes, clients=30, alpha=.3, seed=2026,
                           val_fraction=.1, max_iter=10000, tol=1e-8):
    """Dirichlet non-IID split with equal (or differ-by-one) client sizes.

    Class totals are preserved exactly and every retained training example is
    assigned exactly once.  Dirichlet draws specify row-wise class preferences;
    iterative proportional fitting reconciles those preferences with both the
    fixed client-size margins and the available class-count margins.
    """
    if clients < 1 or classes < 1 or alpha <= 0 or not (0 <= val_fraction < 1):
        raise ValueError('invalid split controls')
    rng=np.random.default_rng(seed)
    labels=np.asarray(labels)
    train_by_class=[]; valid=[]
    for c in range(classes):
        ids=rng.permutation(np.flatnonzero(labels==c))
        if len(ids)==0: raise ValueError(f'class {c} has no examples')
        v=max(1,int(len(ids)*val_fraction)) if val_fraction>0 else 0
        valid.extend(ids[:v]); train_by_class.append(ids[v:].astype(np.int64))
    col=np.array([len(v) for v in train_by_class],dtype=np.int64)
    total=int(col.sum())
    if total < clients:
        raise ValueError('fewer retained training examples than clients')
    row=np.full(clients,total//clients,dtype=np.int64)
    row[:total%clients]+=1

    # Positive preference matrix; IPFP preserves non-IID row preferences while
    # enforcing the exact continuous row/column margins.
    pref=rng.dirichlet(np.full(classes,float(alpha)),size=clients)
    x=np.maximum(pref,1e-15)
    for _ in range(max_iter):
        x*=row[:,None]/x.sum(1)[:,None]
        x*=col[None,:]/x.sum(0)[None,:]
        err=max(float(np.max(np.abs(x.sum(1)-row))),
                float(np.max(np.abs(x.sum(0)-col))))
        if err <= tol: break
    else:
        raise RuntimeError('fixed-size IPFP did not converge')

    # Integerize while keeping both margins exact. Floors leave small degree
    # deficits; fill one count at a time, favoring large fractional remainders
    # and columns with the most remaining capacity.
    counts=np.floor(x+1e-12).astype(np.int64)
    rdef=row-counts.sum(1); cdef=col-counts.sum(0)
    frac=x-np.floor(x)
    while int(rdef.sum())>0:
        rows=np.flatnonzero(rdef>0); cols=np.flatnonzero(cdef>0)
        if not len(rows) or not len(cols):
            raise AssertionError('integerization margin mismatch')
        r=int(rows[np.argmax(rdef[rows])])
        score=frac[r,cols] + 1e-9*cdef[cols]
        c=int(cols[np.argmax(score)])
        counts[r,c]+=1; rdef[r]-=1; cdef[c]-=1
    if np.any(rdef) or np.any(cdef): raise AssertionError('integerization failed')
    if not np.array_equal(counts.sum(1),row) or not np.array_equal(counts.sum(0),col):
        raise AssertionError('fixed-size margins not preserved')

    parts=[[] for _ in range(clients)]
    for c,ids in enumerate(train_by_class):
        ids=rng.permutation(ids); start=0
        for k in range(clients):
            take=int(counts[k,c])
            if take:
                parts[k].extend(ids[start:start+take].tolist())
            start+=take
        if start!=len(ids): raise AssertionError('class assignment mismatch')
    parts=[np.asarray(rng.permutation(p),dtype=np.int64) for p in parts]
    if [len(p) for p in parts] != row.tolist(): raise AssertionError('client-size mismatch')
    return parts,np.asarray(valid,dtype=np.int64),counts


def optimal_augmented_distribution(original_counts,pstar,q,atol=1e-12):
    """Best label distribution after adding exactly q synthetic samples.

    The optimal additions solve a simplex projection. With T=n+q and
    d_y=T p_star(y)-n_y, write additions as a_y=max(d_y-tau,0), where
    tau is chosen so sum_y a_y=q. This is O(C log C), permits every q>=0,
    and becomes exact p_star repair once T p_star dominates all originals.
    """
    c=np.asarray(original_counts,dtype=float)
    p=np.asarray(pstar,dtype=float)
    if c.ndim!=1 or p.ndim!=1 or len(c)!=len(p) or np.any(c<0) or c.sum()<=0:
        raise ValueError('invalid original_counts/pstar')
    if np.any(p<0) or not np.isclose(p.sum(),1.,atol=1e-10):
        raise ValueError('pstar must be a probability vector')
    q=float(q)
    if q < -atol or not np.isfinite(q): raise ValueError('q must be finite and nonnegative')
    q=max(q,0.0); n=float(c.sum()); total=n+q
    if q<=atol:
        final=c.copy(); return final/n,final
    d=total*p-c
    order=np.argsort(-d)
    ds=d[order]; css=np.cumsum(ds)
    rho=0; tau=0.0
    for j in range(1,len(ds)+1):
        t=(css[j-1]-q)/j
        if ds[j-1]>t-atol:
            rho=j; tau=t
    if rho==0:
        raise AssertionError('simplex projection found no active coordinate')
    tau=(css[rho-1]-q)/rho
    add=np.maximum(d-tau,0.0)
    diff=q-add.sum()
    if abs(diff)>1e-9:
        add[int(np.argmax(add))]+=diff
    final=c+add
    if np.any(add<-1e-8) or not np.isclose(add.sum(),q,atol=1e-7):
        raise AssertionError('continuous augmentation projection failed')
    return final/total,final


def absolute_target_counts(original_counts,pstar,q):
    """Integer training plan for exactly q purchased synthetic samples."""
    q_int=int(q)
    if q_int<0 or q_int!=q: raise ValueError('q must be a nonnegative integer')
    c=np.asarray(original_counts,dtype=np.int64)
    target_cont,final_cont=optimal_augmented_distribution(c,pstar,q_int)
    add_cont=np.maximum(0.0,final_cont-c)
    add=np.floor(add_cont+1e-12).astype(np.int64)
    remaining=q_int-int(add.sum())
    if remaining<0: raise AssertionError('rounded too many additions')
    if remaining:
        frac=add_cont-add
        order=np.argsort(-frac,kind='stable')
        add[order[:remaining]]+=1
    final=c+add
    if int(add.sum())!=q_int or np.any(final<c): raise AssertionError('invalid integer plan')
    target=final/final.sum()
    return target,final,add


def repair_quantity(original_counts,pstar):
    """Smallest continuous q for which exact p_star is addition-only feasible.

    This is a diagnostic/reference quantity used to normalize budgets, not a
    constraint on q. The feasible domain remains q >= 0 with only the budget.
    """
    c=np.asarray(original_counts,dtype=float); p=np.asarray(pstar,dtype=float)
    n=float(c.sum())
    positive=p>0
    if np.any((~positive)&(c>0)): return float('inf')
    total_needed=float(np.max(c[positive]/p[positive]))
    return max(0.0,total_needed-n)


def optimal_augmented_distribution_with_derivative(original_counts,pstar,q,atol=1e-12):
    """Return optimal augmented distribution and its local derivative d/dq.

    The derivative is valid away from the finitely many active-set breakpoints
    of the simplex projection; at a breakpoint it returns one consistent
    one-sided active-set derivative, which is sufficient for local heuristic
    optimization.
    """
    c=np.asarray(original_counts,dtype=float); p=np.asarray(pstar,dtype=float)
    q=float(q); n=float(c.sum()); T=n+q
    dist,final=optimal_augmented_distribution(c,p,q,atol=atol)
    if q<=atol:
        q_probe=max(1e-7,1e-9*n)
        return_dist,_=optimal_augmented_distribution(c,p,q_probe,atol=atol)
        deriv=(return_dist-dist)/q_probe
        return dist,deriv
    d=T*p-c
    add=final-c
    active=add>max(atol,1e-10)
    if not np.any(active):
        h=max(1e-6,1e-7*T)
        d2,_=optimal_augmented_distribution(c,p,q+h,atol=atol)
        return dist,(d2-dist)/h
    m=int(active.sum()); P=float(p[active].sum())
    tau_prime=(P-1.0)/m
    fprime=np.zeros_like(c)
    fprime[active]=p[active]-tau_prime
    deriv=(fprime*T-final)/(T*T)
    deriv-=deriv.sum()/len(deriv)
    return dist,deriv
