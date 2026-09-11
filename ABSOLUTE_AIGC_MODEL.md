# Absolute-count AIGC procurement prototype

This path is an experimental alternative to the legacy `u in [0,1]` enhancement model. It is designed to test whether endpoint behavior was partly caused by unequal client sizes and by treating “full repair” as a hard decision endpoint.

## Data construction

For CIFAR-10, 10% of the 50,000 training images are reserved as validation data. The remaining 45,000 real images are partitioned across 30 clients with fixed size

\[
n_k=1500,\qquad k=1,\ldots,30.
\]

Dirichlet draws control only class composition. Iterative proportional fitting reconciles those non-IID preferences with exact client-size and class-total margins. Thus the experiment removes client-size heterogeneity while retaining label skew.

## Absolute AIGC decision

The server chooses

\[
q_k\ge 0,
\]

the absolute number of AIGC samples purchased for client `k`. There is no constraint of the form `q_k/n_k <= rho`, no `u_k <= 1`, and no requirement `q_k <= q_k^repair`. The only quantity restriction is the monetary budget.

For fixed `q_k`, generated class counts are chosen by the addition-only projection

\[
\min_{f_k}\left\|\frac{f_k}{n_k+q_k}-p_*\right\|_2^2,
\quad f_k\ge n_k^{class},\quad \mathbf 1^T f_k=n_k+q_k.
\]

Writing `d_y=(n_k+q_k)p_*(y)-n_{ky}`, the optimal additions have the simplex-projection form

\[
a_y(q_k)=\max\{d_y-\tau,0\},\qquad \sum_y a_y(q_k)=q_k.
\]

Once exact label repair becomes feasible, further AIGC samples follow `p_*` and preserve zero label skew. They remain allowed; they are not clipped.

## Generator mismatch

The AIGC residual uses the actual synthetic exposure

\[
s_k(q_k)=\frac{q_k}{n_k+q_k},
\]

so excessive AIGC remains visible to the generator-aware term even after label skew is fully repaired. The current CIFAR-10 default freezes the measured trajectory value `chi=0.06180083094278078` for controlled comparison.

## Economic cost and Stackelberg implementation

The complete-information client cost is

\[
C_k(q)=c^R_k+\frac{c_A}{N}q+\frac{c_Q}{2Nn_k}q^2.
\]

The weak `c_Q>0` term represents increasing marginal generation/effort cost and gives a unique continuous follower response. It is not a quantity cap. Under the signed two-part tariff,

\[
r_k=C_k'(q_k),\qquad b_k=c^R_k-\frac{c_Q}{2Nn_k}q_k^2,
\]

so the implemented payment equals true client cost. The signed-fixed-transfer caveat from the legacy mechanism therefore still applies.

The sole resource constraint is

\[
\sum_k C_k(q_k)\le B.
\]

`q_k^repair` is computed only to define a reproducible reference budget and an unconstrained label-repair baseline. It is never used as a feasible-set upper bound.

## 30-client heuristic

All 30 clients are fixed active in this prototype. This deliberately removes participant-set renormalization jumps and isolates AIGC quantity allocation. The personalized allocation is obtained by multi-start local SLSQP using a piecewise analytic gradient of the simplex projection. It is a heuristic: no global-optimality certificate is claimed.

For numerical conditioning only, the solver internally uses `x_k=q_k/n_k`; `x_k` has no exogenous cap. Its finite numerical bound is exactly the monetary-budget-implied upper bound transformed into this coordinate. Contracts and experiments report and implement absolute `q_k` counts.

Main baselines are: personalized absolute `q_k`, uniform absolute `q` under the same budget, no AIGC, and all-client exact label repair as a generally over-budget reference. All use fixed original-data local compute, so downstream ACC comparisons have identical gradient-step counts.
