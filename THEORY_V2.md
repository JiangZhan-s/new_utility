# THEORY V2 — finite-data-aware Stackelberg pricing for additive AIGC

This document is a non-production theory revision. It does **not** yet replace
`FINAL_GUIDE.md` or the current solver. The purpose is to repair the structural
failure in the old certificate: under the idealized relation
`u_k=1 => F_k=F`, one fully enhanced client could drive the label-bias and
heterogeneity terms to zero, so the server had almost no theoretical reason to
buy additional real data.

The revision keeps the leader-follower game, keeps personalized pricing, and
does not add an artificial reward/penalty for the number of clients. The new
terms are derived from (i) the exact sample count required by addition-only
label repair, (ii) AIGC class-conditional mismatch, and (iii) finite-sample
gradient uncertainty.

## 1. Target distribution and exact addition-only geometry

Let client k have original size `n_k` and label distribution `p_k`. Let the
reference label distribution be

\[
p_\star=\sum_k d_k p_k,\qquad d_k=n_k/\sum_j n_j.
\]

Use normalized enhancement `u_k in [0,1]` and target

\[
p_k^{(u)}=(1-u_k)p_k+u_kp_\star. \tag{1}
\]

Original samples are never deleted. In the continuous-count relaxation, the
smallest final dataset size capable of realizing (1) is

\[
N_k(u)=n_k\max_{y:p_k(y)>0}
\frac{p_k(y)}{(1-u)p_k(y)+u p_\star(y)}. \tag{2}
\]

Define

\[
R_k=\max_{y:p_k(y)>0}\frac{p_k(y)}{p_\star(y)},\qquad
\gamma_k=1-\frac{1}{R_k}. \tag{3}
\]

If `p_k=p_*`, then `R_k=1`, `gamma_k=0`, and no enhancement is needed.
Otherwise `0<gamma_k<1`, and (2) simplifies exactly to

\[
\boxed{N_k(u)=\frac{n_k}{1-\gamma_k u}},\qquad
\boxed{m_k(u)=N_k(u)-n_k=\frac{n_k\gamma_k u}{1-\gamma_k u}}. \tag{4}
\]

Hence the synthetic fraction is

\[
\boxed{\frac{m_k(u)}{N_k(u)}=\gamma_k u}. \tag{5}
\]

The function `m_k(u)` is increasing and strictly convex whenever
`gamma_k>0`:

\[
m_k'(u)=\frac{n_k\gamma_k}{(1-\gamma_k u)^2},\qquad
m_k''(u)=\frac{2n_k\gamma_k^2}{(1-\gamma_k u)^3}>0. \tag{6}
\]

Thus deep enhancement is naturally increasingly expensive even before any
ad-hoc quadratic effort term is introduced.

Integer target counts used in experiments are a rounded implementation of
(4); their rounding error should be recorded separately rather than silently
absorbed into the theory.

## 2. Population gradient decomposition with imperfect AIGC

Assume real clients differ primarily through label proportions, with shared
real class-conditionals. Define

\[
g_y(w)=\mathbb E_{P(X\mid Y=y)}[\nabla\ell(w;X,y)],\qquad
M(w)=[g_1(w),\ldots,g_C(w)]. \tag{7}
\]

Let the class-conditional AIGC distribution be `Q_y` and define

\[
\widetilde g_y(w)=\mathbb E_{Q_y}[\nabla\ell(w;X,y)],\qquad
\delta_y(w)=\widetilde g_y(w)-g_y(w). \tag{8}
\]

Assume a verifiable generator mismatch bound

\[
\sup_{w\in\Omega,y}\|\delta_y(w)\|\le \delta_A. \tag{9}
\]

Let `omega^A_{ky}(u)=m_{ky}(u)/N_k(u)` be the fraction of final client data
that is synthetic class y. The population augmented gradient satisfies

\[
\nabla F_k^{\rm pop}(w;u)-\nabla F(w)
=(1-u)M(w)e_k+g^A_k(w;u), \tag{10}
\]

where `e_k=p_k-p_*` and

\[
g^A_k(w;u)=\sum_y\omega^A_{ky}(u)\delta_y(w). \tag{11}
\]

By (5) and (9),

\[
\boxed{\|g^A_k(w;u)\|\le \gamma_k u\,\delta_A}. \tag{12}
\]

This corrects the old implication `u=1 => F_k=F`. Under V2, full label repair
only eliminates the label-skew component. A finite generator mismatch remains
unless `delta_A=0`.

## 3. Finite-data gradient uncertainty

Let `\widehat F_k(w;u)` be the empirical objective on the finite augmented
set of size `N_k(u)`, and define

\[
\zeta_k(w;u)=\nabla\widehat F_k(w;u)-\nabla F_k^{\rm pop}(w;u). \tag{13}
\]

For theorem use, assume independent client datasets and a uniform gradient
concentration event over the common trajectory region `Omega`. With a common
per-sample gradient-variance proxy `psi^2`, write the weighted concentration
bounds as

\[
\sup_{w\in\Omega}\left\|\sum_{k\in\mathcal P}a_k\zeta_k(w;u_k)\right\|^2
\le S_B(\mathcal P,u), \tag{14}
\]

\[
\sup_{w\in\Omega}\sum_{k\in\mathcal P}a_k
\|\zeta_k-\bar\zeta\|^2
\le S_H(\mathcal P,u), \tag{15}
\]

where a standard weighted empirical-process/concentration calculation gives
proxies of the form

\[
\boxed{S_B=\kappa_\Omega\psi^2
\sum_k\frac{a_k^2}{N_k(u_k)}} ,\qquad
\boxed{S_H=\kappa_\Omega\psi^2
\sum_k\frac{a_k(1-a_k)}{N_k(u_k)}}. \tag{16}
\]

`kappa_Omega` is a complexity/confidence factor determined by the hypothesis
class, trajectory region, and desired confidence; it is not a tuning reward
for client count. For the theorem-matched bounded softmax model it can be made
explicit by an epsilon-net/vector-concentration argument.

Using (4),

\[
\frac{1}{N_k(u_k)}=\frac{1-\gamma_k u_k}{n_k}. \tag{17}
\]

Therefore, for a fixed participant set, both finite-data terms are affine in
`u`:

\[
S_B=\kappa_\Omega\psi^2\sum_k
\frac{a_k^2}{n_k}(1-\gamma_k u_k), \tag{18}
\]

\[
S_H=\kappa_\Omega\psi^2\sum_k
\frac{a_k(1-a_k)}{n_k}(1-\gamma_k u_k). \tag{19}
\]

At `u=0`, since `a_k=n_k/N_P`,

\[
\boxed{S_B(\mathcal P,0)=\frac{\kappa_\Omega\psi^2}{N_\mathcal P}},
\qquad N_\mathcal P=\sum_{k\in\mathcal P}n_k. \tag{20}
\]

This is the missing theoretical value of acquiring more base real data.

Mini-batch SGD noise is distinct from (13). Keep the original Theorem-1 terms
`V_loc` and `V_agg` for stochastic optimization noise; with fixed batch size,
those terms need not decrease with total dataset size.

## 4. Revised bias and heterogeneity certificates

Retain

\[
B_{\rm label}=\left\|\sum_k a_k(1-u_k)e_k\right\|^2, \tag{21}
\]

\[
V_{\rm label}=\sum_k a_k(1-u_k)^2\|e_k\|^2-B_{\rm label}. \tag{22}
\]

Assume `sup_w ||M(w)||_op <= G_cls`.

For generator mismatch define the conservative aggregate term

\[
B_A=\delta_A^2\left(\sum_k a_k\gamma_k u_k\right)^2, \tag{23}
\]

and the conservative within-client mismatch term

\[
V_A=\frac{\delta_A^2}{2}
\sum_{i\ne j}a_i a_j(\gamma_i u_i+\gamma_j u_j)^2. \tag{24}
\]

The pairwise form in (24) is deliberate: with only one participant, the true
within-participant heterogeneity is zero, so the bound should also be zero.

Using `||x+y+z||^2 <= 3(||x||^2+||y||^2+||z||^2)`, on the concentration event
(14)-(15),

\[
\boxed{\mathcal B^2\le
3\left(G_{\rm cls}^2 B_{\rm label}+B_A+S_B\right)}, \tag{25}
\]

\[
\boxed{\mathcal H^2\le
3\left(G_{\rm cls}^2 V_{\rm label}+V_A+S_H\right)}. \tag{26}
\]

The factor 3 is proof slack, not a free hyperparameter.

## 5. Theorem 1+ — finite-data-aware accuracy certificate

Keep the original Theorem-1 learning assumptions and coefficients
`C_0,C_B,C_H,C_L,C_A`. Under (9) and the finite-data concentration event
(14)-(16),

\[
\begin{aligned}
\mathbb E[F(w_T)-F^\star]\le{}&C_0
+3C_B\left(G_{\rm cls}^2B_{\rm label}+B_A+S_B\right)\\
&+3C_H\left(G_{\rm cls}^2V_{\rm label}+V_A+S_H\right)
+C_LV_{\rm loc}+C_AV_{\rm agg}. \tag{27}
\end{aligned}
\]

Hence define the profile-dependent server risk proxy

\[
\boxed{
J^+(\mathcal P,u)=
3C_B(G_{\rm cls}^2B_{\rm label}+B_A+S_B)
+3C_H(G_{\rm cls}^2V_{\rm label}+V_A+S_H)
+C_LV_{\rm loc}+C_AV_{\rm agg}.} \tag{28}
\]

The natural-log cross-entropy bridge remains

\[
\boxed{\mathbb E[\mathrm{Acc}_Q(w_T)]\ge
\left[1-\frac{F^\star+\Gamma+C_0+J^+}{\ln2}\right]_+.} \tag{29}
\]

Thus maximizing server learning utility `U_s=-J^+` still has the same formal
accuracy-certificate interpretation as before, but now base-data quantity and
AIGC quality/quantity both enter the certificate.

### Corollary: one full client is no longer automatically perfect

For `P={k}` and `u_k=1`,

\[
B_{\rm label}=V_{\rm label}=V_A=S_H=0,
\]

but

\[
B_A=\delta_A^2\gamma_k^2,\qquad
S_B=\frac{\kappa_\Omega\psi^2}{N_k(1)}.
\]

Therefore

\[
\boxed{J^+_{\rm one-full}\ge
3C_B\left(\delta_A^2\gamma_k^2+
\frac{\kappa_\Omega\psi^2}{N_k(1)}\right)} \tag{30}
\]

apart from stochastic-optimization terms. Full enhancement is still allowed
to be optimal when the client is genuinely large, cheap, and its AIGC is
extremely accurate; the theory no longer forces or forbids that outcome.

### Corollary: why more unbiased clients can improve the bound

For P equal-size statistically identical unbiased participants, each with
augmented size `N_A`, `delta_A=0`, and equal aggregation weights, the finite
sample part of (28) is

\[
J_{\rm stat}(P)=\frac{3\kappa_\Omega\psi^2}{N_A}
\left[C_H+\frac{C_B-C_H}{P}\right]. \tag{31}
\]

Hence if

\[
\boxed{C_B>C_H}, \tag{32}
\]

then `J_stat(P)` strictly decreases as the number of independent useful
participants increases. This is a consequence of the risk bound, not an
artificial client-count reward.

## 6. Count-aware client cost

The old quadratic enhancement cost can be replaced by a cost tied to the
actual addition-only workload. Let

\[
C_{k,0}=S_k+c_{k,R}n_k \tag{33}
\]

be the task-level fixed/base-data cost and let `kappa_k>0` be the incremental
cost per synthetic sample, including generation and its additional local
training. Define

\[
\boxed{C_k(u)=C_{k,0}+\kappa_k m_k(u)
=C_{k,0}+\frac{\kappa_k n_k\gamma_k u}{1-\gamma_k u}.} \tag{34}
\]

For `gamma_k>0`, this is increasing and strictly convex.

This cost is directly linked to the experiment: deeper label repair requires
more generated samples and, under local epochs, more local computation.

## 7. Genuine Stackelberg game

The server remains the leader. It posts personalized prices

\[
(p_k,r_k), \tag{35}
\]

where `p_k>=0` is the participation/base-data reward and `r_k>=0` is the reward
per unit normalized enhancement. Client k is the follower and chooses

\[
x_k\in\{0,1\},\qquad u_k\in[0,1]. \tag{36}
\]

Its task-level utility is

\[
\boxed{U_k=x_k\left[p_k+r_ku_k-C_k(u_k)\right].} \tag{37}
\]

Conditional on participation, `p_k` does not distort enhancement choice. The
quality response solves

\[
\max_{0\le u\le1}\;r_ku-C_k(u). \tag{38}
\]

From (34),

\[
C_k'(u)=\frac{\kappa_k n_k\gamma_k}{(1-\gamma_k u)^2}. \tag{39}
\]

Therefore, for `gamma_k>0`,

\[
\boxed{
u_k^*(r_k)=
\begin{cases}
0, & r_k\le \kappa_kn_k\gamma_k,\\
\dfrac{1-\sqrt{\kappa_kn_k\gamma_k/r_k}}{\gamma_k},
& \kappa_kn_k\gamma_k<r_k<
\dfrac{\kappa_kn_k\gamma_k}{(1-\gamma_k)^2},\\
1, & r_k\ge\dfrac{\kappa_kn_k\gamma_k}{(1-\gamma_k)^2}.
\end{cases}} \tag{40}
\]

This is a continuous, monotone follower response derived from the physical
addition-only sample count, not imposed by a quadratic curve.

Participation is

\[
\boxed{x_k^*=1\iff
p_k+r_ku_k^*(r_k)-C_k(u_k^*(r_k))\ge0}, \tag{41}
\]

with zero-utility ties assigned to participation.

## 8. Personalized implementation cost and convex reduced leader problem

Under complete information, to implement an interior target `u`, the smallest
marginal price is

\[
r_k(u)=C_k'(u). \tag{42}
\]

With the economically natural constraint `p_k>=0`, the smallest participation
payment is

\[
p_k^{\min}(u)=[C_k(u)-uC_k'(u)]_+. \tag{43}
\]

Hence minimum total server payment is

\[
\boxed{\pi_k(u)=uC_k'(u)+[C_k(u)-uC_k'(u)]_+
=\max\{C_k(u),uC_k'(u)\}.} \tag{44}
\]

This explicitly keeps incentive rent when a non-negative participation reward
cannot extract it; no signed entry fee is required.

For (34), both `C_k(u)` and `uC_k'(u)` are convex, so `pi_k(u)` is convex.
For a fixed participant set, the complete-information personalized leader
problem is therefore

\[
\boxed{\min_{0\le u_k\le1}J^+(\mathcal P,u)
\quad\text{s.t.}\quad
\sum_{k\in\mathcal P}\pi_k(u_k)\le B.} \tag{45}
\]

Every term of `J^+` in (28) is a convex quadratic or affine function of `u`
for fixed `P`: `B_label` and `V_label` are PSD quadratics, `B_A` and `V_A` are
sums of squares of linear forms, and `S_B,S_H` are affine by (17). Thus (45)
is a convex optimization problem for each fixed participant set.

Participation selection remains discrete. The full problem is mixed-discrete
convex, not globally convex merely because the fixed-P subproblem is convex.
For K=20/50 a scalable branch-and-bound, mixed-integer convex, or decomposition
solver is required instead of exhaustive mode enumeration.

## 9. Main pricing theorem: personalized versus uniform pricing

Let the uniform-price benchmark restrict

\[
p_k=p,\qquad r_k=r,\qquad \forall k. \tag{46}
\]

The personalized mechanism permits arbitrary client-specific `(p_k,r_k)`.
Every outcome induced by a uniform pair `(p,r)` is therefore feasible under
personalized pricing by setting all personalized prices equal to that pair.
Consequently, under the same hard budget B and the same follower model,

\[
\boxed{\mathcal F_{\rm uniform}(B)\subseteq
\mathcal F_{\rm personalized}(B)}, \tag{47}
\]

and hence

\[
\boxed{J^*_{\rm personalized}(B)\le
J^*_{\rm uniform}(B)}. \tag{48}
\]

Combining (48) with (29) gives a weakly higher theoretical accuracy lower bound
for the optimal personalized mechanism wherever the certificate is unclipped.

This is the proposed primary theorem-level economic claim. It does **not**
claim that personalized pricing must recruit more clients or synthesize more
samples in every instance. Those are endogenous outcomes. If one genuinely
large, cheap client with near-perfect AIGC is statistically sufficient, a
concentrated optimum is economically and statistically legitimate.

## 10. What V2 fixes and what remains open

V2 fixes four structural issues:

1. full label repair no longer implies zero total learning uncertainty;
2. acquiring more independent base data reduces the finite-data term naturally;
3. actual synthetic sample count enters both learning and client cost;
4. client enhancement remains an endogenous follower decision with a closed-form
   Stackelberg response.

V2 intentionally does not claim unconditional continuous-over-discrete
superiority. A discrete endpoint follower and a continuous follower have
different implementation thresholds under posted prices, so such a theorem
requires a carefully matched benchmark.

One important mismatch remains: the practical CNN experiment currently uses
complete local epochs, so the number of optimizer steps depends on `N_k(u)`,
while the existing Theorem 1 assumes a common fixed h. Either a variable-local-
step convergence theorem must be derived, or theorem-matched experiments must
use the fixed-h protocol. This issue is separate from the finite-data repair
above and should not be hidden in the final paper.

## 11. Relation to prior work

IMFL-AIGC (IEEE TMC 2024, arXiv:2406.08526) explicitly models client data
quantity, non-IID quality, AIGC quality, data-computing cost, and a stochastic
gradient error term that decreases with local batch/data size. Its AIGC model
keeps client dataset size fixed, whereas V2 above retains this project's
addition-only semantics and derives the exact expansion function (4).

The purpose of citing that work is methodological: learning quantity/quality
must enter the convergence analysis before designing the incentive mechanism.
The equations in V2, especially (4), (10)-(12), (18)-(19), and the resulting
Stackelberg response (40), are specific to this project's addition-only model.
