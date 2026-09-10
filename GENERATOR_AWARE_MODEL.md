# Generator-Aware Continuous AIGC Model

This document replaces the *ideal-AIGC-is-perfect-at-full-enhancement* assumption for the new generator-aware experiments. The economic two-part tariff and data-calibrated cost model are unchanged.

## 1. Why the ideal model collapses to endpoints

The old population interpolation was

\[
\nabla F_k(w;u_k)-\nabla F(w)=(1-u_k)\Delta_k(w).
\]

Hence \(u_k=1\) implied zero client bias. For any selected subset whose clients were all fully enhanced, both the label-direction bias and heterogeneity terms vanished. The server therefore frequently preferred a cheap all-full subset to a larger partially enhanced set.

That implication is too strong for finite-quality generators: matching a label marginal does not imply matching the real class-conditional distribution.

## 2. Additive mixture geometry

Let \(p_k\) be client \(k\)'s real label distribution and \(p_\star\) the real population label distribution. The target marginal remains

\[
p_k^{(u)}=(1-u_k)p_k+u_kp_\star.
\]

For addition-only repair, define

\[
R_k=\max_y \frac{p_k(y)}{p_\star(y)},\qquad
\gamma_k=1-\frac1{R_k}.
\]

The continuous augmented size is

\[
N_k(u_k)=\frac{n_k}{1-\gamma_k u_k},
\]

and the number of generated samples is

\[
m_k(u_k)=\frac{n_k\gamma_k u_k}{1-\gamma_k u_k}.
\]

Therefore the synthetic fraction in the resulting real+synthetic mixture is exactly

\[
\boxed{\frac{m_k(u_k)}{N_k(u_k)}=\gamma_k u_k.}
\]

This identity is the bridge between enhancement level and generator-distribution error.

## 3. Generator-aware gradient model

Under the label-direction model,

\[
\Delta_{k,L}(w)=M(w)e_k,\qquad \|M(w)\|_{op}\le G_{\rm cls}.
\]

Let \(Q_y\) denote the generator class-conditional distribution and \(P_y\) the real one. Assume

\[
\left\|\mathbb E_{Q_y}\nabla\ell(w;X,y)-\mathbb E_{P_y}\nabla\ell(w;X,y)\right\|
\le \delta_A
\]

on the theorem trajectory region. Because the synthetic fraction is \(\gamma_k u_k\), write

\[
\boxed{
\Delta_k(w;u_k)=(1-u_k)M(w)e_k+g^A_k(w;u_k),
}
\]

with

\[
\boxed{\|g^A_k(w;u_k)\|\le \gamma_k u_k\delta_A.}
\]

Thus full enhancement removes the *label-marginal* component but does not make the client equal to the true population objective:

\[
u_k=1\quad\Rightarrow\quad
\|\Delta_k(w;1)\|\le \gamma_k\delta_A,
\]

not zero.

## 4. New bias and heterogeneity certificates

For a participant set \(\mathcal P\), let

\[
a_k=\frac{d_k}{\sum_{j\in\mathcal P}d_j},
\qquad
v_L(u)=\sum_{k\in\mathcal P}a_k(1-u_k)e_k.
\]

Define

\[
B_{\rm label}(u)=\|v_L(u)\|^2,
\]

\[
V_{\rm label}(u)=
\sum_k a_k(1-u_k)^2\|e_k\|^2-B_{\rm label}(u).
\]

Using \(\|x+y\|^2\le2\|x\|^2+2\|y\|^2\),

\[
\boxed{
\mathcal B^2
\le
2G_{\rm cls}^2B_{\rm label}
+2\delta_A^2\left(\sum_k a_k\gamma_k u_k\right)^2.
}
\]

For weighted heterogeneity, split the label and generator residuals and use that weighted variance is at most the weighted second moment:

\[
\boxed{
\mathcal H^2
\le
2G_{\rm cls}^2V_{\rm label}
+2\delta_A^2\sum_k a_k\gamma_k^2u_k^2.
}
\]

These are conservative upper bounds. They preserve the existing FedAvg theorem: Theorem 1 is applied with the new upper bounds for \(\mathcal B^2\) and \(\mathcal H^2\).

## 5. Normalized server objective

Let

\[
\chi=\delta_A/G_{\rm cls}.
\]

The normalized profile optimized by the server is

\[
\boxed{
\begin{aligned}
J_{\rm GA}(u)= {}&
2C_BG_{\rm cls}^2\left[
B_{\rm label}(u)
+\chi^2\left(\sum_k a_k\gamma_k u_k\right)^2
\right]\\
&+2C_HG_{\rm cls}^2\left[
V_{\rm label}(u)
+\chi^2\sum_k a_k\gamma_k^2u_k^2
\right]\\
&+C_LV_{\rm loc}+C_AV_{\rm agg}.
\end{aligned}
}
\]

The old ideal-AIGC profile is recovered only as the special case \(\chi=0\) (up to the conservative factor introduced by the split bound).

For fixed participants this objective is a convex quadratic in \(u\):

\[
J_{\rm GA}(u)=(1-u)^TQ_L(1-u)+u^TQ_Au+c,
\]

with \(Q_L\succeq0\) and \(Q_A\succeq0\). Therefore the continuous follower-implementable allocation remains computationally tractable.

A one-dimensional analogue is

\[
f(u)=A(1-u)^2+Du^2,
\]

whose unconstrained minimizer is

\[
\boxed{u^\star=\frac{A}{A+D}\in(0,1)}
\]

whenever \(A,D>0\). Hence an interior optimum is now a statistical consequence of the real/synthetic trade-off rather than an artificial restriction against full enhancement.

## 6. Quality calibration

IMFL-AIGC reports on CIFAR-10

\[
g_{\rm data}=1.75,\qquad g_{\rm diff}=0.54,
\]

and defines

\[
\theta=\frac{g_{\rm diff}}{2g_{\rm data}}\approx0.1543.
\]

Because the present bound uses the generator gradient residual \(\delta_A\) directly relative to the real-gradient scale \(G_{\rm cls}\), its dimensionless transfer quantity is

\[
\boxed{\chi=\frac{g_{\rm diff}}{g_{\rm data}}\approx0.3086=2\theta.}
\]

The repository uses this only as an external CIFAR-10 reference. The preferred empirical setting measures \(g_{\rm data}\) and \(g_{\rm diff}\) on the exact real/synthetic cache at a common reference model and stores the full per-class diagnostic.

This cache-specific estimate is pointwise at the reference model. It is an empirical profile parameter, not by itself a uniform-in-\(w\) theorem certificate.

## 7. Accuracy protocol

A second artifact in the old experiment was that CNN local epochs were run over the enlarged augmented multiset. Full enhancement therefore obtained both a different data mixture and many more SGD updates.

The generator-aware experiment fixes client \(k\)'s local optimization budget to the number of steps corresponding to the *original* dataset:

\[
\boxed{h_k=E\left\lceil\frac{n_k}{b}\right\rceil.}
\]

For every enhancement level, those \(h_k\) batches are sampled from the real+synthetic mixture. Thus \(u=0.3\) and \(u=1\) receive the same number of local updates. A direct `selected_full` ablation uses exactly the same participants as the continuous solution, sets their \(u_k=1\), and keeps this fixed-compute protocol. This is the decisive test of whether the interior mixture has higher test accuracy than full enhancement.

## 8. Literature basis

1. G. Huang, Q. Wu, J. Li, X. Chen, **IMFL-AIGC: Incentive Mechanism Design for Federated Learning Empowered by Artificial Intelligence Generated Content**, 2024, arXiv:2406.08526. It explicitly models nonzero quality mismatch for AIGC-enhanced data and empirically estimates real/generated gradient discrepancy on CIFAR-10.
2. A. Shidani, T. Farghly, Y. Sun, H. Ganjgahi, G. Deligiannidis, **Beyond Real Data: Synthetic Data through the Lens of Regularization**, AISTATS 2026, PMLR 300. It derives an optimal real/synthetic ratio from algorithmic stability and distribution discrepancy and validates U-shaped test error versus synthetic proportion on CIFAR-10 and brain MRI.

These papers support the modeling principle that more synthetic data is not monotonically better when the generator distribution differs from the real target distribution. They do not prove that the present CNN experiment must have its optimum at a particular numerical \(u\); that remains an empirical claim to test on the actual cache.
