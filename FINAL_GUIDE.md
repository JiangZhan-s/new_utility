# BAQP 最终整理版：模型、理论、机制、实现、参数与引用

> 本服务器真实图像实验入口见 [README.md](README.md)，新求解推导核对见 [STACKELBERG_KKT_REVIEW.md](STACKELBERG_KKT_REVIEW.md)。本地新实现为 `baqp/`（NumPy + SciPy + PyTorch）；本文原包的受控结果/10项检查不等于当前三数据集的训练结果。实际运行状态见 [outputs/RUN_STATUS.md](outputs/RUN_STATUS.md)。


本版统一此前讨论，采用“连续增强 + 标签方向 + 可实施价格 + 预算约束”的模型。主效用的系数从指定 FedAvg 协议推导；标签信息不足时显式加入梯度残差。配套程序可以运行，不依赖商用优化器。

**结论范围：** 在声明的学习条件下，服务器优化测试准确率下界对应的风险证书；在完全信息和质量可验证条件下，价格诱导预算可行的客户端最优响应。它不保证连续机制的实际 ACC 必然超过三状态机制。附带的默认数值证书截断后为零，不能用它宣称已认证了正的准确率下界。

## 1. 系统目标与符号

有 \(K\) 个客户端，原始样本数为 \(n_k>0\)，令

\[
N=\sum_kn_k,\qquad d_k=\frac{n_k}{N},\qquad
F(w)=\sum_kd_kF_{k,L}(w).
\tag{1}
\]

参与集合 \(\mathcal P\ne\varnothing\) 在整个 \(T\) 轮任务中固定，使用

\[
D_{\mathcal P}=\sum_{k\in\mathcal P}d_k,\qquad
a_k=\frac{d_k}{D_{\mathcal P}}.
\tag{2}
\]

令 \(p_k\) 为原始标签分布，**参考标签分布始终取**

\[
p_\star=\sum_kd_kp_k,\qquad e_k=p_k-p_\star.
\tag{3}
\]

公共参考数据用于估计类别条件信息，不自行决定 \(p_\star\)。如果测试目标是另一个分布 \(Q\)，保留 (1)，另外处理目标分布偏移；不能悄悄把 \(p_\star\) 改成均匀分布。

成本、预算和报酬统一按完整 \(T\) 轮任务计算。\(d_k\) 为固定归一化权重，合成数据不改变支付权重和服务器聚合权重。

## 2. 完美 AIGC 与连续增强

令 \(\lambda_k\) 是原始梯度偏差的统一上界，满足

\[
\Delta_k(w)=\nabla F_{k,L}(w)-\nabla F(w),\qquad
\|\Delta_k(w)\|\le\lambda_k.
\tag{4}
\]

理想 AIGC 最大增强满足 \(F_{k,A}=F\)，故 \(\theta_k=0\)。本版将中间状态的实现模型明确规定为

\[
0\le q_k\le\lambda_k,\quad u_k^{\rm enh}=q_k/\lambda_k,\quad
\rho_k=1-u_k^{\rm enh},
\tag{5}
\]

\[
F_k(w;q_k)=\rho_kF_{k,L}(w)+(1-\rho_k)F(w).
\tag{6}
\]

因此

\[
\nabla F_k(w;q_k)-\nabla F(w)=\rho_k\Delta_k(w).
\tag{7}
\]

这里 \(u_k^{\rm enh}\) 只表示增强比例；客户端经济效用用 \(U_k\) 表示，避免符号重名。\(\lambda_k=0\) 的客户端无需增强，直接限制 \(q_k=0\)。程序对其单独处理。

有限合成样本通常不能使经验目标处处等于 \(F\)。式 (6) 是理想混合目标，可由抽样 oracle 或损失重加权精确实现；普通生成器是这一接口的近似实现，必须核验其误差。

## 3. 客户端效用与最优响应

固定 \(\bar\lambda\ge\max_k\lambda_k\)，定义

\[
c_k=1-\lambda_k/\bar\lambda,\qquad
\phi_k(q_k)=c_k+q_k/\bar\lambda.
\tag{8}
\]

服务器给出个性化单价 \(r_k\ge0\)。参加时的报酬、成本、净效用分别为

\[
R_k(q_k;r_k)=d_kr_k\phi_k(q_k),
\tag{9}
\]

\[
C_k(q_k)=d_k(s_k+\alpha_kq_k+\beta_kq_k^2),
\qquad s_k,\alpha_k,\beta_k>0,
\tag{10}
\]

\[
\boxed{U_k(q_k;r_k)=d_k\left[
r_k\left(c_k+\frac{q_k}{\bar\lambda}\right)
-s_k-\alpha_kq_k-\beta_kq_k^2\right].}
\tag{11}
\]

不参加的效用为零。条件于参加，严格凹性给出

\[
\widehat q_k(r_k)=\left[
\frac{r_k/\bar\lambda-\alpha_k}{2\beta_k}
\right]_0^{\lambda_k}.
\tag{12}
\]

完整响应是

\[
(x_k^*,q_k^*)=
\begin{cases}
(1,\widehat q_k),&U_k(\widehat q_k;r_k)\ge0,\\
(0,0),&U_k(\widehat q_k;r_k)<0.
\end{cases}
\tag{13}
\]

在净效用等于零时约定参加。此为完全信息模型；不包含私有类型真实申报的证明。质量验证也作为合同前提，未在本版设计抗欺骗审计协议。

## 4. 学习误差与 Theorem 1

定义实际参与者目标 \(G=\sum_{k\in\mathcal P}a_kF_k(\cdot;q_k)\)，以及

\[
\mathcal B^2=\sup_{w\in\Omega}\|\nabla G(w)-\nabla F(w)\|^2,
\tag{14}
\]

\[
\mathcal H^2=\sup_{w\in\Omega}\sum_ka_k
\|\nabla F_k(w;q_k)-\nabla G(w)\|^2.
\tag{15}
\]

每步随机梯度满足条件无偏和条件方差上界 \(v_k\)，不同客户端的同一步噪声条件不相关。定义

\[
V_{\rm loc}=\sum_ka_kv_k,\qquad V_{\rm agg}=\sum_ka_k^2v_k.
\tag{16}
\]

### Theorem 1：连续增强下的准确率下界

假设：

1. 各 \(F_k(\cdot;q_k)\) 凸、二阶可微、\(L\)-smooth；\(F\) 满足 \(\mu I\preceq\nabla^2F\preceq LI\)，有无约束最优点 \(w^\star\)。这些常数对全部可行配置共同有效。
2. FedAvg 每轮执行相同 \(h\) 步本地 SGD，共 \(T\) 轮；固定步长 \(0<\eta<1/L\)，没有动量、Adam 或投影。
3. 上述噪声条件成立；所有实际和证明所需对照迭代及最优点位于共同凸区域 \(\Omega\)，误差界与光滑性在其上统一成立。
4. 目标测试交叉熵有统一风险桥接 \(R_Q^{\rm CE}(w)\le F(w)+\Gamma\)。

令

\[
\tau=1-\mu\eta,\quad A_T=1-\tau^{Th},\quad
\omega_j=\frac{\sum_{e=0}^{h-1}\tau^{h-1-e}e^j}
{\sum_{e=0}^{h-1}\tau^{h-1-e}},\ j=1,2.
\tag{17}
\]

定义

\[
C_0=\frac L2\tau^{Th}\|w_0-w^\star\|^2,\qquad
C_B=\frac{LA_T}{\mu^2},
\tag{18}
\]

\[
C_H=\frac{2L^3\eta^2A_T}{\mu^2}\omega_2,\quad
C_L=\frac{2L^3\eta^2A_T}{\mu^2}\omega_1,\quad
C_A=\frac{L\eta A_T}{2\mu}.
\tag{19}
\]

则最终轮输出满足

\[
\mathbb E[F(w_T)-F^\star]
\le C_0+C_B\mathcal B^2+C_H\mathcal H^2+C_LV_{\rm loc}+C_AV_{\rm agg}.
\tag{20}
\]

对于最大概率分类器和自然对数交叉熵，有

\[
\boxed{\mathbb E[\operatorname{Acc}_Q(w_T)]\ge
\left[1-\frac{F^\star+\Gamma+C_0+
C_B\mathcal B^2+C_H\mathcal H^2+C_LV_{\rm loc}+C_AV_{\rm agg}}{\ln2}\right]_+.}
\tag{21}
\]

这是本版按指定训练协议推导的保守界，不是把已有论文中的大 \(O\) 项改名所得。其异质性与本地更新分析思路可以联系 [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html) 和 [Wang 等人的 Theorem A.9](https://arxiv.org/html/2407.15567v1)，但新系数与新机制的证明责任属于本文。

### 证明

一轮内记真实局部轨迹为 \(w_{k,e}\)，虚拟平均为 \(m_e=\sum_ka_kw_{k,e}\)。另定义无噪声局部轨迹 \(u_{k,e}\) 与对参与者目标 \(G\) 做梯度下降的公共轨迹 \(v_e\)，两者和真实轨迹都从轮首模型出发。

凸光滑函数的梯度更新映射在该步长下非扩张。条件无偏噪声给出

\[
\sum_ka_k\mathbb E\|w_{k,e}-u_{k,e}\|^2\le\eta^2eV_{\rm loc}.
\]

对局部确定性轨迹与公共轨迹比较，加权欧氏范数每步增加至多 \(\eta\mathcal H\)，故

\[
\sum_ka_k\|u_{k,e}-v_e\|^2\le\eta^2e^2\mathcal H^2.
\]

将两式相加并使用平方三角界，得到

\[
\sum_ka_k\mathbb E\|w_{k,e}-v_e\|^2
\le2\eta^2(e^2\mathcal H^2+eV_{\rm loc}).
\tag{22}
\]

令

\[
\delta_e=\sum_ka_k[\nabla F_k(w_{k,e};q_k)-\nabla F_k(m_e;q_k)].
\]

光滑性及加权均值最小化平方距离给出

\[
\mathbb E\|\delta_e\|^2\le2L^2\eta^2(e^2\mathcal H^2+eV_{\rm loc}).
\tag{23}
\]

真实平均更新可写成

\[
m_{e+1}=m_e-\eta\nabla F(m_e)-\eta[b(m_e)+\delta_e]-\eta\bar\xi_e,
\]

其中 \(b=\nabla G-\nabla F\)，\(\mathbb E[\bar\xi_e\mid\mathscr F_e]=0\)，\(\mathbb E\|\bar\xi_e\|^2\le V_{\rm agg}\)。固定目标的更新映射有收缩因子 \(\tau\)。在 Young 不等式中取参数 \((1-\tau)/\tau\)，得到

\[
\begin{aligned}
\mathbb E\|m_{e+1}-w^\star\|^2\le{}&
\tau\mathbb E\|m_e-w^\star\|^2+
\frac{2\eta}{\mu}\mathcal B^2\\
&+\frac{4L^2\eta^3}{\mu}(e^2\mathcal H^2+eV_{\rm loc})+\eta^2V_{\rm agg}.
\end{aligned}
\tag{24}
\]

轮末聚合后重置局部模型，将该递推连接全部 \(Th\) 步。几何权重之和为 \(A_T/(\mu\eta)\)，带 \(e^j\) 的和为 \(A_T\omega_j/(\mu\eta)\)。因此

\[
\begin{aligned}
\mathbb E\|w_T-w^\star\|^2\le{}&\tau^{Th}\|w_0-w^\star\|^2+
\frac{2A_T}{\mu^2}\mathcal B^2\\
&+\frac{4L^2\eta^2A_T}{\mu^2}(\omega_2\mathcal H^2+\omega_1V_{\rm loc})+
\frac{\eta A_T}{\mu}V_{\rm agg}.
\end{aligned}
\]

由 \(F(w)-F^\star\le L\|w-w^\star\|^2/2\) 得 (20)。误分类时 \(p_y\le1/2\)，故逐样本有 \(\mathbf1[\widehat y\ne y]\le-\ln p_y/\ln2\)。取期望，结合风险桥接并利用准确率非负，得 (21)。证毕。

## 5. 从标签信息得到可计算服务器效用

### 5.1 理想共享类别条件模型

设 \(L_y(w)\) 为共同类别条件损失，局部目标是

\[
F_{k,L}(w)=\sum_yp_k(y)L_y(w)+\mathcal R(w),
\tag{25}
\]

其中 \(\mathcal R\) 为共同非负正则项。定义

\[
M(w)=[\nabla L_1(w),\ldots,\nabla L_C(w)],\qquad
\sup_w\|M(w)\|_{\rm op}\le G_{\rm cls}.
\tag{26}
\]

此时 \(\Delta_k=M e_k\)。令

\[
v=\sum_{k\in\mathcal P}a_k\rho_ke_k,\quad
B_{\rm label}=\|v\|^2,
\tag{27}
\]

\[
V_{\rm label}=\sum_ka_k\rho_k^2\|e_k\|^2-\|v\|^2
=\sum_ka_k\|\rho_ke_k-v\|^2\ge0.
\tag{28}
\]

由算子范数界直接有 \(\mathcal B^2\le G_{\rm cls}^2B_{\rm label}\)、\(\mathcal H^2\le G_{\rm cls}^2V_{\rm label}\)。最终学习目标为

\[
\boxed{J_{\rm label}=
C_BG_{\rm cls}^2B_{\rm label}+C_HG_{\rm cls}^2V_{\rm label}
+C_LV_{\rm loc}+C_AV_{\rm agg},\quad U_s=-J_{\rm label}.}
\tag{29}
\]

最大化 \(U_s\) 严格等价于最小化未截断风险证书。准确率下界的截断可能形成零值平台，不能声称两者的全部最优解集合相同。

### 5.2 保留原 Section 的有限经验目标时，加入残差

纯 label-skew 的总体假设不会消除有限客户端样本的类别条件经验误差。若保留原 Section 的经验目标，写成

\[
\Delta_k(w)=M(w)e_k+r_k^{\rm grad}(w),\qquad
\sup_w\|r_k^{\rm grad}(w)\|\le\varepsilon_k.
\tag{30}
\]

注意 \(r_k^{\rm grad}\) 是梯度残差，不是价格。\(\varepsilon_k\) 必须来自有保证的审计或统一界。

一个保持凸二次结构的保守修正为

\[
\widehat{\mathcal B}^2=2G_{\rm cls}^2B_{\rm label}
+2\left(\sum_ka_k\rho_k\varepsilon_k\right)^2,
\tag{31}
\]

\[
\widehat{\mathcal H}^2=2G_{\rm cls}^2V_{\rm label}
+2\sum_ka_k\rho_k^2\varepsilon_k^2.
\tag{32}
\]

推导分别使用平方三角界，以及残差的加权方差不超过其二阶矩。令

\[
\boxed{J_{\rm residual}=C_B\widehat{\mathcal B}^2+C_H\widehat{\mathcal H}^2
+C_LV_{\rm loc}+C_AV_{\rm agg}.}
\tag{33}
\]

这时使用 \(U_s=-J_{\rm residual}\)。程序提供 `residual_bounds` 参数并实现 (33)；当全部残差为零时，直接使用更紧的 (29)，无需多乘 2。

对于归一化特征 softmax，若只能利用粗略梯度范数界，\(\varepsilon_k=\sqrt2R_x(2+\|e_k\|_1)\) 是一个非常保守的可用界：原始与全局的梯度差至多 \(2\sqrt2R_x\)，类别映射项至多 \(\sqrt2R_x\|e_k\|_1\)。这种界可能失去实用区分能力，应诚实报告。仅在初始化测量残差并设为全程界没有证明效力。

### 5.3 噪声与数据量

若 \(v_k=\sigma^2/b_k\)，应始终先计算

\[
V_{\rm loc}=\sigma^2\sum_k\frac{a_k}{b_k},\qquad
V_{\rm agg}=\sigma^2\sum_k\frac{a_k^2}{b_k}.
\tag{34}
\]

默认实现所有客户端 \(b_k=32\)，所以 \(V_{\rm loc}=\sigma^2/32\) 是配置无关项，\(V_{\rm agg}=(\sigma^2/32)\sum a_k^2\)。此时不能使用 \(h/(ND_{\mathcal P})\) 作为数据量项。

只有 \(b_k=n_k/h\) 精确成立时，才可化简为

\[
V_{\rm loc}=\frac{\sigma^2h|\mathcal P|}{ND_{\mathcal P}},\qquad
V_{\rm agg}=\frac{\sigma^2h}{ND_{\mathcal P}}.
\tag{35}
\]

程序的 `batch_mode="proportional"` 使用 \(b_k=\max(1,\lfloor n_k/h\rfloor)\)，并按实际整数 batch 计算 (34)，不假装取整之后仍精确满足 (35)。固定 batch 的受控总体实验主要体现原始数据量的相对权重，而不是有限样本泛化收益；不能用它单独证明增加绝对样本数的统计收益。

## 6. Theorem 2：最低实施价格与支付

设 \(t_k=\bar\lambda-\lambda_k\)。对于 \(\lambda_k>0\)：

**原始参与。** 可实施当且仅当 \(t_k>0\) 且 \(s_k\le\alpha_kt_k\)。最低价格与支付为

\[
r_{k,\min}(0)=s_k/c_k,\qquad \pi_k(0)=d_ks_k.
\tag{36}
\]

**内部增强 \(0<q<\lambda_k\)。** 唯一可实施价格为

\[
r_{k,\min}(q)=\bar\lambda(\alpha_k+2\beta_kq),
\tag{37}
\]

且必须满足

\[
f_k(q)=\alpha_kt_k+2\beta_kt_kq+\beta_kq^2-s_k\ge0.
\tag{38}
\]

最低支付为

\[
\pi_k(q)=d_k(\alpha_k+2\beta_kq)(t_k+q).
\tag{39}
\]

**完全增强。** 最低价格与支付为

\[
\boxed{r_{k,\min}(\lambda_k)=
\max\{\bar\lambda(\alpha_k+2\beta_k\lambda_k),
s_k+\alpha_k\lambda_k+\beta_k\lambda_k^2\},
\quad\pi_k(\lambda_k)=d_kr_{k,\min}(\lambda_k).}
\tag{40}
\]

**证明。** 对 (11) 求导得到 (12)。原始端点要求右导数非正，完全增强端点要求左导数非负，内部点要求导数为零。将这些条件分别与参与净效用非负条件联立，即得 (36)—(40)。对内部点抬价补足亏损会改变最优增强量，因此不能把参与约束当作事后补贴而维持原增强量。

\(\lambda_k=0\) 时只需要原始参与，\(c_k=1\)，最低价格 \(s_k\)，支付 \(d_ks_k\)。

## 7. Theorem 3：服务器求解与均衡

用 \(J\) 表示按适用条件选择的 (29) 或 (33)。原价格博弈等价于

\[
\boxed{\begin{aligned}
\min_{\mathcal P\ne\varnothing,\mathbf q}\quad&J(\mathcal P,\mathbf q)\\
\mathrm{s.t.}\quad&q_k\text{ 满足最低价格可实施条件},\\
&\sum_{k\in\mathcal P}\pi_k(q_k)\le B.
\end{aligned}}
\tag{41}
\]

任意价格诱导的配置，其支付不低于最低实施支付；反之 (41) 的解可通过 (36)—(40) 恢复价格。因此最优配置和最优值在两种表述之间对应。

### 7.1 模式分解

每个客户端至多有四类模式：不参与、原始参与、连续参与分支、完全增强端点。连续分支下界是

\[
q_{k,\min}=\begin{cases}
0,&s_k\le\alpha_kt_k,\\
\sqrt{t_k^2+(s_k-\alpha_kt_k)/\beta_k}-t_k,&s_k>\alpha_kt_k.
\end{cases}
\tag{42}
\]

若 \(q_{k,\min}\le\lambda_k\)，连续模式为闭区间 \([q_{k,\min},\lambda_k]\)，支付用 (39)。其零端点可能比 (36) 支付更多，但保留原始参与模式就不会损失最优性。若 \(q_{k,\min}>\lambda_k\)，连续模式不存在，完全增强仍可通过 (40) 补足参与成本。

固定模式组合后，\(\mathcal P,a_k\) 固定。标签偏差是仿射表达式的平方范数；标签方差可写为

\[
V_{\rm label}=\frac12\sum_{i,j}a_ia_j\|\rho_ie_i-\rho_je_j\|^2.
\tag{43}
\]

残差修正也为 PSD 二次形式，所以固定模式的目标凸。预算中的支付二阶导为 \(4d_k\beta_k>0\)，因此该子问题为凸二次约束问题。

### 7.2 配套程序如何求解

程序枚举模式组合，以归一化增强比例 \(u=q/\lambda\) 为变量，将目标统一除以 \(C_B>0\) 改善数值尺度。单个变量的支付写成

\[
\pi_k(u)=A_ku^2+B_ku+C_k,
\]

\[
A_k=2d_k\beta_k\lambda_k^2,\quad
B_k=d_k(\alpha_k\lambda_k+2\beta_kt_k\lambda_k),\quad
C_k=d_k\alpha_kt_k.
\tag{44}
\]

对子问题使用投影梯度迭代。投影至区间与预算交集的坐标公式为

\[
u_k(\zeta)=\left[\frac{y_k-\zeta B_k}{1+2\zeta A_k}\right]_{\ell_k}^{h_k},
\tag{45}
\]

其中预算乘子 \(\zeta\ge0\) 通过一维二分计算。满足预算时取零，否则取使支付达到预算的乘子。程序使用预算可行的一侧作为二分返回值。

为了不把“算法停止”误当成“全局最优”，程序还计算凸函数支持超平面与 Lagrange 对偶下界。对当前点 \(u\)、梯度 \(g\)，任意 \(\zeta\ge0\) 给出

\[
\mathrm{LB}=J(u)-g^\top u-\zeta B_{\rm free}
+\sum_k\min_{v_k\in[\ell_k,h_k]}\{g_kv_k+\zeta\pi_k(v_k)\}.
\tag{46}
\]

各模式下界的最小值是全局下界，当前最佳可行目标是全局上界。输出二者之差，记录在 `normalized_absolute_gap`。所有量使用浮点计算，这属于数值最优性间隙，不是区间算术验证。

枚举复杂度最坏 \(4^K\)。程序拒绝超过 100,000 个模式组合，防止将小规模精确方法误当成大规模算法。大规模版本需要实现带界的分支定界；若换成局部搜索，应报告其为启发式。

### 7.3 均衡与性质

若至少一个非空配置预算可行，则有限个紧模式域上的连续目标存在最优解。用最低价格实施该解，得到给定 tie-breaking 下的 Stackelberg 均衡。程序求到有限数值间隙时，应称为相应容差下的近似均衡。

机制满足完全信息下的最优响应、个体理性和预算可行。预算不必花完，因为继续增强可能增加目标偏差；在内部最优点，边际风险证书下降与边际支付之比等于预算乘子。

## 8. 具体增强实现

### 8.1 与目标混合精确一致的采样方法

对客户端 \(k\) 的每个 batch 样本，独立执行：

1. 以概率 \(1-u_k^{\rm enh}\)，从原始客户端分布抽样；
2. 以概率 \(u_k^{\rm enh}\)，从固定目标分布抽样。

这使每步随机梯度的条件期望精确对应 (6)。控制的是采样权重，不是把合成数据简单拼接后继续均匀读取。

受控 label-skew 程序中，同一类别的条件分布是一个共同的有限支持池 \(P_y\)。先抽标签

\[
y\sim \rho_kp_k+(1-\rho_k)p_\star,
\]

再从 \(P_y\) 均匀抽特征。这精确实现 (25)、(6)。\(n_k\) 是原始标签计数的总量和合同份额基础；共享条件池是理想总体定义，不是每个客户端互不相交的真实经验数据集。

对于原 Section 的真实有限数据经验目标，模拟器中可以让目标 oracle 先按 \(d_j\) 抽客户端 \(j\)，再均匀抽其原始训练样本。这样目标分支精确对应原始全体经验分布。集中数据访问只用于受控模拟，不是现实联邦部署协议；真实部署应由符合接口的生成服务提供样本。

### 8.2 如果必须只追加合成样本

想得到标签目标 \(p'_k=\rho_kp_k+(1-\rho_k)p_\star\)，原始类别数量为 \(n_{ky}\)。忽略取整时可选

\[
n'_k\ge\max_{y:p'_k(y)>0}\frac{n_{ky}}{p'_k(y)},\qquad
m_{ky}=n'_kp'_k(y)-n_{ky}\ge0.
\tag{47}
\]

按 \(m_{ky}\) 生成每类样本，再混合训练。但取整和生成类别条件误差都会破坏精确目标混合，应单独审计。只对齐标签比例不能保证特征条件分布对齐。追加样本还可能改变计算量，因此实际成本要重新测量，不能自动沿用固定成本系数。

主实现优先使用重加权或混合采样：它把学习目标、增强比例和固定本地步数对应起来。

### 8.3 接入已有训练代码的接口

需要提供的输入：客户端原始数量 \(n_k\)、标签比例 \(p_k\)、已认证的 \(\lambda_k\) 或其构造所需界、经济参数、batch sizes、训练参数，以及有限数据情况下的 \(\varepsilon_k\)。

需要替换的训练位置只有：原有客户端 dataloader 采用混合采样；聚合按原始 \(n_k\) 归一化；训练仍使用事先固定的 \(T,h,\eta\)。服务器的选择与增强方案从求解器输出读取。不要按合成后数据量重算聚合权重，也不要将 \(q_k\) 当成额外本地 epoch 数。

配套代码 `build_instance` 构造受控数据；`continuous_response` 给出响应；`solve` 求模式配置；`minimum_price` 恢复价格；`train` 实现混合采样 FedAvg。接入真实项目时保留求解与价格接口，替换数据构造和训练采样，并按真实模型重新核算证书。

## 9. 参数取值：理论量与实验量分开

以下数字是本版提供的可运行起点，**不是从引用论文抄来的最优参数，也不是经过真实数据调优的结果**。

### 9.1 默认受控实验

| 参数 | 默认值 | 含义 |
|---|---:|---|
| 客户端数 \(K\) | 6 | 便于穷举模式并报告全局数值间隙 |
| 类别数 \(C\) | 3 | 多分类 softmax |
| 特征维数 | 8 | 无单独截距项，全部参数正则化 |
| 每类条件支持数 | 100 | 定义共同的有限支持总体 |
| \(n_k\) | 整数均匀抽样 200–600 | 用于原始计数与数据份额 |
| Dirichlet 参数 | 0.3 | 生成每个客户端的标签比例，再抽原始类别计数 |
| 特征范数上界 \(R_x\) | 1 | 每个特征向量缩放到单位球内 |
| \(T\) | 100 | 每次任务的通信轮数 |
| \(h\) | 5 | 每轮本地 SGD 步数 |
| batch mode | fixed | 每个客户端每步 batch 为 32 |
| L2 系数 \(\mu\) | 0.05 | 目标加入 \(\mu\|W\|_F^2/2\) |
| \(\eta\) | \(0.1/L\approx0.181818\) | 在理论允许范围内固定 |
| 数据/经济类型种子 | 2026 | 固定配置，训练种子另设 |
| 训练种子 | 0,1,2,3,4 | 同一合同做 5 次独立训练 |
| 预算比例 | \(0.4 B_{\rm full}\) | 所有方法使用同一个绝对预算 |
| 求解归一化绝对容差 | \(10^{-7}\) | 目标先整体除以 \(C_B\) |
| 单模式迭代上限 | 800 | 未达到容差时仍报告剩余间隙 |

### 9.2 可计算的理论常数

对于无截距、全部权重正则化的多分类 softmax，\(\|x\|\le R_x\) 时，

\[
\|\nabla_W\ell_{\rm CE}\|_F\le\sqrt2R_x,\qquad
L\le R_x^2/2+\mu,\qquad G_{\rm cls}^2\le2CR_x^2.
\tag{48}
\]

理由是 \(\nabla_W\ell=(p-e_y)x^\top\)，\(\|p-e_y\|\le\sqrt2\)；softmax 概率协方差矩阵的谱范数至多 \(1/2\)。L2 正则提供全参数强凸性。存在未正则化截距时不能直接声称最小曲率为 \(\mu\)。

每步有放回独立采样，\(b_k\) 个样本的均值梯度可使用

\[
v_k=2R_x^2/b_k.
\tag{49}
\]

在理想标签模型下，

\[
\lambda_k=\sqrt2R_x\|p_k-p_\star\|_1.
\tag{50}
\]

有残差时，可用 \(\lambda_k=\sqrt2R_x\|e_k\|_1+\varepsilon_k\)。这个值是保守上界，不声称等于真实梯度偏差。\(q_k\) 是这个既定质量尺度下的减少量。

默认取 \(\bar\lambda=1.1\max_k\lambda_k\)，使非退化客户端原始质量报酬系数为正。这个 1.1 是合同归一化选择，会影响价格激励，必须固定并报告，不应随不同预算方法分别调整。

代入默认训练配置：

\[
L=0.55,\quad \eta=0.18181818,\quad G_{\rm cls}^2=6,
\]

\[
C_B\approx217.712669,\quad C_H\approx26.444138,\quad
C_L\approx8.788034,\quad C_A\approx0.989603.
\tag{51}
\]

这些是计算结果，不是四个待调权重。优化时程序整体除以 \(C_B\)，不改变最优解。

### 9.3 经济参数

建议用“完成全部增强的线性成本”和“完成全部增强的二次成本”参数化：

\[
A_k^{\rm cost}=\alpha_k\lambda_k,\qquad
B_k^{\rm cost}=\beta_k\lambda_k^2.
\tag{52}
\]

默认抽样为

\[
s_k\sim U(0.01,0.03),\quad A_k^{\rm cost}\sim U(0.02,0.08),\quad
B_k^{\rm cost}\sim U(0.04,0.12).
\tag{53}
\]

再由 \(\alpha_k=A_k^{\rm cost}/\lambda_k\)、\(\beta_k=B_k^{\rm cost}/\lambda_k^2\) 换回原模型。这样梯度单位变化不会无意中改变完成同等增强比例的设定成本。

这些是整个任务的归一化货币单位。实际客户端总基线成本为 \(d_ks_k\)，不是直接 \(s_k\)。真实部署应测量训练/通信的基线成本和生成/增强成本曲线，再拟合 \(\alpha_k,\beta_k\)；需报告拟合误差。\(T\) 改变时，任务级成本也应重新核算。

定义同一个连续个性化合同下的全体完全增强支出

\[
B_{\rm full}=\sum_kd_kr_{k,\min}(\lambda_k).
\tag{54}
\]

默认预算为 \(0.4B_{\rm full}\)，扫描建议为 \(\{0.2,0.4,0.6,0.8,1.0\}B_{\rm full}\)。不可行预算如实报告。基线方法使用这个共同绝对预算，不能各自用自己的 \(B_{\rm full}\) 归一化后声称预算相同。

### 9.4 真实数据的建议起点

可先在 [Fashion-MNIST 官方数据](https://github.com/zalandoresearch/fashion-mnist) 上用归一化像素或固定特征训练全参数 L2 正则 softmax，设置 \(K=10\)、Dirichlet 参数 \(\{0.1,0.3,1.0\}\)、\(T=100\)、\(h\in\{1,5,10\}\)、batch 32、5 个训练种子。\(K=10\) 的模式数仍可能很大，应看实际模式数与计算预算。

这组数字仅为实验设计起点。真实有限数据应启用残差处理或改用已认证梯度矩阵，不能因为使用 Dirichlet label split 就把 \(\varepsilon_k\) 设零。CNN/ResNet 可以作为经验扩展，但本版强凸定理不自动适用。不得查看测试 ACC 来选择证书系数、预算或合同。

## 10. 泛化项与数值准确率证书

默认受控实验把 \(Q\) 定义为 \(p_\star\) 加共同有限类别支持构成的总体，直接遍历支持点精确计算该总体风险。\(F\) 为它的交叉熵加非负正则，故 \(\Gamma=0\)。这不是未见过的真实图像测试集上的泛化实验。

真实有限数据中，需要一个对全部可行模型统一的桥接。一个充分条件是：独立原始样本，\(\Omega\) 上交叉熵有界 \(M_\ell\) 且对参数一致 \(L_w\)-Lipschitz。对 \(\varepsilon\)-覆盖网使用 Hoeffding 不等式和 union bound，得到以概率至少 \(1-\delta\) 成立的

\[
\Gamma_{\rm gen}=M_\ell\sqrt{\frac{\ln[2\mathcal N(\varepsilon,\Omega)/\delta]}{2N}}
+2L_w\varepsilon.
\tag{55}
\]

若目标总体不同，且样本空间中的损失对所有模型统一 \(\kappa\)-Lipschitz，可再加 \(\kappa W_1(P,Q)\)。否则不能直接使用该分布偏移公式。FedBary 的相关定理有自己的 Lipschitz 等条件，应逐项核对；它不是无需条件的测试风险桥梁。[FedBary 官方页面](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Data_Valuation_and_Detections_in_Federated_Learning_CVPR_2024_paper.html)

若风险桥接来自概率事件，则准确率定理是在该事件上的训练期望保证。不能把期望界写成每一次训练必然成立的界。

默认程序为避免偷偷使用未知 \(F^\star\)，采用可验证的 \(F^\star\le F(0)=\ln C\)，并由强凸性、\(F^\star\ge0\) 得 \(R_0^2\le2F(0)/\mu\)。这非常保守，且 \(C\ge2\) 时这个 \(F^\star\) 上界本身已使最终 ACC 下界非正。程序因此诚实输出零。要得到正的数值证书，需要更紧的共同 \(F^\star\) 上界、初始距离界及整体学习误差界，而不是删除正项或手调权重。

## 11. 三状态基线与比较方式

配套代码实现三个机制：连续个性化、三状态个性化、三状态公共价格。它们共享原始客户端、成本函数、标签效用、训练协议和绝对预算。后两者是同成本对照重建，**不是声称复现 IMFL-AIGC 论文全部设置或算法**。

三状态动作只有不参与、原始参与、完全增强；并列时优先完全增强，再原始参与，再不参与。完全增强价格要求

\[
r\ge\bar\lambda(\alpha_k+\beta_k\lambda_k)
\quad\text{且}\quad r\ge s_k+\alpha_k\lambda_k+\beta_k\lambda_k^2.
\tag{56}
\]

在这个并列规则下，三状态原始参与要求
\(s_k<t_k(\alpha_k+\beta_k\lambda_k)\)。等号时最低原始参与价会使完全增强并列胜出，所以代码不保留该原始模式。

公共价格基线枚举所有原始参与、完全参与和两端点切换的价格阈值。每个阈值间的学习配置不变，提高价格只会增加支出，因此检查阈值即可。

连续机制的完全增强门槛含 \(2\beta_k\lambda_k\)，三状态只含 \(\beta_k\lambda_k\)。新增内部动作可能提高实施相同端点的成本。因此不存在本版博弈下一般成立的“连续 ACC 下界最优值必不差于三状态”的推论。

## 12. 已执行的默认结果

使用 `config.json` 的 6 客户端实例，原始数量为 \([541,271,210,456,346,387]\)，共同预算为 0.1680560816501262。5 次训练的默认结果如下；它们仅用于运行与机制检查。

| 机制 | 支付 | \(J/C_B\) | 受控总体 ACC 均值 | ACC 样本标准差 |
|---|---:|---:|---:|---:|
| 连续个性化 | 0.154947797 | 0.002621382 | 87.2201% | 0.3344 个百分点 |
| 三状态个性化 | 0.155095371 | 0.002597434 | 87.2714% | 约 0 个百分点 |
| 三状态公共价格 | 0.134592610 | 0.146951370 | 85.6138% | 0.2806 个百分点 |

连续个性化方案的归一化全局数值间隙约为 \(4.08\times10^{-11}\)。默认参与客户端为第 2、3、5 个；增强比例约为 \([0,0.9999941,1,0,0.9999952,0]\)。价格约为 \([0,0.69223635,0.44869148,0,0.17563301,0]\)。这些是特定模拟实例的输出，不是应复制到真实实验的固定单价。

三状态个性化在这个实例中的目标与 ACC 略优于连续机制，因此这里没有报告连续化普遍优越。默认连续解接近端点，说明连续动作允许内部增强，但并不强迫最优解位于内部。不同预算/成本条件可能产生内部解。

数值证书对三种机制均截断为零。不能将上表中的实际 ACC 写成证书下界。零标准差只说明本例五次分类决策恰好给出相同加权正确率，不能推断训练无随机性。

机器可读结果、每次训练的交叉熵和正则化损失、实际价格、支付、全局间隙与全部实例元数据见 `results/results.json`；条件支持池见 `results/instance.npz`。

### 12.1 可解析核对的内部增强例子

另提供 `config_interior_example.json`：两客户端 \(n_1=n_2=400\)，共同 \(s=0.02\)、\(\alpha\lambda=0.05\)、\(\beta\lambda^2=0.1\)，\(h=2\)，\(\bar\lambda=1.1\lambda\)，预算 0.11。其他数据生成设置与默认配置保持同类结构。由两个客户端原始份额相同可知 \(e_1=-e_2\)、\(\lambda_1=\lambda_2\)。

在两个客户端采用相同增强比例 \(u\) 时，总支付为

\[
(0.05+0.2u)(0.1+u)=0.11.
\]

其正根为 \(u=0.5704025758\)，对应价格 \(r=1.1(0.05+0.2u)=0.1804885667\)。求解器返回这一配置，各支付 0.055，并通过客户端响应和预算校验。这是一个构造的对称内部解验证例子，不构成一般准确率优势定理。该预算下公共价格三状态没有非空可行解，结果文件如实标记为 infeasible。

### 12.2 已完成的检查

10 项自动检查覆盖：连续与离散端点价格差异、内部参与亏损、不可实施原始参与、零异质性客户端、预算投影、对偶下界、与独立稠密网格的最优值比较、单步漂移项消失、实际整数 batch 方差、共享类别梯度与残差上界。具体见 `test_baqp.py`；默认和内部例子的价格恢复后均再次检查了客户端最优响应。

## 13. 引用与各自用途

1. **McMahan et al., AISTATS 2017, Communication-Efficient Learning of Deep Networks from Decentralized Data.** 用于 FedAvg 的本地训练与模型聚合背景。[官方论文](https://proceedings.mlr.press/v54/mcmahan17a.html)
2. **Karimireddy et al., ICML 2020, SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.** 用于异质性与 client drift 的理论背景，不引用它来声称本机制具有同样的算法保证。[官方论文](https://proceedings.mlr.press/v119/karimireddy20a.html)
3. **Wang et al., ICML 2024, A New Theoretical Perspective on Data Heterogeneity in Federated Optimization.** 用于异质性、本地更新与噪声的分析背景；其 Theorem A.9 和 Corollary A.10 有特定条件，不能把渐近阶直接作为本版精确系数。[官方页面](https://proceedings.mlr.press/v235/wang24bu.html)，[含附录全文](https://arxiv.org/html/2407.15567v1)
4. **Huang et al., 2024, IMFL-AIGC: Incentive Mechanism Design for Federated Learning Empowered by Artificial Intelligence Generated Content.** 用于 AIGC 质量激励、原始/增强参与状态与收敛代理的背景。本包按可核验的 arXiv 版本引用，未猜测正式卷期页码。[原文](https://arxiv.org/abs/2406.08526)
5. **Bartlett, Jordan, McAuliffe, JASA 2006, Convexity, Classification, and Risk Bounds.** 用于 surrogate risk 与分类风险关系的背景；本文使用的 \(\mathbf1[\widehat y\ne y]\le\ell_{CE}/\ln2\) 已直接证明，不冒称为该文的一般多分类定理。[论文全文](https://sites.stat.washington.edu/courses/stat527/s14/readings/Bartlett_etal_JASA_2006.pdf)
6. **Boyd and Vandenberghe, 2004, Convex Optimization.** 用于凸子问题、KKT 与对偶下界的通用依据；本版最低实施价格和模式分解另行推导。[作者官网](https://web.stanford.edu/~boyd/cvxbook/)
7. **Li et al., CVPR 2024, Data Valuation and Detections in Federated Learning.** 可用于分布距离与风险迁移的相关工作，默认 \(Q=P\) 的受控实现不依赖它。[官方论文](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Data_Valuation_and_Detections_in_Federated_Learning_CVPR_2024_paper.html)

完整书目信息见 `references.bib`。本文的客户端二次成本、连续合同、标签/残差证书及求解组合是本版建模选择与推导，不能用这些引用替代自身证明。
