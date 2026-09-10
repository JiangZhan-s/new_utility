# BAQP：三个真实数据集与离线 AIGC 的实验

代码对应 `FINAL_GUIDE.md` 的价格与预算机制。主实验是 CNN 的经验扩展，服务器 J 是理论启发的代理目标；没有正的 test accuracy 下界证书，也不预设连续机制一定优于基线。你的 KKT 推导及数值例子已独立核验，见 `STACKELBERG_KKT_REVIEW.md`。

## 重要更新：AIGC 为“追加增强”，不替换原始数据

当前实现采用 **addition-only augmentation**：客户端原始样本始终全部保留，AIGC 只补充缺失类别。连续增强量 `u_k=q_k/lambda_k` 定义目标标签分布

\[
p_k^{(u)}=(1-u_k)p_k+u_kp_\star.
\]

对客户端原始类别计数 `n_ky`，程序寻找最小整数增强后总量 `N'_k`，使最终类别计数近似 `N'_k p_k^(u)` 且逐类不少于原始计数；新增数量为

\[
m_{ky}=n'_{ky}-n_{ky}\ge 0.
\]

训练时从 `D_real ∪ D_syn` 的增强数据集均匀抽样。**不会删除、降采样或用 AIGC 覆盖任何原始样本。** FedAvg 聚合权重仍按原始客户端数据量归一化，因此生成更多样本不会增加客户端的聚合 voting power。

此前 `outputs/real_baqp` 中的结果来自旧的 replacement-mixture 训练语义，不能作为当前 addition-only 实现的实验结果。新实验默认输出目录改为 `outputs/real_baqp_additive`，必须重新运行后再比较 ACC。

## 环境和数据

使用 `/share/home/202521044325/miniconda3/envs/aigc39/bin/python`，已确认 PyTorch 2.5.1+cu121、NumPy 2.0.2、SciPy 1.13.1。实现不需要下载数据或 torchvision。

| 名称 | 原始数据 | 增强缓存 |
|---|---|---|
| cifar10 | data/cifar-10-batches-py | aigc_imgs/cifar10_edm/tensor_cache.pt |
| cifar100 | data/cifar-100-python | aigc_imgs/cifar100_styleganxl/tensor_cache.pt |
| fmnist | data/FashionMNIST/raw | aigc_imgs/fmnist_edm_oracle_6000pc_seed42/tensor_cache.pt |

所有缓存通过 `weights_only=True` 加载，检查形状、类型和类别覆盖。增强图片来自上述缓存；不以原始测试图片替代增强数据。Fashion-MNIST 缓存生成质量/是否曾利用测试集不能仅凭文件名判定，本实验没有认证其上游生成过程。

## 默认实验

- 6 个客户端，Dirichlet α=0.3，训练集按类别划出10%验证数据，剩余90%全部且互斥地分配客户端。
- split/economic seed=2026 固定；训练 seed=0,1,2,3,4。
- 共同预算是全员连续合同完全增强支付的40%；所有经济基线使用同一绝对预算上限，但实际开支可以不同。
- CNN：100轮，每轮每参与客户端5步，batch32，SGD，无动量，lr=0.05，L2=0.0005。
- softmax 可选：单位球像素特征，无偏置，全参数 L2=0.05，lr=0.1/0.55；可使用 `--residual` 加入原始有限数据的保守残差，仍未认证生成器偏差及风险桥接。
- 每个参与客户端先根据合同 `u_k` 构造 addition-only 类别补充计划，然后每个本地 SGD batch 从该客户端的“原始+新增 AIGC”数据集中均匀抽样。
- FedAvg 始终按参与客户端**原始数据量**归一化加权，不按增强后样本量重算权重。
- 每个方法目录保存 `augmentation_plan.json`，记录原始样本量、每类新增 AIGC 数量、增强后类别计数和目标标签分布。
- 最终固定轮 test accuracy 是主指标；每10轮只报告 validation accuracy。不通过测试集选择合同、权重或 checkpoint。

CNN 的 `.5+mu` 等常数只是沿用 softmax 的代理尺度，不能声称是 CNN 的光滑性、强凸性或梯度上界。真实 AIGC 的类别条件分布误差仍未被理论证书覆盖。

## 方法

| 方法 | 用途 |
|---|---|
| continuous | 连续个性化价格；枚举模式、凸求解、对偶数值间隙 |
| three_state | 离散动作{退出、原始、完全增强}的个性化价格 |
| public_price | 三状态共同价格，枚举响应切换阈值 |
| no_aigc_budget | 在连续客户端响应下可实施的原始参与集合，零增强、共同预算 |
| random_budget | 固定种子随机可行合同，不按目标或 accuracy 挑选随机样本 |
| fedavg_all | 全员原始训练参考，不属于预算可行激励合同 |
| selected_no_aigc | continuous 同一参与集合、强制零增强的学习消融，不属于可实施合同 |
| full_balance_all | 全员保留全部原始数据，并追加 AIGC 至各客户端标签分布达到 `p_star`；不属于预算可行经济机制 |

`full_balance_all` 取代旧名称 `synthetic_only`。在当前实现中不存在“用纯合成数据替换全部原始数据”的默认基线。

## 运行

先做轻量检查：

```bash
conda activate aigc39
python -m unittest discover -s tests -v
python -m baqp.check_kkt
python -m baqp.experiment --dataset cifar10 --prepare-only --output outputs/prepare_additive_c10
```

GPU 训练示例：

```bash
mkdir -p logs
sbatch --export=ALL,OUTPUT_ROOT=outputs/real_baqp_additive run_baqp_real.slurm
```

改变模型、协议、代码或增强语义后使用新的输出目录。旧 `outputs/real_baqp` 不得与新的 addition-only 结果混合汇总。

预算/Non-IID 扫描示例：

```bash
sbatch --export=ALL,ALPHA=0.1,BUDGET_FRACTION=0.2,OUTPUT_ROOT=outputs/scan_additive_a01_b02 run_baqp_real.slurm
```

## 结果与判断

```bash
python -m baqp.summarize outputs/real_baqp_additive
```

每个数据集/模型/α/预算/seed 目录包含 metadata、partition，以及每种方法的 contract、augmentation_plan、history、result、final_model。合同保存价格、支付、净效用、增强量、预算；连续求解器保存全局数值下界和 gap。

只有新的 addition-only 多种子结果支持正向差异时，才可以据此讨论实际 ACC。旧 replacement 结果只保留作开发历史，不应继续进入论文主表。

## 理论与实际实现的边界

服务器机制与风险证书仍使用 `FINAL_GUIDE.md` 的连续质量变量和 `J`。在纯 label-skew 且 AIGC 类别条件分布与目标分布一致的理想模型下，新增样本使客户端标签分布沿 `p_k -> p_star` 移动，对应梯度偏差缩减。真实生成池一般只近似这一条件，因此 CNN 主实验仍是经验扩展，而不是 Theorem 1 的严格数值证书。
