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

训练数据是 `D_real ∪ D_syn`，不会删除、降采样或用 AIGC 覆盖任何原始样本。FedAvg 聚合权重仍按原始客户端数据量归一化，因此生成更多样本不会增加客户端的聚合 voting power。

## 重要更新：CNN 使用完整 local epochs

CNN 主实验不再使用“每轮每客户端固定 5 个 mini-batch”的训练方式。每个参与客户端现在对自己的**完整增强数据集**执行固定数量的 local epochs；默认 `local_epochs=2`。每个 local epoch 都随机打乱 `D_real ∪ D_syn`，并且无放回遍历一遍全部样本。

因此客户端 k 每轮的实际优化步数为

\[
h_k(q_k)=E\left\lceil\frac{n_k+m_k(q_k)}{b}\right\rceil,
\]

其中 `E=local_epochs`、`b=batch size`。AIGC 增加样本后，这些样本会真正增加本地训练计算量，而不是只扩大数据池却保持固定 5 步。

为保留 Theorem 1 的固定本地步数接口，`softmax` 理论验证路径仍使用 `--steps h`，不切换到 local-epoch 语义。因此：

- `cnn`：`--local-epochs` 控制实际本地训练；
- `softmax`：`--steps` 控制理论中的固定本地步数 h。

CNN 的实际 `gradient_steps` 会逐方法记录在 `history.json` 和 `result.json` 中，同时记录 `actual_real_draws` 与 `actual_synthetic_draws`。

此前 `outputs/real_baqp` 中的结果来自旧 replacement-mixture 训练语义，`outputs/real_baqp_additive` 来自 addition-only 但 fixed-step 的中间版本。两者都不能作为当前 local-epoch 实现的最终实验结果。当前默认新目录为 `outputs/real_baqp_additive_epochs`。

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
- CNN：100轮，每轮每参与客户端 **2 个完整 local epochs**，batch32，SGD，无动量，lr=0.05，L2=0.0005。
- softmax 可选：单位球像素特征，无偏置，全参数 L2=0.05，lr=0.1/0.55，每轮固定 `h=5` 个随机梯度步；可使用 `--residual` 加入原始有限数据的保守残差，仍未认证生成器偏差及风险桥接。
- 每个参与客户端先根据合同 `u_k` 构造 addition-only 类别补充计划；CNN 随后完整遍历增强数据集，softmax 理论路径按固定步数采样。
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
python -m baqp.experiment --dataset cifar10 --prepare-only --output outputs/prepare_additive_epochs_c10
```

GPU 训练示例：

```bash
mkdir -p logs
sbatch --export=ALL,OUTPUT_ROOT=outputs/real_baqp_additive_epochs,LOCAL_EPOCHS=2 run_baqp_real.slurm
```

改变模型、协议、代码或增强语义后使用新的输出目录。旧 `outputs/real_baqp` 和 `outputs/real_baqp_additive` 不得与新的 local-epoch 结果混合汇总。

预算/Non-IID 扫描示例：

```bash
sbatch --export=ALL,ALPHA=0.1,BUDGET_FRACTION=0.2,LOCAL_EPOCHS=2,OUTPUT_ROOT=outputs/scan_additive_epochs_a01_b02 run_baqp_real.slurm
```

## 结果与判断

```bash
python -m baqp.summarize outputs/real_baqp_additive_epochs
```

每个数据集/模型/α/预算/seed 目录包含 metadata、partition，以及每种方法的 contract、augmentation_plan、history、result、final_model。合同保存价格、支付、净效用、增强量、预算；连续求解器保存全局数值下界和 gap。

只有新的 addition-only + local-epoch 多种子结果支持正向差异时，才可以据此讨论实际 ACC。旧 replacement 和 fixed-step additive 结果只保留作开发历史，不应继续进入论文主表。

## 理论与实际实现的边界

服务器机制与风险证书仍使用 `FINAL_GUIDE.md` 的连续质量变量和 `J`。在纯 label-skew 且 AIGC 类别条件分布与目标分布一致的理想模型下，新增样本使客户端标签分布沿 `p_k -> p_star` 移动，对应梯度偏差缩减。

注意：CNN 改为 local-epoch 以后，实际本地步数 h_k 会随 `q_k` 和增强后数据量变化；因此 CNN 主实验更贴近实际数据增强训练，但它**不再对应 Theorem 1 中固定 h 的那组系数**。CNN 仍属于经验扩展。严格固定-h 的理论验证应使用 softmax 路径。

真实生成池一般只近似共享类别条件分布，因此 CNN 主实验也不是 Theorem 1 的严格数值证书。

## 实施检查

若某类所需追加量大于该类缓存大小，代码会有放回抽取缓存索引，增强集合含重复图片；这些重复项随后在每个 local epoch 中各访问一次。各类计数存在整数取整误差，应对照 `augmentation_plan.json` 的目标和实际比例。

经济参数仍按增强程度 u 设置，未按本次真实追加张数重新标定成本。服务器 J 和价格机制保留原模型；真实类别条件分布偏差和 CNN 理论适用范围仍需区分。
