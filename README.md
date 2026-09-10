# BAQP：三个真实数据集与离线 AIGC 的实验

代码对应 `FINAL_GUIDE.md` 的价格与预算机制。主实验是 CNN 的经验扩展，服务器 J 是理论启发的代理目标；没有正的 test accuracy 下界证书，也不预设连续机制一定优于基线。你的 KKT 推导及数值例子已独立核验，见 `STACKELBERG_KKT_REVIEW.md`。

## 环境和数据

使用 `/share/home/202521044325/miniconda3/envs/aigc39/bin/python`，已确认 PyTorch 2.5.1+cu121、NumPy 2.0.2、SciPy 1.13.1。实现不需要下载数据或 torchvision。与文档描述的原 NumPy 求解器不同，这份新实现依赖 SciPy 求凸子问题；依赖见 `requirements.txt`。

| 名称 | 原始数据 | 增强缓存 |
|---|---|---|
| cifar10 | data/cifar-10-batches-py | aigc_imgs/cifar10_edm/tensor_cache.pt |
| cifar100 | data/cifar-100-python | aigc_imgs/cifar100_styleganxl/tensor_cache.pt |
| fmnist | data/FashionMNIST/raw | aigc_imgs/fmnist_edm_oracle_6000pc_seed42/tensor_cache.pt |

所有缓存通过 `weights_only=True` 加载，检查形状、类型和类别覆盖。增强图片来自上述缓存；不以原始测试图片替代增强数据。Fashion-MNIST 缓存生成质量/是否曾利用测试集不能仅凭文件名判定，本实验没有认证其上游生成过程。

## 默认实验

- 6 个客户端，Dirichlet α=0.3，训练集按类别划出10%验证数据，剩余90%全部且互斥地分配客户端。
- split/economic seed=2026 固定；训练 seed=0,1,2,3,4。配对种子共享初始化、划分、成本及固定每客户端抽样随机流。该设计估计的是固定划分下训练随机性，非多划分泛化结论。
- 共同预算是全员连续合同完全增强支付的40%；所有经济基线使用同一绝对预算上限，但实际开支可以不同。
- CNN：100轮，每轮每参与客户端5步，batch32，SGD，无动量，lr=0.05，L2=0.0005；3层卷积32/64/128，固定平均池化后线性分类头（避免CUDA自适应池化的非确定性反向传播）。输入缩放至[-1,1]。
- softmax 可选：单位球像素特征，无偏置，全参数 L2=0.05，lr=0.1/0.55；可使用 `--residual` 加入原始有限数据的保守残差，仍未认证生成器偏差及风险桥接。
- 每个训练样本独立以u_k概率取合成池，以1−u_k概率取客户端原始池；合成标签从固定p_star抽样，而不是按缓存的类别频率。FedAvg始终按参与客户端原始数据量归一化加权。
- 最终固定轮 test accuracy 是主指标；每10轮只报告 validation accuracy。不通过测试集选择合同、权重或checkpoint。短烟测的低准确率不是正式结论。

CNN 的 `.5+mu` 等常数只是沿用 softmax 的代理尺度，不能声称是 CNN 的光滑性、强凸性或梯度上界。训练长度默认沿用文档起点；若验证集显示未收敛，应预先确定新协议后对所有方法统一重跑，任务成本也应重新核算。

## 方法

| 方法 | 用途 |
|---|---|
| continuous | 连续个性化价格；枚举模式、凸求解、对偶数值间隙 |
| three_state | 离散动作{退出、原始、完全增强}的个性化价格；不同动作空间，不能直接拿连续端点价格代替 |
| public_price | 三状态共同价格，枚举响应切换阈值 |
| no_aigc_budget | 在连续客户端响应下可实施的原始参与集合，零增强、共同预算 |
| random_budget | 固定种子随机可行合同，不按目标或accuracy挑选随机样本 |
| fedavg_all | 全员原始训练参考，不属于预算可行激励合同 |
| selected_no_aigc | continuous 同一参与集合、强制零增强的学习消融，不属于可实施合同 |
| synthetic_only | 全员仅合成池训练参考，不属于预算可行合同 |

后两项帮助区分“合成数据本身带来的收益”和“参与/分配决策带来的收益”。不同方案的参与客户端数可以不同，因此总梯度步数不同；结果保存实际梯度步数和期望合成抽样量，不声称全方法总算力严格相同。所有方案的每客户端本地步数相同。

## 运行

先在登录节点做轻量检查：

```bash
conda activate aigc39
python -m unittest discover -s tests -v
python -m baqp.check_kkt
python -m baqp.experiment --dataset cifar10 --prepare-only --output outputs/prepare_c10
```

GPU只能在Slurm分配内运行。默认使用原脚本中的账户a_xmwang、分区gpuA800；启动前创建日志目录。

```bash
mkdir -p logs
# 三数据集各两轮烟测，串行使用一张GPU，限时10分钟/任务
sbatch --array=0-2%1 --time=00:10:00 --export=ALL,SEEDS=0,ROUNDS=2,STEPS=2,OUTPUT_ROOT=outputs/gpu_smoke run_baqp_real.slurm
# 三个数据集 × 5种子，每任务顺序执行8种方法，最多3个GPU并发
sbatch run_baqp_real.slurm
# 仅CIFAR-10，5种子
sbatch run_aigc_cifar10_real_multiseed.slurm
```

不要同时提交全数据脚本和单CIFAR-10脚本到相同输出目录。改变模型、协议、代码等后使用新 `OUTPUT_ROOT`。脚本默认 `--resume` 仅跳过已完成的方法；不恢复方法内部的训练中断。已存在输出的配置/源代码指纹不同会报错，避免混合旧结果。

单独运行softmax：

```bash
sbatch --export=ALL,MODEL=softmax,RESIDUAL=1,OUTPUT_ROOT=outputs/softmax_residual run_baqp_real.slurm
```

预算/非IID扫描示例（每次仍是三数据集×5种子）：

```bash
sbatch --export=ALL,ALPHA=0.1,BUDGET_FRACTION=0.2,OUTPUT_ROOT=outputs/scan_a01_b02 run_baqp_real.slurm
```

改变 `SEEDS`/`DATASETS` 时同步调整 `--array=0-(数据集数×种子数−1)`。文件名中的 `cifar10` 旧脚本已替换，不再引用不存在的 `src.experiments.run_fl`。

## 结果与判断

```bash
squeue -u "$USER"
python -m baqp.summarize outputs/real_baqp
```

每个数据集/模型/α/预算/seed目录包含metadata、partition，以及每种方法的contract、history、result、final_model。合同保存价格、支付、净效用、增强量、预算；连续求解器保存全局数值下界、间隙和是否达到指定容差。`solver_failures` 是部分子问题SLSQP未宣告成功的数量；支持下界依然有效，最终是否达标以全局gap判断。

汇总产生 `summary.csv/json` 和 `paired.csv/json`：固定末轮test accuracy均值、样本标准差，以及continuous相对各基线的配对差（百分点）与95% t区间。5个训练种子样本较少，区间是固定划分下的探索性统计；没有进行多重比较校正。不完整/不可行方法不会伪造结果，需核对每组种子数及contract状态。

只有正式多种子结果支持正向差异时，才可说当前设置中有效。解析KKT通过、J更小、单个最好seed或两轮烟测都不能证明test accuracy提升。
# new_utility
