# 追加增强版本同步与验证

- 从本地e9f593f快进同步到origin/main的c725381。
- 远端变更：addition-only计数、增强集合采样、full_balance_all、逐客户端augmentation_plan、新输出目录。
- 本地补充：自动汇总默认目录改为outputs/real_baqp_additive；README补充缓存重复使用、原始数据保留与固定步数采样的区别；新增tests/test_additive.py的3项完整采样链路检查。
- 11项自动测试通过。baqp/mechanism.py未修改；经济参数、价格响应、服务器J保持原实现。
- 原输出outputs/real_baqp保留为replacement历史，不并入新结果。
- 追加规模预审计见continuous_additions.json。
- GPU烟测：1588748，三数据集各2轮×2步。
- 正式任务：1588749，依赖烟测全部成功，三数据集×5种子，100轮×5步。
- 自动汇总：1588752，读取outputs/real_baqp_additive。
- 本地补充文件未提交/推送到GitHub。
