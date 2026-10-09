# 模型登记

当前交接版本：**R10**。

| 项目 | 值 |
|---|---|
| 模型 ID | `step-r10-r9-both-static2-smart2-other1-rot3-tmse-50epochs-20261008` |
| PyTorch checkpoint | `models/R10/causal_cnn_best.pt` |
| 归一化参数 | `models/R10/norm_stats.json` |
| 部署协议 | `models/R10/deployment.json` |
| 手机模型 | `phone_inhand_step_r10_r9_both_static2_smart2_other1_rot3_tmse_50epochs_20261008.ms` |
| 训练配方 | `PhoneInHand/configs/R10.json` |
| 数据划分 | `data/splits/R10.json` |
| 训练 | 169 recordings，633846 windows，50 epochs，seed 42，MPS |
| 验证 | 47 recordings，59874 有效帧 |
| 最佳轮 | 12 |

`models/R10/train_history.json` 保留原 R10 的 50 轮训练记录。`models/R10/reference_metrics.json` 是用交接目录独立副本重新评估的结果。checkpoint 的模型权重、模型配置与归一化参数保持原样，仅将历史实验中绑定本机绝对路径的元数据替换为可移植配方。

端侧模型和参数由 Demo 的 `HoldDetector.ets` 使用，未包含旧模型或 R11。
