# 模型登记

Demo 当前默认使用 **R10**。R8 是此前部署版本，作为历史模型归档，供复现和对照时查阅。

| 项目 | R10（当前） | R8（归档） |
|---|---|---|
| 模型 ID | `step-r10-r9-both-static2-smart2-other1-rot3-tmse-50epochs-20261008` | `step-r8-r7-today-smart2-other1-rot3-tmse-20261008` |
| PyTorch checkpoint | `models/R10/causal_cnn_best.pt` | `models/R8/causal_cnn_best.pt` |
| 归一化参数 | `models/R10/norm_stats.json` | `models/R8/norm_stats.json` |
| 手机模型 | Demo 中的 R10 `.ms` | `models/R8/phone_inhand_step_r8_r7_today_smart2_other1_rot3_tmse_20261008.ms` |
| 配方 / 数据清单 | `PhoneInHand/configs/R10.json` / `data/splits/R10.json` | `models/R8/experiment_config.json` / `data/splits/R8.json` |
| 训练规模 | 169 recordings，633846 windows，50 epochs，seed 42，MPS | 161 recordings，599038 windows，40 epochs，seed 42，MPS |
| 验证与选轮 | 47 recordings，59874 有效帧；最佳 epoch 12 | 39 recordings，50634 有效帧；最佳 epoch 31 |
| 逐帧验证指标 | Accuracy 97.4897%；Macro-F1 0.974865；False Free Rate 1.6822% | Accuracy 98.5701%；Macro-F1 0.985659；False Free Rate 1.6434% |

R8 保留 R7 的 156/34 录制划分，并将 2026-10-08 新采集的 10 条录制按时间排序后交替分入训练和验证。SmartRotate 的 72 条训练录制使用 stride 2、不做旋转增强；其他 89 条训练录制使用 stride 1 和 3 个固定种子的 SO(3) 旋转副本。每段裁掉开头 3 秒，窗口长度为 128 帧，归一化参数随模型归档。R8 与 R10 的验证划分不同，因此两者的指标不应直接比较。

`models/R10/train_history.json` 保留 R10 的 50 轮训练记录；`models/R8/train_history.json` 保留 R8 的训练记录。`reference_metrics.json` 保存逐录制结果。模型与对应归一化参数、部署协议应作为整体使用。R8 的 ONNX 导出和签名材料未纳入本仓库。
