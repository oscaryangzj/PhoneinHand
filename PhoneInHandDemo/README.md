# HarmonyOS Demo

Demo 负责 6 轴 IMU 采集、实验场景标注、CSV 导出和 R10 实时检测。

## 打开与运行

1. 用 DevEco Studio 打开本目录，使用 HarmonyOS SDK 6.0.1 / API 21。
2. 同步项目，在 IDE 中配置自己的设备和本地签名。
3. 编译并运行 `entry` 模块。

项目构建配置没有绑定本机路径或签名证书。本目录不含签名/转换脚本、已签名 HAP、测试目录、缓存或私钥。上传代码时保持共享 `build-profile.json5` 不含本地签名材料。

## 主要代码

| 文件 | 职责 |
|---|---|
| `entry/src/main/ets/pages/Index.ets` | 用户与采集场景选择，按手持/非手持分组 |
| `entry/src/main/ets/model/SessionTypes.ets` | 场景、标签、采样与采集参数 |
| `ImuRecorder.ets` / `ImuAligner.ets` | 高频采集、对齐与 100 Hz 输出 |
| `DataExporter.ets` | CSV 生成与导出 |
| `ImuStreamCollector.ets` | 实时 100 Hz 输入 |
| `HoldDetector.ets` / `pages/DetectPage.ets` | R10 流式推理与结果展示 |

采集维持原有 500 Hz 缓存与 100 Hz 落盘流程。数据接口见 [DATA_CONTRACT.md](../docs/DATA_CONTRACT.md)。

## R10 模型

`entry/src/main/resources/rawfile/phone_inhand_step_r10_r9_both_static2_smart2_other1_rot3_tmse_50epochs_20261008.ms` 已随代码提供。`HoldDetector.ets` 中的归一化、输入/状态形状与 [部署协议](../models/R10/deployment.json) 一致，直接使用现成模型即可复现当前版本，无需本地转换。

Python 重训只产出 PyTorch checkpoint，不自动替换手机里的 `.ms`。交接范围保留当前 R10 的训练、评估和已部署模型，未包含生成新的 MindSpore Lite 模型的转换工具。

无签名版本已在独立目录编译通过。安装到自己的真机仍需在 IDE 中配置本机签名。
