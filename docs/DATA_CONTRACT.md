# 数据与推理接口

Demo CSV 的核心字段：

```text
timestamp_ms,user_name,hold_state,scene_tag,phase,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
```

`timestamp_ms` 为毫秒，acc 为 m/s²，gyro 为 rad/s，目标输出采样率为 100 Hz。输入顺序固定为 `acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z`，标签映射为 `non_handheld=0, handheld=1`。每份 recording 的标签固定，不按 window 重新拆分训练/验证。

R10 先按固定均值与标准差归一化 6 轴 IMU，再在模型内部进行 EMA 去重力，alpha 为 `0.9700000286102295`。不要在模型外额外去重力或更改归一化。

端侧每次输入一帧：

| 名称 | 形状 |
|---|---|
| imu | `[1,1,6]` |
| cnn_buffer | `[1,32,60]` |
| grav_state | `[1,3]` |
| logits | `[1,1,2]` |

输出的 `cnn_buffer_out`、`grav_state_out` 回传下一帧。首次输入时，CNN buffer 清零，重力状态用首帧归一化加速度初始化。模型感受野为 61，忽略最初 60 帧的分类判决。

稳定判决 FSM：非手持转手持使用 `p(handheld)>=0.95`、60 帧窗口；手持转非手持使用 `p(handheld)<=0.1`、35 帧窗口；窗口需至少 90% 帧为对应强证据且末帧符合。尚未确认的状态显示为 `unknown`。

离线评估先裁头 3 秒，统计时忽略后续 60 帧；手机实时推理从开始检测时持续接收数据，仅使用模型 60 帧预热，不执行离线裁头。
