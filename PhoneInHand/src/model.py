"""纯流式 Causal CNN 手持分类（对齐 SmartRotation CNN 前端，无 GRU）。

输入端含因果去重力（加速度 EMA），突出微抖、弱化放置角度；
训练/推理共享 ``step_chunk`` 延迟线 + 重力状态。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGravityHighPass(nn.Module):
    """因果去重力：g_t = α·g_{t-1} + (1-α)·a_t，输出 a_t - g_t；陀螺仪原样通过。"""

    def __init__(self, alpha: float = 0.97, acc_channels: int = 3):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"gravity alpha must be in (0,1), got {alpha}")
        self.acc_channels = int(acc_channels)
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    @property
    def alpha_value(self) -> float:
        return float(self.alpha.item())

    def forward(
        self,
        x: torch.Tensor,
        grav_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          [B, T, F]  F>=acc_channels，前若干维为加速度
            grav_state: [B, A] 上一帧重力估计；None 时用首帧加速度初始化
        Returns:
            y:     [B, T, F] 线性加速度 + 原陀螺
            g_last:[B, A]
        """
        if x.dim() != 3:
            raise ValueError(f"gravity expects [B,T,F], got {tuple(x.shape)}")
        b, t, _ = x.shape
        a_ch = self.acc_channels
        acc = x[:, :, :a_ch]
        gyro = x[:, :, a_ch:]
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)

        if grav_state is None:
            g = acc[:, 0, :]
        else:
            if grav_state.shape != (b, a_ch):
                raise ValueError(
                    f"grav_state expected {(b, a_ch)}, got {tuple(grav_state.shape)}"
                )
            g = grav_state.to(dtype=x.dtype, device=x.device)

        lin_frames: list[torch.Tensor] = []
        for i in range(t):
            a_t = acc[:, i, :]
            g = alpha * g + (1.0 - alpha) * a_t
            lin_frames.append(a_t - g)
        lin = torch.stack(lin_frames, dim=1)
        y = torch.cat([lin, gyro], dim=-1)
        return y, g


class CausalConv1d(nn.Module):
    """严格因果 1D 卷积（仅左填充）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.left_pad = (kernel_size - 1) * dilation

    @property
    def delay_len(self) -> int:
        return self.left_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)

    def step_chunk(
        self,
        x: torch.Tensor,
        delay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:         [B, C_in, T]
        delay:     [B, C_in, delay_len]
        y:         [B, C_out, T]
        delay_new: [B, C_in, delay_len]
        """
        full = torch.cat([delay, x], dim=-1)
        y = self.conv(full)
        delay_new = full[:, :, -delay.shape[-1] :]
        return y, delay_new


class CausalConvBlock(nn.Module):
    """Depthwise-separable causal block: DW + PW + BN + GELU。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.dw = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation, groups=channels
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    @property
    def delay_len(self) -> int:
        return self.dw.delay_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        return self.act(self.norm(y))

    def step_chunk(
        self,
        x: torch.Tensor,
        delay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, delay_new = self.dw.step_chunk(x, delay)
        y = self.pw(y)
        y = self.act(self.norm(y))
        return y, delay_new


class CausalCNN(nn.Module):
    """流式 Causal CNN：去重力 + 逐帧二分类 logits。"""

    def __init__(
        self,
        input_dim: int = 6,
        num_classes: int = 2,
        cnn_channels: int = 32,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
        step_chunk_size: int = 1,
        gravity_alpha: float = 0.97,
        acc_channels: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.acc_channels = int(acc_channels)
        if step_chunk_size < 1:
            raise ValueError(f"step_chunk_size must be >= 1, got {step_chunk_size}")
        self.step_chunk_size = int(step_chunk_size)

        self.gravity = CausalGravityHighPass(
            alpha=gravity_alpha, acc_channels=self.acc_channels
        )
        self.input_proj = nn.Linear(input_dim, cnn_channels)
        self.cnn = nn.ModuleList(
            [CausalConvBlock(cnn_channels, kernel_size, d) for d in self.dilations]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(cnn_channels, cnn_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels // 2, num_classes),
        )

        self._receptive_field = 1 + sum((kernel_size - 1) * d for d in self.dilations)
        self._cnn_state_len = sum((kernel_size - 1) * d for d in self.dilations)

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def cnn_state_len(self) -> int:
        return self._cnn_state_len

    @property
    def gravity_alpha(self) -> float:
        return self.gravity.alpha_value

    def _cnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F]（已去重力）-> cnn features [B, T, C]."""
        h = self.input_proj(x).transpose(1, 2)  # [B, C, T]
        for block in self.cnn:
            h = block(h)
        return self.dropout(h.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] 标准化后的 IMU（模型内再去重力）
        Returns:
            logits: [B, T, num_classes]
        """
        x, _ = self.gravity(x, None)
        feats = self._cnn_forward(x)
        return self.head(feats)

    def step_chunk(
        self,
        x_chunk: torch.Tensor,
        cnn_state: torch.Tensor | None,
        grav_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        流式步进。

        Args:
            x_chunk:    [B, T, F]
            cnn_state:  [B, C, S]  packed delay lines（None → 全零）
            grav_state: [B, A]    重力 EMA 状态（None → 用 chunk 首帧加速度初始化）

        Returns:
            logits:     [B, T, num_classes]
            state_new:  [B, C, S]
            grav_new:   [B, A]
        """
        if x_chunk.dim() != 3:
            raise ValueError(f"step_chunk expects [B,T,F], got {tuple(x_chunk.shape)}")
        b, t, f = x_chunk.shape
        if t < 1:
            raise ValueError(f"step_chunk expects T>=1, got T={t}")
        if f != self.input_dim:
            raise ValueError(f"step_chunk expects F={self.input_dim}, got F={f}")

        x_chunk, grav_new = self.gravity(x_chunk, grav_state)
        x = self.input_proj(x_chunk).transpose(1, 2)  # [B, C, T]
        c = x.shape[1]
        s = self._cnn_state_len
        if cnn_state is None:
            cnn_state = torch.zeros(b, c, s, device=x_chunk.device, dtype=x_chunk.dtype)

        offset = 0
        new_parts: list[torch.Tensor] = []
        for block in self.cnn:
            delay_len = block.delay_len
            delay = cnn_state[:, :, offset : offset + delay_len]
            x, delay_new = block.step_chunk(x, delay)
            new_parts.append(delay_new)
            offset += delay_len

        state_new = torch.cat(new_parts, dim=-1)
        feats = self.dropout(x.transpose(1, 2))
        return self.head(feats), state_new, grav_new


def build_model(**kwargs) -> CausalCNN:
    return CausalCNN(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_config_dict(
    model: CausalCNN,
    *,
    extra: dict | None = None,
) -> dict:
    cfg = {
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "cnn_channels": model.cnn_channels,
        "kernel_size": model.kernel_size,
        "dilations": list(model.dilations),
        "dropout": float(model.dropout.p),
        "step_chunk_size": model.step_chunk_size,
        "receptive_field": model.receptive_field,
        "cnn_state_len": model.cnn_state_len,
        "gravity_alpha": model.gravity_alpha,
        "acc_channels": model.acc_channels,
    }
    if extra:
        cfg.update(extra)
    return cfg


if __name__ == "__main__":
    m = CausalCNN().eval()
    x = torch.randn(2, 128, 6)
    # 模拟带重力的加速度
    x = x.clone()
    x[:, :, 2] += 9.8
    y = m(x)
    print(
        f"params={count_parameters(m)} RF={m.receptive_field} "
        f"state={m.cnn_state_len} alpha={m.gravity_alpha} out={tuple(y.shape)}"
    )

    state = None
    grav = None
    outs = []
    for i in range(0, 128, m.step_chunk_size):
        yi, state, grav = m.step_chunk(x[:, i : i + m.step_chunk_size], state, grav)
        outs.append(yi)
    y_stream = torch.cat(outs, dim=1)
    diff = (y - y_stream).abs().max().item()
    print(f"max |forward - step_chunk| = {diff:.2e}")

    x2 = x.clone()
    x2[:, -1] += 10
    with torch.no_grad():
        d = (m(x)[:, :-1] - m(x2)[:, :-1]).abs().max().item()
    print(f"max |Δ| past after future edit = {d:.2e}")
