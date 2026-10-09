"""同标签相邻帧的截断 log-probability MSE（MS-TCN TMSE 改造）。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def same_label_tmse(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 TMSE 和有效帧对数；logits 为 [B,T,C]。

    只连接窗口内两个均有效、标签相同的相邻帧。前一帧停止梯度，
    每类平方差截断到 tau²，再按有效帧对及类别数归一化。
    无有效帧对时返回可反传的零值。训练窗口之间不会连接。
    """
    if tau <= 0:
        raise ValueError("TMSE tau 必须为正")
    log_p = F.log_softmax(logits, dim=-1)
    pairs = ((mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
             & (labels[:, 1:] == labels[:, :-1]))
    count = pairs.sum()
    squared = (log_p[:, 1:] - log_p[:, :-1].detach()).square()
    loss = (squared.clamp(max=tau * tau) * pairs.unsqueeze(-1)).sum()
    loss = loss / (count.clamp(min=1) * logits.shape[-1])
    return loss, count
