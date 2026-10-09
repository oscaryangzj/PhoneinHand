"""逐帧损失与按独立录制的流式评估。"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from .model import CausalCNN

def _masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits/labels/mask: [B, T, ...] → (mean_loss, weight_sum)."""
    b, t, c = logits.shape
    flat_logits = logits.reshape(b * t, c)
    flat_labels = labels.reshape(b * t)
    flat_mask = mask.reshape(b * t)
    per = nn.functional.cross_entropy(
        flat_logits, flat_labels, weight=weight, reduction="none"
    )
    wsum = flat_mask.sum().clamp(min=1.0)
    loss = (per * flat_mask).sum() / wsum
    return loss, wsum

@torch.no_grad()
def evaluate_streaming(
    model: CausalCNN,
    dataset,
    device: torch.device,
    class_weight: torch.Tensor | None = None,
) -> dict:
    """整段录音 step_chunk 评估，与端侧一致。"""
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    total_w = 0.0
    for i in range(len(dataset)):
        item = dataset[i]
        imu = item["imu"].unsqueeze(0).to(device)
        labels = item["labels"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        logits, _, _ = model.step_chunk(imu, None, None)
        loss, wsum = _masked_ce(logits, labels, mask, class_weight)
        total_loss += float(loss.item()) * float(wsum.item())
        total_w += float(wsum.item())
        pred = logits.argmax(dim=-1)
        valid = mask[0] > 0.5
        all_y.append(labels[0][valid].cpu().numpy())
        all_p.append(pred[0][valid].cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.zeros(0, dtype=np.int64)
    y_pred = np.concatenate(all_p) if all_p else np.zeros(0, dtype=np.int64)
    return {
        "loss": total_loss / max(total_w, 1.0),
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
    }
