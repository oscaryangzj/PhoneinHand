"""固定种子的空间旋转增强，保留 R10 原有窗口与副本顺序。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from . import config
from .dataset import ImuWindowDataset, NormStats, iter_recordings

def sample_rotations(count: int, seed: int) -> np.ndarray:
    """均匀单位四元数对应 Haar-uniform SO(3)，四元数顺序为 x,y,z,w。"""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], axis=-1).reshape(count, 3, 3).astype(np.float32)

def rotate_imu(raw: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """在物理单位下，对整段 acc / gyro 使用同一个固定旋转。"""
    result = np.empty_like(raw)
    result[:, :3] = raw[:, :3] @ rotation.T
    result[:, 3:] = raw[:, 3:] @ rotation.T
    return result

class RotationCopiesDataset(Dataset):
    """每个基础窗口保留原样本，另有固定种子的 N 个旋转版本。"""

    def __init__(self, base: ImuWindowDataset, raw_windows: np.ndarray,
                 stats: NormStats, rotation_copies: int = 15, seed: int = 42):
        if len(base) != len(raw_windows) or rotation_copies < 1:
            raise ValueError("raw windows 与基础窗口必须一致，旋转次数必须为正")
        self.base = base
        self.raw_windows = raw_windows
        self.stats = stats
        self.rotation_copies = rotation_copies
        self.copies = rotation_copies + 1
        self.rotations = sample_rotations(len(base) * rotation_copies, seed).reshape(
            len(base), rotation_copies, 3, 3
        )
        self.label_counts = base.label_counts * self.copies

    def __len__(self) -> int:
        return len(self.base) * self.copies

    def __getitem__(self, idx: int) -> dict:
        base_idx, copy_idx = divmod(idx, self.copies)
        item = self.base[base_idx]
        if copy_idx:
            raw = rotate_imu(self.raw_windows[base_idx], self.rotations[base_idx, copy_idx - 1])
            item["imu"] = torch.from_numpy(self.stats.normalize(raw))
        return item

class CombinedDataset(ConcatDataset):
    """将旋转增强本地数据与未增强SmartRotation数据合并。"""

    def __init__(self, datasets):
        super().__init__(datasets)
        self.label_counts = sum(
            (dataset.label_counts for dataset in datasets),
            np.zeros(len(config.LABEL_NAMES), dtype=np.int64),
        )

def read_training_windows(directory: Path, base: ImuWindowDataset, stride: int) -> np.ndarray:
    """按 ImuWindowDataset 的排序与滑窗规则，加载待旋转的原始物理单位窗。"""
    raw_windows = np.empty(
        (len(base), base.window_size, len(config.FEATURE_COLS)), dtype=np.float32
    )
    offset = 0
    for path, raw, _, _ in iter_recordings(directory):
        starts = range(0, len(raw) - base.window_size + 1, stride)
        count = len(starts)
        if base.meta[offset][0] != path.name:
            raise ValueError(f"训练窗口排序与原始数据不一致: {path.name}")
        for index, start in enumerate(starts):
            raw_windows[offset + index] = raw[start:start + base.window_size]
        offset += count
    if offset != len(base):
        raise ValueError(f"原始旋转窗口数量{offset}与基础训练窗口数量{len(base)}不一致")
    return raw_windows
