"""IMU 录制读取、标准化和训练滑窗。"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from . import config

@dataclass
class NormStats:
    mean: list[float]
    std: list[float]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (x - mean) / np.clip(std, 1e-6, None)

def load_recording(csv_path: Path) -> tuple[np.ndarray, int, str]:
    df = pd.read_csv(csv_path)
    feats = df[config.FEATURE_COLS].to_numpy(dtype=np.float32)
    label_name = str(df["hold_state"].iloc[0])
    if label_name not in config.LABEL_MAP:
        raise ValueError(f"unknown hold_state={label_name} in {csv_path}")
    label = config.LABEL_MAP[label_name]
    scene = str(df["scene_tag"].iloc[0]) if "scene_tag" in df.columns else ""
    return feats, label, scene

def iter_recordings(data_dir: Path) -> Iterable[tuple[Path, np.ndarray, int, str]]:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no csv found in {data_dir}")
    for p in paths:
        feats, label, scene = load_recording(p)
        yield p, feats, label, scene

class ImuSequenceDataset(Dataset):
    """整段录音，逐帧标签（文件级标签广播到每一帧）。"""

    def __init__(
        self,
        data_dir: Path,
        stats: NormStats,
        ignore_warmup: int = 0,
    ):
        self.items: list[dict] = []
        self.label_counts = np.zeros(len(config.LABEL_NAMES), dtype=np.int64)
        for path, feats, label, scene in iter_recordings(data_dir):
            feats = stats.normalize(feats)
            n = len(feats)
            labels = np.full((n,), label, dtype=np.int64)
            mask = np.ones((n,), dtype=np.float32)
            if ignore_warmup > 0:
                warm = min(ignore_warmup, n)
                mask[:warm] = 0.0
            self.items.append(
                {
                    "path": path.name,
                    "scene": scene,
                    "imu": feats,
                    "labels": labels,
                    "mask": mask,
                    "file_label": label,
                }
            )
            self.label_counts[label] += int(mask.sum())

        if not self.items:
            raise RuntimeError(f"no sequences from {data_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "path": item["path"],
            "scene": item["scene"],
            "imu": torch.from_numpy(item["imu"]),
            "labels": torch.from_numpy(item["labels"]),
            "mask": torch.from_numpy(item["mask"]),
            "file_label": item["file_label"],
        }

class ImuWindowDataset(Dataset):
    """滑窗样本：imu [T, F]，逐帧标签与 mask（窗内可忽略 warmup）。"""

    def __init__(
        self,
        data_dir: Path,
        stats: NormStats,
        window_size: int = config.WINDOW_SIZE,
        stride: int = config.STRIDE,
        ignore_warmup: int = 0,
    ):
        self.window_size = window_size
        xs, ys, masks, meta = [], [], [], []
        label_counts = np.zeros(len(config.LABEL_NAMES), dtype=np.int64)
        for path, feats, label, scene in iter_recordings(data_dir):
            feats = stats.normalize(feats)
            n = len(feats)
            if n < window_size:
                continue
            mask_win = np.ones((window_size,), dtype=np.float32)
            if ignore_warmup > 0:
                mask_win[: min(ignore_warmup, window_size)] = 0.0
            for s in range(0, n - window_size + 1, stride):
                xs.append(feats[s : s + window_size])
                ys.append(np.full((window_size,), label, dtype=np.int64))
                masks.append(mask_win.copy())
                meta.append((path.name, scene, label))
                label_counts[label] += int(mask_win.sum())
        if not xs:
            raise RuntimeError(f"no windows from {data_dir}")
        self.x = np.stack(xs, axis=0)
        self.y = np.stack(ys, axis=0)
        self.mask = np.stack(masks, axis=0)
        self.meta = meta
        self.label_counts = label_counts

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict:
        return {
            "imu": torch.from_numpy(self.x[idx]),
            "labels": torch.from_numpy(self.y[idx]),
            "mask": torch.from_numpy(self.mask[idx]),
            "file_label": self.meta[idx][2],
            "path": self.meta[idx][0],
            "scene": self.meta[idx][1],
        }
