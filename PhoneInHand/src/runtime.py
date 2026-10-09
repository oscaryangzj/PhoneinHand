"""跨平台设备选择与配置读取。"""
from __future__ import annotations

import json
from pathlib import Path
import platform

import torch

from . import config


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"交接清单必须使用目录内的相对路径：{relative}")
    return config.PROJECT_ROOT / path


def resolve_device(requested: str, *, training: bool = False) -> torch.device:
    if requested == "auto":
        requested = "mps" if training and platform.system() == "Darwin" else (
            "cuda" if torch.cuda.is_available() else "cpu")
    if training and platform.system() == "Darwin" and requested != "mps":
        raise RuntimeError("macOS 训练必须显式使用 --device mps")
    device = torch.device(requested)
    if device.type == "mps" and not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        raise RuntimeError("MPS 不可用，停止执行；不自动改用 CPU")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请在支持的平台明确选择其他设备")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
