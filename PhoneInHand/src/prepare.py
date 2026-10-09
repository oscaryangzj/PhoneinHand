"""按固定 session 清单生成 R10 的裁剪数据，保留全部原始 CSV。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config
from .runtime import package_path, read_json


def prepare_dataset(manifest_path: Path = config.MANIFEST_PATH,
                    cache_dir: Path = config.CACHE_DIR) -> dict:
    manifest = read_json(manifest_path)
    train = [r for part in manifest["parts"] for r in part["recordings"]]
    valid = manifest["validation"]
    train_paths = [r["path"] for r in train]
    valid_paths = [r["path"] for r in valid]
    if (len(set(train_paths)) != len(train_paths) or len(set(valid_paths)) != len(valid_paths)
            or set(train_paths) & set(valid_paths)):
        raise ValueError("训练/验证必须按完整 recording 隔离，不能重复或重叠")
    if len(train) != manifest["training_recordings"] or len(valid) != manifest["validation_recordings"]:
        raise ValueError("录制数量与清单不一致")

    def crop(records: list[dict], directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        expected_names = {Path(r["path"]).name for r in records}
        if len(expected_names) != len(records):
            raise ValueError("同一数据分组存在重复文件名")
        if {p.name for p in directory.glob("*.csv")} - expected_names:
            raise ValueError(f"缓存含清单之外的 CSV，请换一个缓存目录：{directory}")
        for record in records:
            source = package_path(record["path"])
            with source.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            header, samples = rows[0], rows[1:]
            required = ["timestamp_ms", "hold_state", "scene_tag", *config.FEATURE_COLS]
            if not all(column in header for column in required) or not samples:
                raise ValueError(f"缺少必要字段或录制为空：{source}")
            if any(len(row) != len(header) for row in samples):
                raise ValueError(f"CSV 字段数不一致：{source}")
            ts = np.asarray([float(row[header.index("timestamp_ms")]) for row in samples])
            imu = np.asarray([[float(row[header.index(c)]) for c in config.FEATURE_COLS] for row in samples])
            holds = {row[header.index("hold_state")] for row in samples}
            scenes = {row[header.index("scene_tag")] for row in samples}
            if (not np.isfinite(ts).all() or not np.isfinite(imu).all() or not (np.diff(ts) > 0).all()
                    or len(holds) != 1 or not holds <= config.LABEL_MAP.keys() or len(scenes) != 1):
                raise ValueError(f"时间戳、IMU 或标签不符合录制要求：{source}")
            first = int(np.searchsorted(ts, ts[0] + manifest["trim_seconds"] * 1000, side="left"))
            retained = samples[first:]
            if len(retained) != record["retained_rows"] or len(retained) < config.WINDOW_SIZE:
                raise ValueError(f"裁剪后的录制长度与 R10 清单不一致：{source}")
            target = directory / source.name
            expected = [header, *retained]
            if target.exists():
                with target.open(encoding="utf-8-sig", newline="") as stream:
                    if list(csv.reader(stream)) != expected:
                        raise ValueError(f"缓存与原始录制不一致，请换一个缓存目录：{target}")
            else:
                with target.open("x", encoding="utf-8", newline="") as stream:
                    csv.writer(stream).writerows(expected)

    parts = []
    for part in manifest["parts"]:
        directory = cache_dir / "train" / part["name"]
        crop(part["recordings"], directory)
        parts.append({**part, "directory": directory})
    valid_dir = cache_dir / "validation"
    crop(valid, valid_dir)
    return {**manifest, "parts": parts, "validation_directory": valid_dir}
