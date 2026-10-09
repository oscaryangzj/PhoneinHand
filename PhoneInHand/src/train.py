"""用固定的数据清单与配方，从随机初始化重训 R10。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader

from . import config
from .augmentation import CombinedDataset, RotationCopiesDataset, read_training_windows
from .dataset import ImuSequenceDataset, ImuWindowDataset, NormStats
from .metrics import _masked_ce, evaluate_streaming
from .model import CausalCNN, model_config_dict
from .prepare import prepare_dataset
from .runtime import package_path, read_json, resolve_device, synchronize, write_json
from .temporal_loss import same_label_tmse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_to_tensors(batch, device):
    return batch["imu"].to(device), batch["labels"].to(device), batch["mask"].to(device)


def build_dataset(manifest: dict, stats: NormStats, recipe: dict) -> CombinedDataset:
    datasets = []
    for part in manifest["parts"]:
        base = ImuWindowDataset(part["directory"], stats, window_size=recipe["window_size"],
                               stride=part["stride"], ignore_warmup=manifest["warmup_frames"])
        if len(base) != part["base_windows"]:
            raise ValueError(f"基础窗口数不一致：{part['name']}")
        dataset = base
        if part["rotation_copies"]:
            raw = read_training_windows(part["directory"], base, part["stride"])
            dataset = RotationCopiesDataset(base, raw, stats, part["rotation_copies"], recipe["rotation_seed"])
        if len(dataset) != part["training_windows"]:
            raise ValueError(f"增强窗口数不一致：{part['name']}")
        datasets.append(dataset)
        print(f"DATASET {part['name']}: stride={part['stride']} "
              f"rotation_copies={part['rotation_copies']} windows={len(dataset):,}", flush=True)
    combined = CombinedDataset(datasets)
    if (len(combined) != recipe["training_windows"]
            or combined.label_counts.tolist() != recipe["effective_class_frames"]):
        raise ValueError("总训练窗口数或类别帧数与 R10 配方不一致")
    return combined


def class_weights(dataset, device: torch.device) -> torch.Tensor:
    counts = np.asarray(dataset.label_counts, dtype=np.float32)
    weights = 1. / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(model, loader, optimizer, device, class_weight,
                temporal_loss_weight=0.15, temporal_loss_tau=4.0, grad_clip=1.0):
    """保留原 R10 的损失、优化步骤与 GPU 指标累加顺序。"""
    model.train()
    totals = torch.zeros(7, dtype=torch.float32, device=device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        imu, labels, mask = _batch_to_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imu)
        ce, weight = _masked_ce(logits, labels, mask, class_weight)
        tmse, pairs = same_label_tmse(logits, labels, mask, temporal_loss_tau)
        loss = ce + temporal_loss_weight * tmse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            valid = mask > .5
            correct = ((logits.argmax(dim=-1) == labels) & valid).sum()
            totals += torch.stack([loss.detach() * weight, ce.detach() * weight,
                tmse.detach() * pairs, pairs, weight, correct, valid.sum()]).to(torch.float32)
        if batch_index % 2000 == 0:
            current = totals.cpu().tolist()
            if not np.isfinite(current).all():
                raise FloatingPointError("训练指标非有限值")
            print(f"BATCH {batch_index}/{len(loader)} loss={current[0]/current[4]:.5f} "
                  f"seconds={time.perf_counter()-started:.1f}", flush=True)
    loss_sum, ce_sum, tmse_sum, pairs, weight, correct, frames = totals.cpu().tolist()
    if not np.isfinite([loss_sum, ce_sum, tmse_sum, pairs, weight, correct, frames]).all():
        raise FloatingPointError("训练指标非有限值")
    return {"loss": loss_sum / weight, "ce": ce_sum / weight,
            "tmse": tmse_sum / pairs if pairs else 0., "temporal_pairs": pairs, "acc": correct / frames}


def train(recipe: dict, output: Path, device: torch.device) -> None:
    manifest = prepare_dataset(package_path(recipe["dataset_manifest"]))
    if (manifest["model_id"] != recipe["model_id"]
            or manifest["training_recordings"] != recipe["training_recordings"]
            or manifest["validation_recordings"] != recipe["validation_recordings"]
            or manifest["trim_seconds"] != recipe["trim_seconds"]
            or manifest["warmup_frames"] != recipe["warmup_frames_after_trim"]):
        raise ValueError("数据清单与训练配方不一致")
    stats = NormStats.load(package_path(recipe["normalization"]))
    dataset = build_dataset(manifest, stats, recipe)
    valid = ImuSequenceDataset(manifest["validation_directory"], stats, manifest["warmup_frames"])
    set_seed(recipe["seed"])
    model = CausalCNN(**recipe["model"]).to(device)
    loader = DataLoader(dataset, batch_size=recipe["batch_size"], shuffle=True, num_workers=0, drop_last=False)
    weights = class_weights(dataset, device) if recipe["class_weights"] else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"])
    output.mkdir(parents=True, exist_ok=False)
    stats.save(output / "norm_stats.json")
    write_json(output / "experiment_config.json", {**recipe, "device": str(device)})
    history, best_f1 = [], -1.0
    print(f"START windows={len(dataset):,} batches/epoch={len(loader):,} device={device}", flush=True)
    for epoch in range(1, recipe["epochs"] + 1):
        started = time.perf_counter()
        tr = train_epoch(model, loader, optimizer, device, weights,
            recipe["temporal_loss"]["weight"], recipe["temporal_loss"]["tau"], recipe["grad_clip"])
        scheduler.step()
        metrics = evaluate_streaming(model, valid, device, weights)
        tn, fp, _, _ = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=[0, 1]).reshape(-1)
        synchronize(device)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
            "train_ce": tr["ce"], "train_tmse": tr["tmse"], "train_temporal_pairs": tr["temporal_pairs"],
            "valid_loss": metrics["loss"], "valid_acc": metrics["acc"], "valid_f1": metrics["f1"],
            "valid_false_free_rate": float(fp / (tn + fp)), "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": time.perf_counter() - started}
        history.append(row)
        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(), "config": model_config_dict(model,
                extra={"feature_cols": config.FEATURE_COLS, "label_map": config.LABEL_MAP,
                       "window_size": recipe["window_size"]}),
                "norm_stats": asdict(stats), "epoch": epoch, "valid_f1": best_f1,
                "valid_acc": metrics["acc"], "experiment": recipe}, output / "causal_cnn_best.pt")
        write_json(output / "train_history.json", history)
        print(f"[{epoch:02d}/{recipe['epochs']}] train={tr['acc']:.4%} "
              f"valid={metrics['acc']:.4%} FFR={row['valid_false_free_rate']:.4%}", flush=True)
    print(f"DONE best validation macro-F1={best_f1:.6f}; checkpoint={output / 'causal_cnn_best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=Path, default=config.ROOT / "runs/R10_retrain")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"请保留已有结果，并使用新的输出目录：{args.output_dir}")
    device = resolve_device(args.device, training=True)
    train(read_json(config.RECIPE_PATH), args.output_dir, device)


if __name__ == "__main__":
    main()
