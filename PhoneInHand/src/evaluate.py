"""评估交接的 R10 checkpoint，报告 accuracy/F1/False Free Rate。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
import torch

from . import config
from .dataset import ImuSequenceDataset, NormStats
from .metrics import evaluate_streaming
from .model import CausalCNN
from .online_debounce import FrameCountDebouncer
from .prepare import prepare_dataset
from .runtime import read_json, resolve_device, write_json


def load_model(path: Path, device: torch.device) -> tuple[CausalCNN, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = CausalCNN(input_dim=cfg["input_dim"], num_classes=cfg["num_classes"],
        cnn_channels=cfg["cnn_channels"], kernel_size=cfg["kernel_size"], dilations=cfg["dilations"],
        dropout=0.0, step_chunk_size=cfg.get("step_chunk_size", 1),
        gravity_alpha=cfg.get("gravity_alpha", config.GRAVITY_ALPHA),
        acc_channels=cfg.get("acc_channels", config.ACC_CHANNELS)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, device: torch.device, manifest: dict) -> dict:
    model, checkpoint = load_model(checkpoint_path, device)
    stats = NormStats(**checkpoint["norm_stats"])
    packaged_stats = NormStats.load(config.MODEL_DIR / "norm_stats.json")
    if stats != packaged_stats:
        raise ValueError("checkpoint 与配套归一化参数不一致")
    dataset = ImuSequenceDataset(manifest["validation_directory"], stats, manifest["warmup_frames"])
    result = evaluate_streaming(model, dataset, device)
    counts = confusion_matrix(result["y_true"], result["y_pred"], labels=[0, 1]).reshape(-1)
    tn, fp, fn, tp = counts.tolist()
    rows, by_scene, offset = [], defaultdict(lambda: np.zeros(4, dtype=np.int64)), 0
    unknown_frames = false_free = false_contact = contact_frames = handheld_frames = 0
    for i in range(len(dataset)):
        item = dataset[i]
        n = int(item["mask"].sum().item())
        confusion = confusion_matrix(result["y_true"][offset:offset+n],
            result["y_pred"][offset:offset+n], labels=[0, 1]).reshape(-1)
        offset += n
        by_scene[item["scene"]] += confusion
        rows.append({"file": item["path"], "scene": item["scene"], "frames": n,
                     "confusion_matrix_tn_fp_fn_tp": confusion.tolist()})
        logits, _, _ = model.step_chunk(item["imu"].unsqueeze(0).to(device), None, None)
        probabilities = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
        debouncer = FrameCountDebouncer()
        for p in probabilities[manifest["warmup_frames"]:]:
            state = debouncer.update(float(p))
            unknown_frames += state == -1
            is_contact = item["file_label"] == 0
            contact_frames += is_contact
            handheld_frames += not is_contact
            false_free += is_contact and state == 1
            false_contact += not is_contact and state == 0
    return {"model_id": manifest["model_id"], "recordings": len(dataset), "frames": len(result["y_true"]),
        "accuracy": result["acc"], "macro_f1": result["f1"],
        "false_free_rate": fp / (tn + fp) if tn + fp else None,
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
        "scene_confusions_tn_fp_fn_tp": {k: v.tolist() for k, v in by_scene.items()},
        "per_recording": rows,
        "fsm": {"unknown_frames": int(unknown_frames), "false_free_frames": int(false_free),
            "false_free_rate": false_free / contact_frames if contact_frames else None,
            "false_contact_frames": int(false_contact),
            "handheld_miss_rate": false_contact / handheld_frames if handheld_frames else None,
            "unknown_policy": "unknown 单独统计，不计作正确或错误；错误率分母包含所有真实状态帧"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_DIR / "causal_cnn_best.pt")
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = evaluate_checkpoint(args.checkpoint, resolve_device(args.device), prepare_dataset(args.manifest))
    if args.output:
        write_json(args.output, report)
    for key in ("model_id", "recordings", "frames", "accuracy", "macro_f1", "false_free_rate",
                "confusion_matrix_tn_fp_fn_tp", "fsm"):
        print(f"{key}: {report[key]}")


if __name__ == "__main__":
    main()
