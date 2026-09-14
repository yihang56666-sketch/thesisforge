# -*- coding: utf-8 -*-
"""轻量 U-Net 语义分割训练脚本（独立子进程运行，依赖 PyTorch）。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from app.networks import build_model


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe(base: Path, name: str) -> Path:
    base = base.resolve()
    p = (base / name).resolve()
    if ".." in p.parts or not p.is_relative_to(base):
        raise ValueError("非法路径")
    return p


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    root = dataset_dir
    if not (root / "images").exists() and len([d for d in root.iterdir() if d.is_dir()]) == 1:
        root = next(d for d in root.iterdir() if d.is_dir())
    pairs = []
    for img in sorted((root / "images").rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        rel = img.relative_to(root / "images")
        mask = root / "masks" / rel.with_suffix(".png")
        if not mask.exists():
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = root / "masks" / rel.with_suffix(ext)
                if candidate.exists():
                    mask = candidate
                    break
        if mask.exists():
            pairs.append((img, mask))
    return pairs


def tensor_from(img_path: Path, mask_path: Path, size: int, num_classes: int):
    from torchvision import transforms

    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    x = transforms.ToTensor()(img)
    y = torch.as_tensor(np.array(mask), dtype=torch.long)
    return x, y


def calculate_iou(pred: np.ndarray, true: np.ndarray, num_classes: int) -> tuple[float, float]:
    intersections, unions = [], []
    for cls in range(1, num_classes):
        p = pred == cls
        t = true == cls
        intersection = int((p & t).sum())
        union = int((p | t).sum())
        intersections.append(intersection)
        unions.append(union)
    iou = [i / u for i, u in zip(intersections, unions) if u]
    dice = [2 * i / (i + u) for i, u in zip(intersections, unions) if i + u]
    return float(np.mean(iou)) if iou else 0.0, float(np.mean(dice)) if dice else 0.0


def save_prediction(img_path: Path, out_path: Path, model, device, size: int) -> None:
    from torchvision import transforms

    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(1)[0].cpu().numpy()
    pred = (pred * max(1, 255 // max(2, int(pred.max()) + 1))).astype(np.uint8)
    Image.fromarray(pred).save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)
        write_json(safe(run_dir, "metrics.jsonl"), events)

    t0 = time.time()
    try:
        from torch.utils.data import DataLoader, TensorDataset

        cfg = json.loads(safe(run_dir, "config.json").read_text(encoding="utf-8"))
        params = cfg.get("params") or {}
        size = int(params.get("imgsz", 256))
        classes = cfg.get("classes") or ["0", "1"]
        num_classes = max(2, len(classes))
        dataset_dir = Path(cfg["dataset_dir"])
        pairs = load_pairs(dataset_dir)
        if len(pairs) < 3:
            raise ValueError("语义分割数据量过小，无法划分训练/验证/测试集")

        seed = int(params.get("seed", 42))
        random.Random(seed).shuffle(pairs)
        n_test = max(1, int(round(len(pairs) * 0.2)))
        n_val = max(1, int(round(len(pairs) * 0.2)))
        n_train = len(pairs) - n_val - n_test
        if n_train < 1:
            raise ValueError("训练集为空，请增加数据或降低验证/测试比例")
        splits = (pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:])

        def ds(items):
            return TensorDataset(*[
                torch.stack([tensor_from(i, m, size, num_classes)[0] for i, m in items]),
                torch.stack([tensor_from(i, m, size, num_classes)[1] for i, m in items]),
            ])

        loaders = [
            DataLoader(ds(items), batch_size=int(params.get("batch_size", 8)), shuffle=True)
            for items in splits
        ]
        device = torch.device("cuda" if params.get("device", "auto") == "gpu"
                              or (params.get("device", "auto") == "auto" and torch.cuda.is_available())
                              else "cpu")
        model = build_model("semantic_segmentation", "unet", params, num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 0.001)),
                                      weight_decay=float(params.get("weight_decay", 0.0001)))
        emit({"type": "split", "n_train": n_train, "n_val": n_val, "n_test": n_test,
              "n_images": len(pairs), "classes": classes, "seed": seed})
        epoch_logs = []
        for epoch in range(1, int(params.get("epochs", 30)) + 1):
            model.train()
            total_loss = 0.0
            for x, y in loaders[0]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
            train_loss = total_loss / max(len(loaders[0]), 1)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in loaders[1]:
                    val_loss += float(criterion(model(x.to(device)), y.to(device)))
            val_loss /= max(len(loaders[1]), 1)
            rec = {"type": "epoch", "epoch": epoch,
                   "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
                   "lr": float(optimizer.param_groups[0]["lr"])}
            events.append(rec)
            write_json(safe(run_dir, "metrics.jsonl"), events)
            epoch_logs.append(rec)
            log(f"epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        torch.save({
            "task": "semantic_segmentation", "model": "unet", "params": params,
            "state_dict": model.state_dict(), "num_classes": num_classes,
            "image_size": size, "classes": classes, "device": str(device),
        }, safe(run_dir, "best.pt"))

        # 评估最后一批样本，避免额外推理开销；指标仍为像素级 IoU/Dice
        x, y = next(iter(loaders[2]))
        x, y = x.to(device), y.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(1).cpu().numpy()
        true = y.cpu().numpy()
        ious, dices, accs = [], [], []
        for p_, t_ in zip(pred, true):
            iou, dice = calculate_iou(p_, t_, num_classes)
            ious.append(iou)
            dices.append(dice)
            accs.append(float((p_ == t_).mean()))
        iou, dice, pixel_acc = float(np.mean(ious)), float(np.mean(dices)), float(np.mean(accs))
        metrics = {"iou": iou, "dice": dice, "pixel_accuracy": pixel_acc}
        emit({"type": "test", **metrics})
        save_prediction(splits[2][0][0], safe(run_dir, "segmentation_prediction.png"), model, device, size)
        summary = {
            "task": "semantic_segmentation", "model": "unet",
            "model_label": cfg.get("model_label", "U-Net 语义分割"),
            "dataset_name": cfg.get("dataset_name"), "params": params, "engine": "torch",
            "classes": classes, "num_classes": num_classes, "metrics": metrics,
            "primary_metric": {"name": "iou", "value": iou},
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "split_scheme": "random train/val/test", "epochs": epoch_logs,
            "train_time_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifacts": ["best.pt", "metrics.jsonl", "segmentation_prediction.png"],
        }
        write_json(safe(run_dir, "summary.json"), summary)
        emit({"type": "summary", "primary_metric": summary["primary_metric"], "metrics": metrics})
        write_json(safe(run_dir, "status.json"),
                   {"state": "done", "error": None, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 0
    except Exception:
        err = traceback.format_exc()
        log("训练失败:\n" + err)
        low = err.lower()
        if "out of memory" in low or "cuda oom" in low:
            kind, hint = "oom", "显存/内存不足。建议减小 batch_size 或 imgsz，或改用 CPU。"
        elif "no space left" in low or ("disk" in low and "space" in low):
            kind, hint = "disk", "磁盘空间不足，请清理实验输出目录后重试。"
        else:
            kind, hint = "error", "训练异常，具体原因见下方错误信息与运行日志。"
        write_json(safe(run_dir, "status.json"),
                   {"state": "failed", "error": err[-1500:], "failure_kind": kind,
                    "failure_hint": hint, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 1


if __name__ == "__main__":
    sys.exit(main())
