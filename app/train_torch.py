"""图像分类训练脚本（PyTorch，独立子进程运行）。

用法: python app/train_torch.py --run-dir data/runs/<run_id>
数据组织为 ImageFolder 布局: dataset_dir/images/<类别名>/<图片>.
产出 metrics.jsonl（逐轮曲线）/ summary.json / best.pt / 图表 / status.json。
"""
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision import datasets, models, transforms  # noqa: E402

from app.plots import plot_confusion_matrix, plot_training_curves, plot_class_balance  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe(base: Path, name: str) -> Path:
    base = base.resolve()
    t = base / name
    if ".." in t.parts or not t.resolve().is_relative_to(base):
        raise ValueError("非法路径")
    return t.resolve()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model(model_key: str, num_classes: int, pretrained: bool):
    if model_key == "cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
    if model_key == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights

            net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            if pretrained:
                log("已加载 ImageNet 预训练权重（迁移学习）")
        except Exception as e:
            log(f"预训练权重下载失败({e.__class__.__name__})，改用随机初始化")
            net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    raise ValueError(f"未知图像模型: {model_key}")


@torch.no_grad()
def evaluate(net, loader, criterion, device):
    """在给定数据上评估：返回 (平均 loss, 准确率, 预测列表, 标签列表)。"""
    net.eval()
    loss_sum = correct = total = 0
    preds: list[int] = []
    labels: list[int] = []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = net(xb)
        bs = yb.size(0)
        loss_sum += criterion(out, yb).item() * bs
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += bs
        preds.extend(pred.cpu().tolist())
        labels.extend(yb.cpu().tolist())
    return loss_sum / max(total, 1), correct / max(total, 1), preds, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()

    cfg = json.loads(safe(run_dir, "config.json").read_text(encoding="utf-8"))
    model_key = cfg["model"]
    params = cfg.get("params") or {}
    seed = int(cfg.get("random_state", 42))
    epochs = int(params.get("epochs", 15))
    batch_size = int(params.get("batch_size", 32))
    lr = float(params.get("lr", 0.001))
    image_size = int(params.get("image_size", 64))
    pretrained = bool(params.get("pretrained", True))
    val_split = float(cfg.get("val_split", 0.2))
    test_split = float(cfg.get("test_size", 0.2))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)
        safe(run_dir, "metrics.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8"
        )

    t0 = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            log(f"使用 GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            log("未检测到 CUDA，使用 CPU 训练（图像任务建议配置 GPU）")

        img_root = safe(Path(cfg["dataset_dir"]).resolve(), "images")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        full = datasets.ImageFolder(str(img_root))
        classes = full.classes
        num_classes = len(classes)
        counts = {classes[ci]: 0 for ci in range(num_classes)}
        for _p, ci in full.samples:
            counts[classes[ci]] += 1
        log(f"数据集: {len(full)} 张图像, {num_classes} 类 {classes}")
        plot_class_balance(counts, safe(run_dir, "class_balance.png"))
        emit({"type": "stage", "name": "load", "n_images": len(full), "classes": classes})

        # 三方划分：训练集拟合权重，验证集挑模型，测试集只在最后用一次
        n_total = len(full)
        n_val = max(2, int(round(n_total * val_split)))
        n_test = max(2, int(round(n_total * test_split))) if test_split > 0 else 0
        n_train = n_total - n_val - n_test
        if n_train < 2:
            # 数据太少时按比例回缩，保证三份都非空
            n_val = max(1, n_total // 5)
            n_test = max(1, n_total // 5) if test_split > 0 else 0
            n_train = n_total - n_val - n_test
        if n_train < 1:
            raise ValueError(f"数据量过小（{n_total} 张），无法同时划分训练/验证/测试集")
        g = torch.Generator().manual_seed(seed)
        sizes = [n_train, n_val] + ([n_test] if n_test else [])
        splits = torch.utils.data.random_split(full, sizes, generator=g)
        train_idx = splits[0].indices
        val_idx = splits[1].indices
        test_idx = splits[2].indices if n_test else []
        # 训练子集用增强变换，验证/测试用确定性变换
        val_ds = datasets.ImageFolder(str(img_root), transform=val_tf)
        train_loader = DataLoader(Subset(datasets.ImageFolder(str(img_root), transform=train_tf), train_idx),
                                  batch_size=batch_size, shuffle=True, num_workers=0,
                                  pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0,
                                pin_memory=(device.type == "cuda"))
        test_loader = DataLoader(Subset(val_ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=0,
                                 pin_memory=(device.type == "cuda")) if test_idx else None
        log(f"划分: 训练 {n_train} / 验证 {n_val}"
            + (f" / 测试 {n_test}" if n_test else "（无测试集）")
            + f"  (val_split={val_split}, test_split={test_split})")
        emit({"type": "split", "n_train": n_train, "n_val": n_val, "n_test": n_test,
              "seed": seed, "val_split": val_split, "test_split": test_split})

        net = build_model(model_key, num_classes, pretrained).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        log(f"模型 {model_key} 参数量: {n_params/1e6:.2f}M")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc, best_epoch = 0.0, -1
        epoch_logs: list[dict] = []
        for epoch in range(1, epochs + 1):
            net.train()
            tr_loss = tr_correct = tr_n = 0
            for bi, (xb, yb) in enumerate(train_loader, 1):
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = net(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                bs = yb.size(0)
                tr_loss += loss.item() * bs
                tr_correct += (out.argmax(1) == yb).sum().item()
                tr_n += bs
                if bi % 20 == 0 or bi == len(train_loader):
                    log(f"epoch {epoch}/{epochs} batch {bi}/{len(train_loader)} loss={loss.item():.4f}")
            train_loss = tr_loss / max(tr_n, 1)
            train_acc = tr_correct / max(tr_n, 1)

            net.eval()
            val_loss, val_acc = evaluate(net, val_loader, criterion, device)[:2]
            scheduler.step()
            rec = {
                "type": "epoch", "epoch": epoch, "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5), "train_acc": round(train_acc, 5), "val_acc": round(val_acc, 5),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
            }
            emit(rec)
            epoch_logs.append(rec)
            marker = ""
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                torch.save({"model": model_key, "classes": classes, "image_size": image_size,
                            "state_dict": net.state_dict(), "val_acc": val_acc, "epoch": epoch},
                           safe(run_dir, "best.pt"))
                marker = " (保存 best.pt)"
            log(f"epoch {epoch}/{epochs} 完成: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

        log("训练结束，生成评估图表 ...")
        plot_training_curves(epoch_logs, safe(run_dir, "curves.png"))

        # 回读 best.pt（按验证集挑出的权重），在从未参与训练的测试集上做唯一一次终评
        if test_loader is not None:
            best = torch.load(safe(run_dir, "best.pt"), map_location=device, weights_only=False)
            net.load_state_dict(best["state_dict"])
            log(f"已回读 best.pt（epoch {best.get('epoch')}，验证集 acc={best.get('val_acc', 0):.4f}），"
                f"在测试集上做最终评估 ...")
            te_loss, te_acc, te_preds, te_labels = evaluate(net, test_loader, criterion, device)
            eval_source = "test"
            eval_n = n_test
            eval_acc, eval_loss = te_acc, te_loss
            emit({"type": "test", "test_accuracy": round(te_acc, 5), "test_loss": round(te_loss, 5),
                  "n_test": n_test, "best_epoch": best_epoch})
            log(f"测试集结果: acc={te_acc:.4f} loss={te_loss:.4f}（{n_test} 张）")
        else:
            va_loss_f, va_acc_f, te_preds, te_labels = evaluate(net, val_loader, criterion, device)
            eval_source = "val"
            eval_n = n_val
            eval_acc, eval_loss = va_acc_f, va_loss_f
            log("未设置测试集，最终指标来自验证集（论文中不应称为测试集结果）")

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p_, l_ in zip(te_preds, te_labels):
            cm[l_, p_] += 1
        per_class_acc = {classes[i]: round(float(cm[i, i] / max(cm[i].sum(), 1)), 4) for i in range(num_classes)}
        cm_labels = [f"{classes[i]}({int(cm[i].sum())})" for i in range(num_classes)]
        plot_confusion_matrix(cm, cm_labels, safe(run_dir, "confusion_matrix.png"))

        metrics = {
            "best_val_accuracy": round(best_acc, 5),
            f"{eval_source}_accuracy": round(eval_acc, 5),
            f"final_{eval_source}_loss": round(eval_loss, 5),
        }
        if n_test:
            metrics["test_accuracy"] = round(eval_acc, 5)
            primary = {"name": "test_accuracy", "value": round(eval_acc, 5)}
        else:
            primary = {"name": "val_accuracy", "value": round(best_acc, 5)}

        summary = {
            "task": "image_classification", "model": model_key,
            "model_label": {"cnn": "CNN(3层卷积)", "resnet18": "ResNet18"}.get(model_key, model_key),
            "dataset_name": cfg.get("dataset_name"), "params": params,
            "classes": classes, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "val_split": val_split, "test_split": test_split if n_test else 0.0,
            "epochs": epoch_logs, "best_epoch": best_epoch,
            "split_scheme": "train/val/test" if n_test else "train/val",
            "eval_source": eval_source, "eval_n": eval_n,
            "metrics": metrics,
            "primary_metric": primary,
            "per_class_accuracy": per_class_acc,
            "device": str(device),
            "artifacts": ["curves.png", "confusion_matrix.png", "class_balance.png"],
            "train_time_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(safe(run_dir, "summary.json"), summary)
        log(f"全部完成：验证集最优 acc={best_acc:.4f} (epoch {best_epoch})，"
            f"{eval_source} acc={eval_acc:.4f}，用时 {summary['train_time_sec']}s")
        write_json(safe(run_dir, "status.json"), {"state": "done", "error": None, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 0
    except Exception:
        err = traceback.format_exc()
        log("训练失败:\n" + err)
        (run_dir / "status.json").write_text(
            json.dumps({"state": "failed", "error": err[-1500:], "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
