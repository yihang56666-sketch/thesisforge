"""通用 PyTorch 训练引擎：表格 MLP、图像 CNN/ResNet18、文本 LSTM/GRU/TextCNN/Transformer。

`train_from_run_dir()` 供训练子进程入口调用，返回退出码并保证异常时写 failed；
`fit(config, run_dir)` 供单元测试直接调用并在失败时上抛。
产物与旧训练脚本一致：metrics.jsonl / summary.json / best.pt / curves.png /
confusion_matrix.png / roc.png / class_balance.png / pred_vs_true.png / status.json。
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STRATEGY: dict = {
    "optimizer": "adam",
    "lr": 0.001,
    "momentum": 0.9,
    "batch_size": 32,
    "epochs": 15,
    "scheduler": "cosine",
    "step_size": 10,
    "gamma": 0.5,
    "weight_decay": 0.0,
    "early_stop_patience": 0,
    "grad_clip": 0.0,
    "seed": 42,
    "device": "auto",
    "image_size": 64,
    "max_seq_len": 128,
    "vocab_size": 5000,
}

_OPTIMIZER_KEYS = {"sgd", "sgd_momentum", "adam", "adamw", "rmsprop"}
_SCHEDULER_KEYS = {"cosine", "step", "plateau", "none"}


def _torch():
    import torch

    return torch


def _safe(base: Path, name: str) -> Path:
    base = base.resolve()
    target = base / name
    if ".." in target.parts or not target.resolve().is_relative_to(base):
        raise ValueError("非法路径")
    return target.resolve()


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(obj):
    """把非有限浮点（NaN/Inf）转成 null，保证产物始终是合法 JSON。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    return obj


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _num(value, default: float, lo=None, hi=None) -> float:
    try:
        v = float(value)
        if not np.isfinite(v):
            v = default
    except (TypeError, ValueError, OverflowError):
        v = default
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return v


def _int(value, default: int, lo=None, hi=None) -> int:
    try:
        v = int(float(value))
    except (TypeError, ValueError, OverflowError):
        v = default
    if lo is not None:
        v = max(int(lo), v)
    if hi is not None:
        v = min(int(hi), v)
    return v


def _bool(value, default: bool = False) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def strategy(config: dict) -> dict:
    """把 config 顶层键与 config.params 合并成训练策略，缺失键用默认值。"""
    out = dict(DEFAULT_STRATEGY)
    merged = dict(config)
    merged.update(config.get("params") or {})
    for key in out:
        if key in merged:
            out[key] = merged[key]
    out["optimizer"] = str(out["optimizer"] or "adam").lower()
    out["scheduler"] = str(out["scheduler"] or "cosine").lower()
    if out["optimizer"] not in _OPTIMIZER_KEYS:
        raise ValueError(f"未知优化器: {out['optimizer']}（可选 {sorted(_OPTIMIZER_KEYS)}）")
    if out["scheduler"] not in _SCHEDULER_KEYS:
        raise ValueError(f"未知学习率调度器: {out['scheduler']}（可选 {sorted(_SCHEDULER_KEYS)}）")
    out["lr"] = _num(out["lr"], 0.001, 1e-8, 10.0)
    out["momentum"] = _num(out["momentum"], 0.9, 0.0, 0.999)
    out["batch_size"] = _int(out["batch_size"], 32, 1, 4096)
    out["epochs"] = _int(out["epochs"], 15, 1, 3000)
    out["step_size"] = _int(out["step_size"], 10, 1, 10000)
    out["gamma"] = _num(out["gamma"], 0.5, 0.01, 0.999)
    out["weight_decay"] = _num(out["weight_decay"], 0.0, 0.0, 1.0)
    out["early_stop_patience"] = _int(out["early_stop_patience"], 0, 0, 10000)
    out["grad_clip"] = _num(out["grad_clip"], 0.0, 0.0, 1000.0)
    out["seed"] = _int(out["seed"], 42, 0, 2**31 - 1)
    out["image_size"] = _int(out["image_size"], 64, 16, 224)
    out["max_seq_len"] = _int(out["max_seq_len"], 128, 4, 2048)
    out["vocab_size"] = _int(out["vocab_size"], 5000, 8, 200000)
    return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(device: str):
    torch = _torch()
    key = str(device or "auto").lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 GPU，但当前环境没有可用的 CUDA 设备")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_indices(n: int, test_size: float, val_split: float, seed: int, stratify=None):
    """训练/验证/测试三方索引划分；分类任务优先分层，样本过少时自动退化为随机划分。"""
    if n < 3:
        raise ValueError("数据量过小（少于 3 条），无法划分训练/验证/测试集")
    n_test = max(1, int(round(n * max(test_size, 0.0)))) if test_size > 0 else 0
    n_val = max(1, int(round(n * max(val_split, 0.0))))
    if n_test == 0 and n_val >= n:
        n_val = max(1, n - 1)
    n_train = n - n_test - n_val
    if n_train < 1:
        n_test = 1 if test_size > 0 else 0
        n_val = 1
        n_train = n - n_test - n_val
    if n_train < 1:
        raise ValueError("数据量过小，无法同时划分训练/验证/测试集")

    idx = np.arange(n)
    strat_all = None
    if stratify is not None:
        s = np.asarray(stratify)
        counts = pd.Series(s).value_counts()
        if counts.min() >= 2 and len(counts) >= 2:
            strat_all = s
    from sklearn.model_selection import train_test_split

    def _pair(arr, strata, te_n, rseed):
        try:
            a, b = train_test_split(arr, test_size=te_n, random_state=rseed,
                                    stratify=strata if strata is not None else None)
        except ValueError:
            a, b = train_test_split(arr, test_size=te_n, random_state=rseed)
        return a, b

    tr, te = _pair(idx, strat_all, n_test, seed) if n_test else (idx, idx[:0])
    tr_strat = np.asarray(stratify)[tr] if stratify is not None else None
    if tr_strat is not None:
        c = pd.Series(tr_strat).value_counts()
        if c.min() < 2 or len(c) < 2:
            tr_strat = None
    tr2, va = _pair(tr, tr_strat, n_val, seed + 1)
    return tr2, va, te


def _class_map(y_train) -> dict:
    classes = sorted({str(v) for v in y_train})
    if len(classes) < 2:
        raise ValueError("分类训练集至少需要 2 个类别")
    return {c: i for i, c in enumerate(classes)}


def _encode_labels(values, cls2idx) -> list[int]:
    mapped = [str(v) for v in values]
    unseen = sorted({c for c in mapped if c not in cls2idx})
    if unseen:
        raise ValueError(f"验证/测试集出现了训练集未见过的类别: {unseen}")
    return [cls2idx[c] for c in mapped]


def _load_tabular(config: dict, st: dict, device, emit):
    torch = _torch()
    from torch.utils.data import DataLoader, TensorDataset

    ds_dir = Path(config["dataset_dir"]).resolve()
    csv_path = _safe(ds_dir, "dataset.csv")
    df = pd.read_csv(csv_path)
    target = config.get("target") or df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"标签列 {target!r} 不在数据集中")
    task = config["task"]
    X_raw = df.drop(columns=[target])
    y_raw = df[target]
    is_class = task == "tabular_classification"
    if is_class:
        y_raw = y_raw.astype(str)
    else:
        y_raw = pd.to_numeric(y_raw, errors="raise")

    tr_idx, va_idx, te_idx = _split_indices(
        len(df), float(config.get("test_size", 0.2)), float(config.get("val_split", 0.2)),
        st["seed"], stratify=y_raw if is_class else None,
    )
    X_tr, X_va, X_te = X_raw.iloc[tr_idx], X_raw.iloc[va_idx], X_raw.iloc[te_idx]
    y_tr, y_va, y_te = y_raw.iloc[tr_idx], y_raw.iloc[va_idx], y_raw.iloc[te_idx]

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    pre.fit(X_tr)

    def _to_dense(m):
        return np.asarray(m.toarray() if hasattr(m, "toarray") else m, dtype=np.float32)

    Xn_tr = _to_dense(pre.transform(X_tr))
    Xn_va = _to_dense(pre.transform(X_va))
    Xn_te = _to_dense(pre.transform(X_te))

    cls2idx = None
    if is_class:
        cls2idx = _class_map(y_tr)
        y_arr = [_encode_labels(y_tr, cls2idx), _encode_labels(y_va, cls2idx), _encode_labels(y_te, cls2idx)]
        y_tensors = [torch.tensor(a, dtype=torch.long) for a in y_arr]
    else:
        y_tensors = [torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1))
                     for y in (y_tr, y_va, y_te)]

    xs = [torch.tensor(x) for x in (Xn_tr, Xn_va, Xn_te)]
    datasets = [TensorDataset(x, y) for x, y in zip(xs, y_tensors)]
    loaders = [
        DataLoader(d, batch_size=st["batch_size"], shuffle=(i == 0), num_workers=0,
                   pin_memory=(device.type == "cuda"))
        for i, d in enumerate(datasets)
    ]
    emit({"type": "split", "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
          "n_test": int(len(te_idx)), "seed": st["seed"]})
    _log(f"划分: 训练 {len(tr_idx)} / 验证 {len(va_idx)} / 测试 {len(te_idx)} "
         f"(test_size={config.get('test_size')}, val_split={config.get('val_split')})")
    class_counts = y_raw.value_counts().to_dict() if is_class else None
    return {
        "train": loaders[0], "val": loaders[1], "test": loaders[2],
        "num_features": int(Xn_tr.shape[1]),
        "classes": sorted(cls2idx) if cls2idx else None,
        "class_counts": {str(k): int(v) for k, v in (class_counts or {}).items()},
        "n": len(df),
    }


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(str(text).lower()) if t]


def _load_text(config: dict, st: dict, device, emit):
    torch = _torch()
    from torch.utils.data import DataLoader, TensorDataset

    ds_dir = Path(config["dataset_dir"]).resolve()
    df = pd.read_csv(_safe(ds_dir, "dataset.csv"))
    target = config.get("target") or df.columns[-1]
    text_column = config.get("text_column")
    if not text_column or text_column not in df.columns:
        raise ValueError(f"文本列 {text_column!r} 不在数据集中")
    if target not in df.columns:
        raise ValueError(f"标签列 {target!r} 不在数据集中")
    texts = df[text_column].astype(str).fillna("")
    y_raw = df[target].astype(str)
    tr_idx, va_idx, te_idx = _split_indices(
        len(df), float(config.get("test_size", 0.2)), float(config.get("val_split", 0.2)),
        st["seed"], stratify=y_raw,
    )

    # 只在训练集上建立词表，避免验证/测试信息泄漏
    from collections import Counter

    vocab_size = st["vocab_size"]
    counter = Counter()
    for txt in texts.iloc[tr_idx]:
        counter.update(_tokenize(txt))
    top = [w for w, _ in counter.most_common(max(2, vocab_size - 2))]
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(top)})
    seq_len = st["max_seq_len"]

    def encode(rows) -> torch.Tensor:
        enc = np.zeros((len(rows), seq_len), dtype=np.int64)
        for r, txt in enumerate(rows):
            toks = [vocab.get(t, 1) for t in _tokenize(txt)][:seq_len]
            enc[r, : len(toks)] = toks
        return torch.tensor(enc)

    X_tr, X_va, X_te = (encode(texts.iloc[i]) for i in (tr_idx, va_idx, te_idx))
    cls2idx = _class_map(y_raw.iloc[tr_idx])
    y_tensors = [torch.tensor(_encode_labels(y_raw.iloc[i], cls2idx), dtype=torch.long)
                 for i in (tr_idx, va_idx, te_idx)]
    datasets = [TensorDataset(x, y) for x, y in zip((X_tr, X_va, X_te), y_tensors)]
    loaders = [
        DataLoader(d, batch_size=st["batch_size"], shuffle=(i == 0), num_workers=0,
                   pin_memory=(device.type == "cuda"))
        for i, d in enumerate(datasets)
    ]
    emit({"type": "split", "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
          "n_test": int(len(te_idx)), "vocab_size": len(vocab), "seed": st["seed"]})
    _log(f"文本划分: 训练 {len(tr_idx)} / 验证 {len(va_idx)} / 测试 {len(te_idx)}，"
         f"词表 {len(vocab)}，序列长度 {seq_len}")
    return {
        "train": loaders[0], "val": loaders[1], "test": loaders[2],
        "num_tokens": len(vocab), "max_seq_len": seq_len,
        "classes": sorted(cls2idx),
        "class_counts": {str(k): int(v) for k, v in y_raw.value_counts().items()},
        "n": len(df),
    }


def _load_image(config: dict, st: dict, device, emit):
    torch = _torch()
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    ds_dir = Path(config["dataset_dir"]).resolve()
    img_root = _safe(ds_dir, "images")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    size = st["image_size"]
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    full = datasets.ImageFolder(str(img_root))
    classes = list(full.classes)
    counts = {classes[ci]: 0 for ci in range(len(classes))}
    for _p, ci in full.samples:
        counts[classes[ci]] += 1
    n_total = len(full)
    if n_total < 3:
        raise ValueError(f"图像数据量过小（{n_total} 张），无法划分训练/验证/测试集")
    n_test = max(1, int(round(n_total * float(config.get("test_size", 0.2))))) if float(config.get("test_size", 0.2)) > 0 else 0
    n_val = max(1, int(round(n_total * float(config.get("val_split", 0.2)))))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        n_test = 1 if n_test else 0
        n_val = 1
        n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError("图像数据量过小，无法同时划分训练/验证/测试集")
    gen = torch.Generator().manual_seed(st["seed"])
    splits = torch.utils.data.random_split(full, [n_train, n_val] + ([n_test] if n_test else []), generator=gen)
    train_idx, val_idx = splits[0].indices, splits[1].indices
    test_idx = splits[2].indices if n_test else []
    train_loader = DataLoader(
        Subset(datasets.ImageFolder(str(img_root), transform=train_tf), train_idx),
        batch_size=st["batch_size"], shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    eval_ds = datasets.ImageFolder(str(img_root), transform=eval_tf)
    val_loader = DataLoader(Subset(eval_ds, val_idx), batch_size=st["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = (DataLoader(Subset(eval_ds, test_idx), batch_size=st["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))
                   if test_idx else None)
    emit({"type": "split", "n_train": n_train, "n_val": n_val, "n_test": n_test,
          "n_images": n_total, "classes": classes, "seed": st["seed"]})
    _log(f"图像划分: 训练 {n_train} / 验证 {n_val} / 测试 {n_test}，类别 {len(classes)}")
    return {
        "train": train_loader, "val": val_loader, "test": test_loader,
        "classes": classes, "class_counts": counts, "num_classes": len(classes),
        "n": n_total,
    }


def _load_data(config: dict, st: dict, device, emit) -> dict:
    task = config["task"]
    if task in ("tabular_classification", "tabular_regression"):
        return _load_tabular(config, st, device, emit)
    if task == "text_classification":
        return _load_text(config, st, device, emit)
    if task == "image_classification":
        return _load_image(config, st, device, emit)
    raise ValueError(f"未知任务类型: {task}")


def _build_optimizer(net, st: dict, torch):
    params = [p for p in net.parameters() if p.requires_grad]
    wd = st["weight_decay"]
    key = st["optimizer"]
    if key == "sgd":
        return torch.optim.SGD(params, lr=st["lr"], weight_decay=wd)
    if key == "sgd_momentum":
        return torch.optim.SGD(params, lr=st["lr"], momentum=st["momentum"], weight_decay=wd)
    if key == "adam":
        return torch.optim.Adam(params, lr=st["lr"], weight_decay=wd)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=st["lr"], weight_decay=wd)
    if key == "rmsprop":
        return torch.optim.RMSprop(params, lr=st["lr"], momentum=st["momentum"], weight_decay=wd)
    raise ValueError(f"未知优化器: {key}")


def _build_scheduler(optimizer, st: dict, torch):
    key = st["scheduler"]
    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, st["epochs"]))
    if key == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, st["step_size"]), gamma=st["gamma"])
    if key == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=st["gamma"], patience=max(1, st["early_stop_patience"] or 3),
        )
    return None


def _train_epoch(net, loader, optimizer, criterion, device, classification: bool, grad_clip: float):
    torch = _torch()
    nn = torch.nn
    net.train()
    loss_sum = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = net(xb)
        loss = criterion(out, yb)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        bs = yb.size(0)
        loss_sum += float(loss.item()) * bs
        if classification:
            correct += int((out.argmax(1) == yb).sum().item())
        total += bs
    return loss_sum / max(total, 1), correct / max(total, 1)


@_torch().no_grad if False else (lambda _f: _f)
def _safe_decorator(func):
    return func


def _evaluate(net, loader, criterion, device, classification: bool):
    torch = _torch()
    net.eval()
    loss_sum = correct = total = 0
    preds: list[int] = []
    labels: list[float] = []
    outputs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            out = net(xb)
            loss_sum += float(criterion(out, yb).item()) * yb.size(0)
            if classification:
                pred = out.argmax(1)
                correct += int((pred == yb).sum().item())
                preds.extend(pred.cpu().tolist())
                labels.extend(yb.cpu().tolist())
                outputs.append(out.cpu())
            else:
                labels.extend(yb.cpu().reshape(-1).tolist())
            total += yb.size(0)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "preds": preds, "labels": labels, "outputs": outputs,
    }


def _plot_pred_true(y_true, y_pred, path) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(y_true, y_pred, s=14, alpha=0.6, color="#4f7cff")
    lims = [min(float(np.min(y_true)), float(np.min(y_pred))),
            max(float(np.max(y_true)), float(np.max(y_pred)))]
    ax.plot(lims, lims, "--", color="#e74c3c")
    ax.set_xlabel("真实值"), ax.set_ylabel("预测值"), ax.set_title("预测值 vs 真实值")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_best(net, data: dict, epoch: int, val_metric: float, path: Path, task: str,
               model: str, params: dict, device, image_size=None) -> None:
    torch = _torch()
    checkpoint = {
        "task": task, "model": model, "params": params,
        "state_dict": net.state_dict(), "epoch": int(epoch),
        "val_metric": float(val_metric),
        "classes": data.get("classes"),
        "num_features": data.get("num_features"),
        "num_tokens": data.get("num_tokens"),
        "max_seq_len": data.get("max_seq_len"),
        "image_size": image_size,
        "device": str(device),
    }
    torch.save(checkpoint, str(path))


def fit(config: dict, run_dir) -> dict:
    """按 config 训练并把产物写入 run_dir；失败时上抛异常，由入口统一写 failed。"""
    torch = _torch()
    nn = torch.nn
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(config)
    task = cfg["task"]
    model_key = cfg["model"]
    params = cfg.get("params") or {}
    st = strategy(cfg)
    _set_seed(st["seed"])
    device = _pick_device(st["device"])
    t0 = time.time()

    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)
        _safe(run_dir, "metrics.jsonl").write_text(
            "\n".join(json.dumps(_json_safe(e), ensure_ascii=False) for e in events),
            encoding="utf-8",
        )

    _log(f"任务: {task} | 模型: {model_key} | 策略: optimizer={st['optimizer']}, "
         f"lr={st['lr']}, epochs={st['epochs']}, scheduler={st['scheduler']}, "
         f"weight_decay={st['weight_decay']}, grad_clip={st['grad_clip']}")
    _log(f"设备: {device}" + (f"（GPU {torch.cuda.get_device_name(0)}）" if device.type == "cuda" else ""))

    data = _load_data(cfg, st, device, emit)
    classification = task != "tabular_regression"
    if classification:
        num_classes = len(data["classes"])
    else:
        num_classes = 2  # 回归输出维度为 1，build_model 对回归固定使用 1

    from app.networks import build_model, count_parameters

    net = build_model(
        task, model_key, params,
        num_features=data.get("num_features"),
        num_classes=num_classes,
        num_tokens=data.get("num_tokens"),
        max_seq_len=data.get("max_seq_len"),
    ).to(device)
    n_params = count_parameters(net)
    _log(f"模型参数量: {n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    optimizer = _build_optimizer(net, st, torch)
    scheduler = _build_scheduler(optimizer, st, torch)

    best_metric = -float("inf") if classification else float("inf")
    best_epoch = 0
    no_impr = 0
    epoch_logs: list[dict] = []
    val_load = data["val"]
    if len(val_load.dataset) == 0:
        raise ValueError("验证集为空，无法按验证集选择模型")

    for epoch in range(1, st["epochs"] + 1):
        train_loss, train_acc = _train_epoch(
            net, data["train"], optimizer, criterion, device, classification, st["grad_clip"],
        )
        ev = _evaluate(net, val_load, criterion, device, classification)
        rec = {
            "type": "epoch", "epoch": epoch,
            "train_loss": round(float(train_loss), 5),
            "val_loss": round(float(ev["loss"]), 5),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        if classification:
            rec["train_acc"] = round(float(train_acc), 5)
            rec["val_acc"] = round(float(ev["acc"]), 5)
            is_better = ev["acc"] > best_metric + 1e-9
        else:
            is_better = ev["loss"] < best_metric - 1e-9
        if scheduler is not None:
            if st["scheduler"] == "plateau":
                scheduler.step(ev["loss"])
            else:
                scheduler.step()

        if is_better:
            best_metric = ev["acc"] if classification else ev["loss"]
            best_epoch = epoch
            no_impr = 0
            _save_best(net, data, epoch, best_metric, _safe(run_dir, "best.pt"),
                       task, model_key, params, device, image_size=st.get("image_size"))
        else:
            no_impr += 1
        emit(rec)
        epoch_logs.append(rec)
        mark = "（保存 best.pt）" if is_better else ""
        _log(f"epoch {epoch}/{st['epochs']}: train_loss={train_loss:.4f} "
             f"val_loss={ev['loss']:.4f}"
             + (f" val_acc={ev['acc']:.4f}" if classification else "") + mark)
        if st["early_stop_patience"] > 0 and no_impr >= st["early_stop_patience"]:
            _log(f"早停触发：连续 {no_impr} 轮验证指标未提升")
            break

    if best_epoch < 1:
        best_epoch = len(epoch_logs)
        _save_best(net, data, best_epoch, best_metric if classification else ev["loss"],
                   _safe(run_dir, "best.pt"), task, model_key, params, device,
                   image_size=st.get("image_size"))
    elif (run_dir / "best.pt").exists():
        ckpt = torch.load(_safe(run_dir, "best.pt"), map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        _log(f"已回读 best.pt（epoch {ckpt.get('epoch')}，验证指标 {ckpt.get('val_metric')}）")

    _log("训练结束，生成评估图表 ...")
    from app.plots import plot_class_balance, plot_confusion_matrix, plot_roc, plot_training_curves

    plot_training_curves(epoch_logs, _safe(run_dir, "curves.png"))
    artifacts = ["curves.png"]

    final_loader = data["test"] if data["test"] is not None and len(data["test"].dataset) > 0 else val_load
    eval_source = "test" if final_loader is data["test"] else "val"
    ev = _evaluate(net, final_loader, criterion, device, classification)
    n_eval = len(final_loader.dataset)

    if classification:
        classes = data["classes"]
        labels_all = sorted(set(ev["labels"]) | set(ev["preds"]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for p_, l_ in zip(ev["preds"], ev["labels"]):
            if 0 <= l_ < len(classes) and 0 <= p_ < len(classes):
                cm[l_, p_] += 1
        cm_labels = [f"{classes[i]}({int(cm[i].sum())})" for i in range(len(classes))]
        plot_confusion_matrix(cm, cm_labels, _safe(run_dir, "confusion_matrix.png"))
        artifacts.append("confusion_matrix.png")

        from sklearn.metrics import (
            accuracy_score, confusion_matrix as sk_cm, f1_score,
            precision_score, recall_score, roc_auc_score, roc_curve,
        )

        yt, yp = np.asarray(ev["labels"]), np.asarray(ev["preds"])
        acc = float(accuracy_score(yt, yp))
        metrics = {
            "accuracy": acc,
            "best_val_accuracy": round(float(best_metric), 5),
            "f1_macro": float(f1_score(yt, yp, average="macro", labels=labels_all, zero_division=0)),
            "f1_weighted": float(f1_score(yt, yp, average="weighted", labels=labels_all, zero_division=0)),
            "precision_macro": float(precision_score(yt, yp, average="macro", labels=labels_all, zero_division=0)),
            "recall_macro": float(recall_score(yt, yp, average="macro", labels=labels_all, zero_division=0)),
        }
        if eval_source == "test":
            metrics["test_accuracy"] = acc
        metrics[f"{eval_source}_loss"] = round(float(ev["loss"]), 5)
        primary = {"name": "test_accuracy" if eval_source == "test" else "val_accuracy", "value": acc}
        emit({"type": "test", "test_accuracy": round(acc, 5), "test_loss": round(ev["loss"], 5),
              "n_test": n_eval, "best_epoch": best_epoch, "eval_source": eval_source})
        per_class_acc = {classes[i]: round(float(cm[i, i] / max(int(cm[i].sum()), 1)), 4)
                         for i in range(len(classes))}

        if len(classes) == 2 and ev["outputs"]:
            try:
                outs = torch.cat(ev["outputs"])
                probs = torch.softmax(outs, dim=1)[:, 1].numpy()
                yb = np.asarray(ev["labels"])
                fpr, tpr, _ = roc_curve((yb == 1).astype(int), probs)
                auc = float(roc_auc_score((yb == 1).astype(int), probs))
                metrics["roc_auc"] = auc
                plot_roc(fpr, tpr, auc, _safe(run_dir, "roc.png"))
                artifacts.append("roc.png")
            except Exception as e:
                _log(f"ROC 计算跳过: {e}")

        if data.get("class_counts"):
            plot_class_balance({str(k): int(v) for k, v in data["class_counts"].items()},
                               _safe(run_dir, "class_balance.png"))
            artifacts.append("class_balance.png")
        summary = {
            "task": task, "model": model_key, "model_label": cfg.get("model_label", model_key),
            "dataset_name": cfg.get("dataset_name"), "params": params, "engine": "torch",
            "classes": classes, "num_classes": len(classes),
            "n_train": len(data["train"].dataset), "n_val": len(data["val"].dataset),
            "n_test": len(data["test"].dataset) if data["test"] is not None else 0,
            "split_scheme": "train/val/test" if data["test"] is not None and len(data["test"].dataset) else "train/val",
            "eval_source": eval_source, "eval_n": n_eval,
            "epochs": epoch_logs, "best_epoch": best_epoch,
            "metrics": metrics, "primary_metric": primary,
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm.tolist(),
            "optimizer": st["optimizer"], "scheduler": st["scheduler"],
            "early_stop_patience": st["early_stop_patience"],
            "weight_decay": st["weight_decay"], "grad_clip": st["grad_clip"],
            "params_count": n_params, "device": str(device),
            "artifacts": artifacts,
            "train_time_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        yt = np.asarray(ev["labels"], dtype=float)
        preds_out: list[float] = []
        with torch.no_grad():
            for xb, _ in final_loader:
                xb = xb.to(device, non_blocking=True)
                preds_out.extend(net(xb).reshape(-1).cpu().tolist())
        y_pred_full = np.asarray(preds_out[: len(yt)], dtype=float)
        if len(y_pred_full) < len(yt):
            raise RuntimeError("回归预测数量与样本数量不一致")
        mse = float(mean_squared_error(yt, y_pred_full))
        metrics = {
            "r2": float(r2_score(yt, y_pred_full)),
            "mae": float(mean_absolute_error(yt, y_pred_full)),
            "rmse": float(np.sqrt(mse)),
            "mse": mse,
        }
        primary = {"name": "r2", "value": metrics["r2"]}
        emit({"type": "test", "test_r2": round(metrics["r2"], 5), "test_loss": round(ev["loss"], 5),
              "n_test": n_eval, "best_epoch": best_epoch, "eval_source": eval_source})
        _plot_pred_true(yt, y_pred_full, _safe(run_dir, "pred_vs_true.png"))
        artifacts.append("pred_vs_true.png")
        summary = {
            "task": task, "model": model_key, "model_label": cfg.get("model_label", model_key),
            "dataset_name": cfg.get("dataset_name"), "params": params, "engine": "torch",
            "n_train": len(data["train"].dataset), "n_val": len(data["val"].dataset),
            "n_test": len(data["test"].dataset) if data["test"] is not None else 0,
            "split_scheme": "train/val/test" if data["test"] is not None and len(data["test"].dataset) else "train/val",
            "eval_source": eval_source, "eval_n": n_eval,
            "epochs": epoch_logs, "best_epoch": best_epoch,
            "metrics": metrics, "primary_metric": primary,
            "optimizer": st["optimizer"], "scheduler": st["scheduler"],
            "early_stop_patience": st["early_stop_patience"],
            "weight_decay": st["weight_decay"], "grad_clip": st["grad_clip"],
            "params_count": n_params, "device": str(device),
            "artifacts": artifacts,
            "train_time_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    _write_json(_safe(run_dir, "summary.json"), summary)
    emit({"type": "summary", "primary_metric": primary, "metrics": metrics})
    _write_json(_safe(run_dir, "status.json"),
                {"state": "done", "error": None, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _log(f"全部完成：主指标 {primary['name']}={primary['value']:.4f}，用时 {summary['train_time_sec']}s")
    return summary


def train_from_run_dir(run_dir) -> int:
    """训练子进程入口：读到配置后调用 fit，任何异常都落 failed 状态并返回非 0。"""
    run_dir = Path(run_dir).resolve()
    try:
        cfg = json.loads(_safe(run_dir, "config.json").read_text(encoding="utf-8"))
        fit(cfg, run_dir)
        return 0
    except Exception:
        err = traceback.format_exc()
        _log("训练失败:\n" + err)
        _safe(run_dir, "status.json").write_text(
            json.dumps(_json_safe({"state": "failed", "error": err[-1500:],
                                   "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}),
                       ensure_ascii=False), encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    sys.exit(train_from_run_dir(Path(__file__).resolve().parent.parent / "run"))
