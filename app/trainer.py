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
import shutil
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
    "test_size": 0.2,
    "val_split": 0.2,
    "seed": 42,
    "device": "auto",
    "image_size": 64,
    "max_seq_len": 128,
    "vocab_size": 5000,
    "lookback": 12,
    "horizon": 1,
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


def _demo_value(v):
    """把 numpy/pandas 标量转成可安全写 JSON 的 Python 标量。"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_demo_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _demo_value(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _classify_failure(err_text: str) -> dict:
    """把训练异常粗略分级，给新手一句可操作的解释而不是只甩 traceback。"""
    low = (err_text or "").lower()
    if "out of memory" in low or "cuda oom" in low or "cuda out of memory" in low:
        return {
            "kind": "oom",
            "hint": "显存/内存不足。建议减小 batch_size、image_size 或 hidden_sizes，降低网络层数，"
                    "或改用 CPU 训练；同时关闭其他占用显存的程序后再试。",
        }
    if "no space left" in low or "errno 28" in low or ("disk" in low and "space" in low):
        return {
            "kind": "disk",
            "hint": "磁盘空间不足。请清理实验输出目录所在磁盘，删除不再需要的实验或临时文件后重试。",
        }
    return {
        "kind": "error",
        "hint": "训练异常，具体原因见下方错误信息与运行日志（data/logs/launch.log）。",
    }


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
    out["test_size"] = _num(out["test_size"], 0.2, 0.0, 0.5)
    out["val_split"] = _num(out["val_split"], 0.2, 0.0, 0.5)
    out["image_size"] = _int(out["image_size"], 64, 16, 224)
    out["max_seq_len"] = _int(out["max_seq_len"], 128, 4, 2048)
    out["vocab_size"] = _int(out["vocab_size"], 5000, 8, 200000)
    out["lookback"] = _int(out["lookback"], 12, 2, 512)
    out["horizon"] = _int(out["horizon"], 1, 1, 128)
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

    from app.prep import build_tabular_transformer, to_dense

    num_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    prep_cfg = config.get("prep") or {}
    pre = build_tabular_transformer(prep_cfg, num_cols, cat_cols)
    pre.fit(X_tr)

    Xn_tr = to_dense(pre.transform(X_tr))
    Xn_va = to_dense(pre.transform(X_va))
    Xn_te = to_dense(pre.transform(X_te))

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
    demo_idx = int(te_idx[0]) if len(te_idx) else int(va_idx[0])
    return {
        "train": loaders[0], "val": loaders[1], "test": loaders[2],
        "num_features": int(Xn_tr.shape[1]),
        "preprocessor": pre,
        "num_classes": len(cls2idx) if cls2idx else 1,
        "classes": sorted(cls2idx) if cls2idx else None,
        "class_counts": {str(k): int(v) for k, v in (class_counts or {}).items()},
        "n": len(df),
        "demo": {
            "kind": "tabular",
            "columns": [str(c) for c in X_raw.columns],
            "row": {str(k): _demo_value(v) for k, v in X_raw.iloc[demo_idx].items()},
            "true_label": _demo_value(y_raw.iloc[demo_idx]),
        },
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
    demo_idx = int(te_idx[0]) if len(te_idx) else int(va_idx[0])
    return {
        "train": loaders[0], "val": loaders[1], "test": loaders[2],
        "num_tokens": len(vocab), "max_seq_len": seq_len,
        "vocab": vocab,
        "num_classes": len(cls2idx),
        "classes": sorted(cls2idx),
        "class_counts": {str(k): int(v) for k, v in y_raw.value_counts().items()},
        "n": len(df),
        "demo": {
            "kind": "text",
            "text": _demo_value(texts.iloc[demo_idx]),
            "true_label": _demo_value(y_raw.iloc[demo_idx]),
        },
    }


def _load_image(config: dict, st: dict, device, emit):
    torch = _torch()
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    ds_dir = Path(config["dataset_dir"]).resolve()
    img_root = _safe(ds_dir, "images")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    size = st["image_size"]
    augment = str((config.get("prep") or {}).get("augment") or "flip_rotate")
    aug_ops = []
    if augment in ("flip_rotate", "all"):
        aug_ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    elif augment == "crop":
        aug_ops.append(transforms.RandomResizedCrop(size, scale=(0.8, 1.0)))
    elif augment == "color_jitter":
        aug_ops.append(transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25))
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        *aug_ops,
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
    demo_idx = int(test_idx[0]) if test_idx else int(val_idx[0])
    return {
        "train": train_loader, "val": val_loader, "test": test_loader,
        "classes": classes, "class_counts": counts, "num_classes": len(classes),
        "image_size": size,
        "n": n_total,
        "demo": {
            "kind": "image",
            "path": str(full.samples[demo_idx][0]),
        "class": classes[int(full.samples[demo_idx][1])],
        },
    }


def _load_time_series(config: dict, st: dict, device, emit):
    """时间序列数据：时间顺序划分 + 滑窗，避免随机划分造成未来信息泄漏。"""
    path = Path(config["dataset_dir"]) / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    frame = pd.read_csv(path)
    target = config.get("target") or (frame.columns or [None])[-1]
    if target not in frame.columns:
        raise ValueError(f"目标列 {target} 不在数据集中")
    y_values = pd.to_numeric(frame[target], errors="coerce")
    if y_values.isna().any():
        raise ValueError("时间序列目标列必须全部为数值")
    features = frame.select_dtypes(include=[np.number]).copy()
    if target not in features.columns:
        features[target] = y_values.to_numpy(dtype=np.float32)
    features = features.loc[:, features.columns.drop_duplicates()]
    values = features.to_numpy(dtype=np.float32)
    lookback = int(st["lookback"])
    horizon = int(st["horizon"])
    if len(values) < lookback + horizon + 2:
        raise ValueError("时间序列样本不足，无法完成回看窗口、预测步数和训练/验证/测试划分")

    test_n = max(1, int(round(len(values) * float(st["test_size"]))))
    val_n = max(1, int(round(len(values) * float(st["val_split"]))))
    train_end = len(values) - test_n - val_n
    if train_end <= 0:
        raise ValueError("测试/验证比例过大，时间序列训练段为空")
    raw_train, raw_val, raw_test = (
        values[:train_end], values[train_end:train_end + val_n], values[-test_n:]
    )
    mean = raw_train.mean(axis=0)
    std = raw_train.std(axis=0)
    std[std < 1e-8] = 1.0
    train = ((raw_train - mean) / std).astype(np.float32)
    val = ((raw_val - mean) / std).astype(np.float32)
    test = ((raw_test - mean) / std).astype(np.float32)

    torch = _torch()

    def make_windows(arr: np.ndarray):
        xs, ys = [], []
        for i in range(len(arr) - lookback - horizon + 1):
            xs.append(arr[i:i + lookback])
            ys.append(arr[i + lookback:i + lookback + horizon, -1])
        return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))

    splits = [make_windows(arr) for arr in (train, val, test)]
    loaders = [
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=int(st["batch_size"]), shuffle=False,
        ) for x, y in splits
    ]
    emit({"type": "data", "n_raw": int(len(values)), "lookback": lookback, "horizon": horizon,
          "n_train": int(splits[0][0].shape[0]), "n_val": int(splits[1][0].shape[0]),
          "n_test": int(splits[2][0].shape[0]), "target": target,
          "features": list(features.columns)})
    _log(f"时间序列划分: raw={len(values)}, train={splits[0][0].shape[0]}, "
         f"val={splits[1][0].shape[0]}, test={splits[2][0].shape[0]}; "
         f"lookback={lookback}, horizon={horizon}")
    return {
        "train": loaders[0], "val": loaders[1], "test": loaders[2],
        "feature_names": list(features.columns), "target_name": target,
        "num_features": int(values.shape[1]), "lookback": lookback, "horizon": horizon,
        "scale_mean": mean.tolist(), "scale_std": std.tolist(),
        "split_scheme": "chronological train/val/test",
    }


def _load_data(config: dict, st: dict, device, emit) -> dict:
    task = config["task"]
    if task in ("tabular_classification", "tabular_regression"):
        return _load_tabular(config, st, device, emit)
    if task == "text_classification":
        return _load_text(config, st, device, emit)
    if task == "image_classification":
        return _load_image(config, st, device, emit)
    if task == "time_series_forecasting":
        return _load_time_series(config, st, device, emit)
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


def _plot_time_series(y_true, y_pred, target_name: str, mean: float, std: float, path) -> None:
    """把标准化后的测试预测还原到原始量纲并绘制真实/预测对比曲线。"""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float).reshape(-1) * std + mean
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1) * std + mean
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(y_true, label="真实值", color="#2c7be5", linewidth=2)
    ax.plot(y_pred, label="预测值", color="#ff7f50", linewidth=2, linestyle="--")
    ax.set_xlabel("时间步")
    ax.set_ylabel(str(target_name))
    ax.set_title("时间序列真实值 vs 预测值")
    ax.legend()
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
        "num_classes": data.get("num_classes"),
        "vocab": data.get("vocab"),
        "preprocessor": data.get("preprocessor"),
        "num_features": data.get("num_features"),
        "num_tokens": data.get("num_tokens"),
        "max_seq_len": data.get("max_seq_len"),
        "image_size": image_size,
        "device": str(device),
    }
    torch.save(checkpoint, str(path))


def _save_demo(run_dir: Path, data: dict, net, device, cfg: dict) -> dict | None:
    """用最佳模型在测试集（无测试集时用验证集）首个样本上生成答辩演示。"""
    demo = data.get("demo")
    if not demo:
        return None
    torch = _torch()
    loader = data["test"] if data.get("test") is not None and len(data["test"].dataset) > 0 else data.get("val")
    if loader is None or len(loader.dataset) == 0:
        return None
    x, _ = next(iter(loader))
    x = x[:1].to(device, non_blocking=True)
    net.eval()
    with torch.no_grad():
        logits = net(x)
    task = cfg.get("task")
    classification = task in ("tabular_classification", "text_classification", "image_classification")
    out = {
        "task": task,
        "model": cfg.get("model"),
        "model_label": cfg.get("model_label"),
        "input_kind": demo.get("kind"),
        "input_columns": demo.get("columns"),
        "input": demo.get("row") if demo.get("kind") == "tabular"
                 else demo.get("text") if demo.get("kind") == "text" else demo.get("path"),
        "image_class": demo.get("class"),
        "true_label": demo.get("true_label"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if classification:
        probs = torch.softmax(logits[0].cpu(), dim=0)
        classes = data.get("classes") or []
        pred_idx = int(probs.argmax(dim=0))
        out["predicted_label"] = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        out["probabilities"] = [
            {"label": classes[i] if i < len(classes) else str(i), "prob": round(float(p), 4)}
            for i, p in enumerate(probs)
        ]
    else:
        out["predicted_value"] = round(float(logits.reshape(-1)[0].cpu()), 4)
    if demo.get("kind") == "image" and demo.get("path"):
        src = Path(str(demo["path"]))
        if src.exists():
            try:
                shutil.copy2(src, _safe(run_dir, "demo_sample.png"))
            except Exception as e:
                _log(f"演示图片复制跳过: {e}")
    _write_json(_safe(run_dir, "demo.json"), out)
    _log("已生成答辩演示：demo.json"
         + (" + demo_sample.png" if (run_dir / "demo_sample.png").exists() else ""))
    return out


def _generate_explanations(run_dir: Path, data: dict, net, cfg: dict, final_loader,
                           classification: bool, artifacts: list[str]) -> None:
    """按任务生成解释图；解释失败只记录日志，不改变训练结果。"""
    torch = _torch()
    task = cfg.get("task")
    model_key = cfg.get("model")
    original_device = next(net.parameters()).device
    try:
        if task == "image_classification":
            xb = next(iter(final_loader))[0].to(next(net.parameters()).device)
            from app.explain import plot_grad_cam_examples

            plot_grad_cam_examples(
                net, xb, _safe(run_dir, "grad_cam.png"),
                classes=data.get("classes"),
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            )
            artifacts.extend(["grad_cam.png"])
            _log("已生成 Grad-CAM 解释图: grad_cam.png")

        elif task == "text_classification":
            xb = next(iter(final_loader))[0].to(next(net.parameters()).device)
            from app.explain import plot_token_attribution, token_attribution

            inv_vocab = {int(i): token for token, i in (data.get("vocab") or {}).items()}
            tokens = [inv_vocab.get(int(v), "<pad>" if int(v) == 0 else f"#{int(v)}")
                      for v in xb[0].tolist()]
            attributions = token_attribution(net, xb, tokens=tokens)
            _write_json(_safe(run_dir, "token_attribution.json"),
                        {"task": task, "model": model_key, "attributions": attributions})
            artifacts.append("token_attribution.json")
            plot = plot_token_attribution(attributions, _safe(run_dir, "token_attribution.png"))
            if plot:
                artifacts.append("token_attribution.png")
            _log("已生成文本 token 归因解释")

        elif task in ("tabular_classification", "tabular_regression") and model_key == "mlp":
            net = net.to(torch.device("cpu"))
            from app.explain import permutation_importance

            feature_names = None
            pre = data.get("preprocessor")
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                feature_names = [str(v) for v in pre.get_feature_names_out()]
            importance = permutation_importance(
                net, final_loader, _safe(run_dir, "permutation_importance.png"),
                classification=classification, feature_names=feature_names,
                seed=int(cfg.get("params", {}).get("seed", 42)),
            )
            if importance:
                _write_json(_safe(run_dir, "permutation_importance.json"),
                            {"task": task, "model": model_key, "importance": importance})
                artifacts.extend(["permutation_importance.png", "permutation_importance.json"])
                _log("已生成表格置换重要性解释")
            else:
                _log("置换重要性无有效信号，解释图跳过")
    except Exception as e:
        _log(f"模型解释生成跳过: {e}")
    finally:
        net.to(original_device)


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
    classification = task in ("tabular_classification", "text_classification", "image_classification")
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

    _save_demo(run_dir, data, net, device, cfg)

    _log("训练结束，生成评估图表 ...")
    from app.plots import plot_class_balance, plot_confusion_matrix, plot_roc, plot_training_curves

    plot_training_curves(epoch_logs, _safe(run_dir, "curves.png"))
    artifacts = ["curves.png"]
    if (run_dir / "demo_sample.png").exists():
        artifacts.append("demo_sample.png")

    final_loader = data["test"] if data["test"] is not None and len(data["test"].dataset) > 0 else val_load
    eval_source = "test" if final_loader is data["test"] else "val"
    ev = _evaluate(net, final_loader, criterion, device, classification)
    n_eval = len(final_loader.dataset)

    _generate_explanations(run_dir, data, net, cfg, final_loader, classification, artifacts)

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
            "split_scheme": data.get("split_scheme") or ("train/val/test" if data["test"] is not None and len(data["test"].dataset) else "train/val"),
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
        if task == "time_series_forecasting":
            _plot_time_series(
                yt, y_pred_full, data.get("target_name", "value"),
                float(data.get("scale_mean", [0.0])[-1]),
                float(data.get("scale_std", [1.0])[-1]),
                _safe(run_dir, "forecast.png"),
            )
            artifacts.append("forecast.png")
        summary = {
            "task": task, "model": model_key, "model_label": cfg.get("model_label", model_key),
            "dataset_name": cfg.get("dataset_name"), "params": params, "engine": "torch",
            "n_train": len(data["train"].dataset), "n_val": len(data["val"].dataset),
            "n_test": len(data["test"].dataset) if data["test"] is not None else 0,
            "split_scheme": data.get("split_scheme") or ("train/val/test" if data["test"] is not None and len(data["test"].dataset) else "train/val"),
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

    try:
        from app.export_predict import write_predict_script

        write_predict_script(run_dir, "torch", cfg)
        if "predict.py" not in artifacts:
            artifacts.append("predict.py")
        summary["artifacts"] = artifacts
        _log("已生成推理脚本: predict.py")
    except Exception as e:
        _log(f"推理脚本生成跳过: {e}")
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
        failure = _classify_failure(err)
        _safe(run_dir, "status.json").write_text(
            json.dumps(_json_safe({"state": "failed", "error": err[-1500:],
                                   "failure_kind": failure["kind"],
                                   "failure_hint": failure["hint"],
                                   "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}),
                       ensure_ascii=False), encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    sys.exit(train_from_run_dir(Path(__file__).resolve().parent.parent / "run"))
