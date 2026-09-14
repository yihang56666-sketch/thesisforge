"""深度模型解释工具：图像 Grad-CAM、文本 token 归因、表格 MLP 置换重要性。

所有工具只在 PyTorch 训练完成后调用，任何失败都不会中断训练。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_cam(cam) -> np.ndarray:
    """把热力图归一化到 0-1；输入为空或常值时返回 0 图。"""
    arr = np.asarray(cam, dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _last_conv_layer(net):
    import torch

    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        raise RuntimeError("模型中没有可用的卷积层，无法生成 Grad-CAM")
    return convs[-1]


def grad_cam_image(net, x, target_layer=None):
    """取第一个样本，返回（归一化 2D 热力图, 预测类别）。"""
    import torch

    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x[0:1].detach()
    if target_layer is None:
        target_layer = _last_conv_layer(net)
    activation = []
    def _hook(_module, _inputs, out):
        activation.append(out)
        out.retain_grad()

    handle = target_layer.register_forward_hook(_hook)
    net.eval()
    try:
        logits = net(x)
        pred_idx = int(logits[0].argmax())
        net.zero_grad(set_to_none=True)
        score = logits[0, pred_idx]
        score.backward()
        grads = activation[0].grad
        weights = grads.mean(dim=(2, 3))[0].detach()
        cam = torch.relu((activation[0][0] * weights[:, None, None]).sum(0))
        return normalize_cam(cam.detach().cpu().numpy()), pred_idx
    finally:
        handle.remove()
        net.zero_grad(set_to_none=True)


def plot_grad_cam_examples(net, x, path: Path, classes: list | None = None,
                           mean: list | None = None, std: list | None = None,
                           max_n: int = 4):
    """把 Grad-CAM 叠加到未归一化图像上，训练完成后自动落盘。"""
    import matplotlib.pyplot as plt
    import torch

    from app.plots import save_fig

    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.detach().cpu()
    n = min(max_n, x.shape[0])
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow), squeeze=False)
    for i in range(n):
        ax = axes[i // ncol][i % ncol]
        cam, pred_idx = grad_cam_image(net, x[i:i + 1])
        img = x[i].clone()
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            img = img * std_t + mean_t
        img = (img - img.min()) / max(float(img.max() - img.min()), 1e-12)
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.imshow(cam, cmap="jet", alpha=0.35, extent=(0, img.shape[2], img.shape[1], 0))
        label = str(classes[pred_idx]) if classes and pred_idx < len(classes) else str(pred_idx)
        ax.set_title(f"预测: {label}", fontsize=9)
        ax.axis("off")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Grad-CAM 模型关注区域", y=1.02)
    return save_fig(fig, Path(path))


def token_attribution(net, x, target_idx: int | None = None, tokens: list | None = None) -> list[dict]:
    """用预测分数对 embedding 的梯度估计 token 重要性。"""
    import torch

    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x[0:1].detach()
    emb_layer = getattr(net, "embedding", None)
    if emb_layer is None:
        raise RuntimeError("模型没有 embedding 层，无法生成 token 归因")
    activation = []
    def _hook(_module, _inputs, out):
        activation.append(out)
        out.retain_grad()

    handle = emb_layer.register_forward_hook(_hook)
    net.eval()
    try:
        logits = net(x)
        idx = int(target_idx if target_idx is not None else logits[0].argmax())
        net.zero_grad(set_to_none=True)
        logits[0, idx].backward()
        grads = activation[0].grad
        importance = grads[0].norm(dim=-1).detach().cpu().numpy()
        importance = normalize_cam(importance)
        return [
            {
                "index": int(i),
                "token": str(tokens[i]) if tokens and i < len(tokens) else f"#{i}",
                "importance": round(float(importance[i]), 5),
            }
            for i in range(len(importance))
        ]
    finally:
        handle.remove()
        net.zero_grad(set_to_none=True)


def plot_token_attribution(attributions: list[dict], path: Path, top_k: int = 20):
    """绘制影响最大的 token；全零或空列表时返回 None。"""
    import matplotlib.pyplot as plt

    from app.plots import save_fig

    items = [a for a in attributions if isinstance(a.get("importance"), (int, float))]
    items = sorted(items, key=lambda a: a["importance"], reverse=True)[:top_k]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(items))))
    names = [str(a["token"]) for a in items]
    values = np.asarray([a["importance"] for a in items], dtype=float)
    ax.barh(range(len(items))[::-1], values, color="#4f7cff")
    ax.set_yticks(range(len(items))[::-1]), ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("token 归因强度")
    ax.set_title(f"Token 重要性 Top {len(items)}")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    return save_fig(fig, Path(path))


def permutation_importance(net, loader, path: Path, classification: bool = True,
                           feature_names: list | None = None, max_samples: int = 512,
                           n_repeats: int = 3, seed: int = 42):
    """对表格 MLP 做基于准确率 / R² 的置换重要性。"""
    import torch

    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
        if sum(x.shape[0] for x in xs) >= max_samples:
            break
    x = torch.cat(xs)[:max_samples].detach().cpu()
    y = torch.cat(ys)[:max_samples].detach().cpu()
    if x.numel() == 0 or y.numel() == 0:
        return []
    net.eval()

    def _metric(pred, target):
        if classification:
            return float((pred.argmax(dim=1) == target).float().mean())
        return -float(torch.mean((pred.reshape(-1) - target.reshape(-1)) ** 2))

    with torch.no_grad():
        baseline = _metric(net(x), y)
    rng = np.random.default_rng(seed)
    n_features = int(x.shape[1])
    names = [str(feature_names[i]) if feature_names and i < len(feature_names) else f"特征 {i + 1}"
             for i in range(n_features)]
    out = []
    for fi in range(n_features):
        drops = []
        for _ in range(n_repeats):
            xp = x.clone()
            xp[:, fi] = xp[rng.permutation(x.shape[0]), fi]
            with torch.no_grad():
                drops.append(float(baseline - _metric(net(xp), y)))
        out.append({"feature": names[fi], "importance": float(np.mean(drops))})
    out = sorted(out, key=lambda a: a["importance"], reverse=True)[:20]
    if not out or all(abs(a["importance"]) < 1e-12 for a in out):
        return []
    from app.plots import plot_feature_importance

    plot_feature_importance([a["feature"] for a in out], [a["importance"] for a in out], Path(path))
    return out
