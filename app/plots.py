"""matplotlib 绘图工具：统一中文字体与样式，训练子进程与后端共用。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACCENT = "#4f7cff"
ACCENT2 = "#ff7f50"
GREEN = "#2ecc71"
RED = "#e74c3c"
GRID = dict(alpha=0.25, linestyle="--")


def save_fig(fig, path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(cm, classes, path, normalize: bool = True) -> str:
    cm = np.array(cm, dtype=float)
    disp = cm.astype(float).copy()
    if normalize:
        row = disp.sum(axis=1, keepdims=True)
        row[row == 0] = 1
        disp = disp / row
    n = len(classes)
    size = max(4.2, 0.6 * n)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)), ax.set_yticks(range(n))
    ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(c) for c in classes], fontsize=8)
    ax.set_xlabel("预测类别"), ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（按行归一化）")
    thr = disp.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.0f}\n{disp[i, j]*100:.1f}%", ha="center", va="center",
                    fontsize=7, color="white" if disp[i, j] > thr else "#333")
    fig.colorbar(im, fraction=0.046)
    return save_fig(fig, path)


def plot_roc(fpr, tpr, auc_value, path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], color="#999", lw=1, linestyle="--", label="随机猜测")
    ax.set_xlabel("假正率 FPR"), ax.set_ylabel("真正率 TPR")
    ax.set_title("ROC 曲线")
    ax.grid(**GRID), ax.legend()
    return save_fig(fig, path)


def plot_feature_importance(names, values, path, top_k: int = 20) -> str | None:
    if values is None or len(values) == 0:
        return None
    idx = np.argsort(values)[::-1][:top_k]
    names = [str(names[i]) for i in idx]
    values = np.array(values)[idx]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(idx))))
    ax.barh(range(len(idx))[::-1], values, color=ACCENT)
    ax.set_yticks(range(len(idx))[::-1]), ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"特征重要性 Top {len(idx)}")
    ax.grid(axis="x", **GRID)
    return save_fig(fig, path)


def plot_class_balance(counts: dict, path) -> str:
    fig, ax = plt.subplots(figsize=(max(4.5, 0.6 * len(counts)), 3.8))
    keys = [str(k) for k in counts]
    vals = [counts[k] for k in counts]
    ax.bar(keys, vals, color=ACCENT)
    for i, v in enumerate(vals):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.set_title("类别分布")
    ax.grid(axis="y", **GRID)
    if len(keys) > 8:
        ax.tick_params(axis="x", rotation=45)
    return save_fig(fig, path)


def plot_numeric_hist(series_dict: dict, path, max_cols: int = 12) -> str:
    """series_dict: {col: np.ndarray}"""
    cols = list(series_dict)[:max_cols]
    n = len(cols)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // ncol][i % ncol]
        ax.hist(np.asarray(series_dict[col], dtype=float), bins=25, color=ACCENT, alpha=0.85)
        ax.set_title(str(col), fontsize=9)
        ax.grid(**GRID)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("数值特征分布直方图", y=1.02)
    return save_fig(fig, path)


def plot_correlation_heatmap(corr_df, path) -> str | None:
    if corr_df.shape[0] < 2 or corr_df.shape[0] > 30:
        return None
    fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * corr_df.shape[1]), max(4, 0.5 * corr_df.shape[0])))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(corr_df.shape[1]))
    ax.set_yticks(range(corr_df.shape[0]))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr_df.values[i, j]) > 0.6 else "#333")
    ax.set_title("数值特征相关性")
    fig.colorbar(im, fraction=0.046)
    return save_fig(fig, path)


def plot_sample_images(image_paths, labels, path, max_n: int = 12) -> str:
    from PIL import Image
    n = min(max_n, len(image_paths))
    ncol = min(6, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.8 * nrow), squeeze=False)
    for i in range(n):
        ax = axes[i // ncol][i % ncol]
        img = Image.open(image_paths[i]).convert("RGB")
        ax.imshow(img), ax.set_title(str(labels[i]), fontsize=8), ax.axis("off")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("数据样例", y=1.02)
    return save_fig(fig, path)


def plot_training_curves(metrics: list[dict], path) -> str:
    epochs = [m["epoch"] for m in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [m["train_loss"] for m in metrics], color=ACCENT, label="train_loss")
    if any(m.get("val_loss") is not None for m in metrics):
        axes[0].plot(epochs, [m.get("val_loss") for m in metrics], color=ACCENT2, label="val_loss")
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("loss"), axes[0].set_title("损失曲线")
    axes[0].grid(**GRID), axes[0].legend()
    if any(m.get("val_acc") is not None for m in metrics):
        axes[1].plot(epochs, [m.get("val_acc") for m in metrics], color=GREEN, label="val_acc")
        if any(m.get("train_acc") is not None for m in metrics):
            axes[1].plot(epochs, [m.get("train_acc") for m in metrics], color=ACCENT, label="train_acc", alpha=0.7)
        axes[1].set_xlabel("epoch"), axes[1].set_ylabel("accuracy"), axes[1].set_title("准确率曲线")
        axes[1].grid(**GRID), axes[1].legend()
    else:
        axes[1].plot(epochs, [m.get("train_loss") for m in metrics], color=ACCENT)
        axes[1].set_title("训练损失")
    return save_fig(fig, path)


def plot_cv_scores(fold_scores: list[float], path) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    xs = list(range(1, len(fold_scores) + 1))
    ax.bar(xs, fold_scores, color=ACCENT, width=0.55)
    mean = float(np.mean(fold_scores))
    ax.axhline(mean, color=RED, linestyle="--", label=f"均值 {mean:.4f}")
    for x, s in zip(xs, fold_scores):
        ax.text(x, s, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("交叉验证折"), ax.set_title("K 折交叉验证得分")
    ax.set_ylim(min(0, min(fold_scores) - 0.05), 1.02)
    ax.grid(axis="y", **GRID), ax.legend()
    return save_fig(fig, path)
