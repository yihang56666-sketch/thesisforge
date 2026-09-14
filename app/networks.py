"""神经网络注册表：表格 MLP、图像 CNN/ResNet18、文本 LSTM/GRU/TextCNN/轻量 Transformer。

torch 只在真正构建模型时导入，保证普通后端（health/models/向导）不因缺 PyTorch 而崩溃。
"""
from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn
except ImportError:  # torch 可选：没有 torch 时仍可安全 import，构建函数再报错
    class _FakeModule:
        pass

    nn = type("nn", (), {"Module": _FakeModule})


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def _int(value, default: int, lo: int | None = None, hi: int | None = None) -> int:
    try:
        v = int(float(value))
    except (TypeError, ValueError, OverflowError):
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _float(value, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        import math

        v = float(value)
        if not math.isfinite(v):
            v = default
    except (TypeError, ValueError, OverflowError):
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _int_list(value, default: list[int], default_str: str = "") -> list[int]:
    if isinstance(value, list):
        items = value
    elif value in (None, ""):
        items = [int(x.strip()) for x in default_str.split(",") if x.strip()] or default
    else:
        items = [x.strip() for x in str(value).split(",") if x.strip()]
    out = []
    for item in items:
        n = _int(item, 0, lo=1, hi=4096)
        if n:
            out.append(n)
    return out or list(default)


def _activation(name: str, nn):
    key = str(name or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.1),
    }.get(key, nn.ReLU())


class MLP(nn.Module):
    """全连接网络，表格分类/回归共用。"""

    def __init__(self, in_features, hidden_sizes, out_features, activation="relu", dropout=0.2):
        nn = _torch()[1]
        super().__init__()
        sizes = [in_features] + list(hidden_sizes)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(_activation(activation, nn))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(sizes[-1], out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ImageCNN(nn.Module):
    """轻量卷积网络：Conv-BN-Activation-MaxPool 循环 + 全局池化 + 全连接。"""

    def __init__(self, num_classes, conv_channels=(32, 64, 128), activation="relu", dropout=0.3):
        _, nn, _ = _torch()
        super().__init__()
        channels = [int(c) for c in conv_channels]
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                _activation(activation, nn),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(in_ch, num_classes),
        ]
        self.features = nn.Sequential(*layers[:-2])
        self.classifier = nn.Sequential(*layers[-2:])

    def forward(self, x):
        return self.classifier(self.features(x))


class TextRNN(nn.Module):
    """LSTM/GRU 文本分类：Embedding -> 双向(可选)RNN -> 掩码均值池化 -> 全连接。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes,
                 bidirectional=False, dropout=0.2, rnn_type="lstm"):
        torch, nn, _ = _torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        out, _ = self.rnn(self.embedding(x))
        out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(self.dropout(out))


class TextCNN(nn.Module):
    """TextCNN：Embedding -> 多尺寸 Conv1d -> 全局最大池化 -> 拼接 -> 全连接。"""

    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, num_classes, dropout=0.2):
        torch, nn, F = _torch()
        super().__init__()
        self._torch = torch
        self._F = F
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, ks, padding=ks // 2) for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        F = self._F
        pooled = [F.adaptive_max_pool1d(F.relu(conv(emb)), 1).squeeze(2) for conv in self.convs]
        cat = self.dropout(self._torch.cat(pooled, dim=1))
        return self.fc(cat)


class LightTransformer(nn.Module):
    """轻量 Transformer 文本分类：Embedding + 可学习位置编码 + TransformerEncoder + 池化。"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward,
                 num_classes, dropout=0.1, max_seq_len=128):
        torch, nn, _ = _torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        seq_len = x.size(1)
        mask = x != 0
        emb = self.embedding(x) + self.pos[:, :seq_len, :]
        out = self.encoder(emb, src_key_padding_mask=~mask)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.fc(self.dropout(pooled))


class TimeSeriesRNN(nn.Module):
    """LSTM/GRU 时间序列回归：连续特征输入 -> RNN -> 未来 horizon 步。"""

    def __init__(self, in_features, hidden_dim, num_layers, horizon,
                 dropout=0.1, rnn_type="lstm"):
        _, nn, _ = _torch()
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            in_features, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))


class TimeSeriesTransformer(nn.Module):
    """Transformer 时间序列回归：线性投影 + 位置编码 + Encoder 均值池化。"""

    def __init__(self, in_features, d_model, nhead, num_layers, dim_feedforward,
                 horizon, dropout=0.1, max_seq_len=512):
        torch, nn, _ = _torch()
        super().__init__()
        self.input = nn.Linear(in_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        seq_len = x.size(1)
        out = self.input(x) + self.pos[:, :seq_len, :]
        out = self.encoder(out).mean(dim=1)
        return self.fc(self.dropout(out))


def build_resnet18(num_classes, pretrained=False, freeze_backbone=False):
    try:
        from torchvision import models
        from torchvision.models import ResNet18_Weights
    except ImportError:
        raise RuntimeError("未安装 torchvision，无法构建 ResNet18")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False
        for p in net.fc.parameters():
            p.requires_grad = True
    return net


def build_model(task: str, model: str, params: dict | None = None, **dims: Any):
    """按任务与模型 key 构建网络；缺失维度用合理默认值，方便只估算参数量。"""
    _ = _torch()
    p = params or {}
    num_classes = _int(dims.get("num_classes"), 2, lo=2, hi=10000)
    if task in ("tabular_classification", "tabular_regression"):
        if model != "mlp":
            raise ValueError(f"表格任务不支持的模型: {model}")
        in_features = _int(dims.get("num_features"), 16, lo=1, hi=100000)
        out_features = num_classes if task == "tabular_classification" else 1
        hidden = _int_list(p.get("hidden_sizes"), [128, 64], "128,64")
        return MLP(
            in_features, hidden, out_features,
            activation=p.get("activation", "relu"),
            dropout=_float(p.get("dropout"), 0.2, 0.0, 0.95),
        )
    if task == "image_classification":
        channels = _int_list(p.get("conv_channels"), [32, 64, 128], "32,64,128")
        if model == "cnn":
            return ImageCNN(
                num_classes, channels,
                activation=p.get("activation", "relu"),
                dropout=_float(p.get("dropout"), 0.3, 0.0, 0.95),
            )
        if model == "resnet18":
            return build_resnet18(
                num_classes,
                pretrained=bool(p.get("pretrained", False)),
                freeze_backbone=bool(p.get("freeze_backbone", False)),
            )
        raise ValueError(f"图像任务不支持的模型: {model}")
    if task == "text_classification":
        num_tokens = _int(dims.get("num_tokens"), 5000, lo=2, hi=1000000)
        max_seq_len = _int(dims.get("max_seq_len"), 128, lo=4, hi=2048)
        embed_dim = _int(p.get("embedding_dim"), 128, lo=16, hi=1024)
        dropout = _float(p.get("dropout"), 0.2, 0.0, 0.95)
        if model in ("lstm", "gru"):
            hidden_dim = _int(p.get("hidden_dim"), 128, lo=8, hi=4096)
            num_layers = _int(p.get("num_layers"), 1, lo=1, hi=8)
            bidirectional = bool(p.get("bidirectional", False))
            return TextRNN(num_tokens, embed_dim, hidden_dim, num_layers, num_classes,
                           bidirectional=bidirectional, dropout=dropout, rnn_type=model)
        if model == "textcnn":
            num_filters = _int(p.get("num_filters"), 64, lo=8, hi=1024)
            kernels = _int_list(p.get("kernel_sizes"), [2, 3, 4], "2,3,4")
            return TextCNN(num_tokens, embed_dim, num_filters, kernels, num_classes, dropout=dropout)
        if model == "transformer":
            d_model = _int(p.get("d_model"), 128, lo=16, hi=1024)
            nhead = _int(p.get("nhead"), 4, lo=1, hi=64)
            num_layers = _int(p.get("num_layers"), 2, lo=1, hi=16)
            dim_ff = _int(p.get("dim_feedforward"), 256, lo=32, hi=4096)
            return LightTransformer(num_tokens, d_model, nhead, num_layers, dim_ff,
                                    num_classes, dropout=dropout, max_seq_len=max_seq_len)
        raise ValueError(f"文本任务不支持的模型: {model}")
    if task == "time_series_forecasting":
        in_features = _int(dims.get("num_features"), 1, lo=1, hi=100000)
        horizon = _int(p.get("horizon"), 1, lo=1, hi=128)
        dropout = _float(p.get("dropout"), 0.1, 0.0, 0.95)
        if model in ("lstm", "gru"):
            return TimeSeriesRNN(
                in_features, _int(p.get("hidden_dim"), 64, lo=8, hi=1024),
                _int(p.get("num_layers"), 1, lo=1, hi=8), horizon,
                dropout=dropout, rnn_type=model,
            )
        if model == "transformer":
            return TimeSeriesTransformer(
                in_features, _int(p.get("d_model"), 64, lo=16, hi=1024),
                _int(p.get("nhead"), 4, lo=1, hi=16),
                _int(p.get("num_layers"), 2, lo=1, hi=16),
                _int(p.get("dim_feedforward"), 128, lo=32, hi=4096),
                horizon, dropout=dropout,
                max_seq_len=_int(p.get("lookback"), 12, lo=2, hi=512),
            )
        raise ValueError(f"时间序列不支持的模型: {model}")
    raise ValueError(f"未知任务类型: {task}")


def count_parameters(model) -> int:
    return sum(int(x.numel()) for x in model.parameters())
