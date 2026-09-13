"""模型目录：任务类型 → 模型 → 参数 schema（供前端动态表单与报告文案使用）。"""
from __future__ import annotations

import math

_TRAINING_PARAMS: dict = {
    "optimizer": {"type": "choice", "default": "adam",
                  "options": ["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"],
                  "label": "优化器"},
    "lr": {"type": "float", "default": 0.001, "min": 0.00001, "max": 1.0, "label": "学习率"},
    "batch_size": {"type": "int", "default": 32, "min": 1, "max": 256, "label": "批大小"},
    "epochs": {"type": "int", "default": 15, "min": 1, "max": 300, "label": "训练轮数"},
    "scheduler": {"type": "choice", "default": "cosine",
                  "options": ["cosine", "step", "plateau", "none"], "label": "学习率调度"},
    "weight_decay": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.5, "label": "权重衰减 L2"},
    "early_stop_patience": {"type": "int", "default": 0, "min": 0, "max": 100,
                            "label": "早停耐心(0=关闭)"},
    "grad_clip": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0,
                  "label": "梯度裁剪(0=关闭)"},
    "seed": {"type": "int", "default": 42, "min": 0, "max": 999999, "label": "随机种子"},
}


CATALOG: dict = {
    "tabular_classification": {
        "label": "表格数据 · 分类",
        "needs_target": True,
        "needs_text_column": False,
        "models": {
            "logistic_regression": {
                "label": "逻辑回归 (Logistic Regression)",
                "desc": "线性分类基线模型，训练快、可解释性强，通过 L2 正则化系数 C 控制过拟合。",
                "params": {
                    "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100, "label": "正则化系数 C"},
                    "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 20000, "label": "最大迭代次数"},
                },
            },
            "random_forest": {
                "label": "随机森林 (Random Forest)",
                "desc": "基于 Bagging 的树集成模型，对特征尺度不敏感，可输出特征重要性，是表格任务常用强基线。",
                "params": {
                    "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 2000, "label": "树数量"},
                    "max_depth": {"type": "int", "default": 0, "min": 0, "max": 64, "label": "最大深度(0=不限)"},
                },
            },
            "svm": {
                "label": "支持向量机 (SVM)",
                "desc": "基于间隔最大化的分类器，配合 RBF 核可在中小规模数据上取得良好效果，对特征尺度敏感（已内置标准化）。",
                "params": {
                    "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 1000, "label": "惩罚系数 C"},
                    "kernel": {"type": "choice", "default": "rbf", "options": ["rbf", "linear", "poly"], "label": "核函数"},
                },
            },
            "gradient_boosting": {
                "label": "梯度提升树 (GBDT)",
                "desc": "Boosting 类树集成，通常是最强的表格基线之一；学习率与树数量需权衡。",
                "params": {
                    "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 2000, "label": "树数量"},
                    "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0, "label": "学习率"},
                    "max_depth": {"type": "int", "default": 3, "min": 1, "max": 16, "label": "最大深度"},
                },
            },
            "knn": {
                "label": "K近邻 (KNN)",
                "desc": "基于距离投票的惰性学习模型，简单直观，适合作为对照基线。",
                "params": {
                    "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 100, "label": "K 值"},
                },
            },
            "mlp": {
                "label": "MLP 神经网络",
                "desc": "全连接神经网络（可自定义隐藏层/激活/丢弃率），配合优化器与学习率调度训练，适合与机器学习基线对比。",
                "engine": "torch",
                "params": {
                    "hidden_sizes": {"type": "string", "default": "128,64", "label": "隐藏层(逗号分隔)"},
                    "activation": {"type": "choice", "default": "relu",
                                   "options": ["relu", "tanh", "gelu", "leaky_relu"], "label": "激活函数"},
                    "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                    **_TRAINING_PARAMS,
                },
            },
        },
    },
    "tabular_regression": {
        "label": "表格数据 · 回归",
        "needs_target": True,
        "needs_text_column": False,
        "models": {
            "linear_regression": {
                "label": "线性回归 (Linear Regression)",
                "desc": "最基础的回归模型，作为所有回归任务的必选基线。",
                "params": {},
            },
            "ridge": {
                "label": "岭回归 (Ridge)",
                "desc": "带 L2 正则化的线性回归，适合特征间存在多重共线性的场景。",
                "params": {
                    "alpha": {"type": "float", "default": 1.0, "min": 0.0001, "max": 1000, "label": "正则化强度 alpha"},
                },
            },
            "random_forest_regressor": {
                "label": "随机森林回归",
                "desc": "树集成回归模型，能拟合非线性关系并输出特征重要性。",
                "params": {
                    "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 2000, "label": "树数量"},
                    "max_depth": {"type": "int", "default": 0, "min": 0, "max": 64, "label": "最大深度(0=不限)"},
                },
            },
            "gradient_boosting_regressor": {
                "label": "梯度提升回归 (GBDT)",
                "desc": "Boosting 回归集成，通常是结构化数据上最强的回归器之一。",
                "params": {
                    "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 2000, "label": "树数量"},
                    "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0, "label": "学习率"},
                    "max_depth": {"type": "int", "default": 3, "min": 1, "max": 16, "label": "最大深度"},
                },
            },
            "mlp": {
                "label": "MLP 神经网络",
                "desc": "全连接回归网络，可自定义隐藏层与训练策略，衡量深度方法在回归任务上的表现。",
                "engine": "torch",
                "params": {
                    "hidden_sizes": {"type": "string", "default": "128,64", "label": "隐藏层(逗号分隔)"},
                    "activation": {"type": "choice", "default": "relu",
                                   "options": ["relu", "tanh", "gelu", "leaky_relu"], "label": "激活函数"},
                    "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                    **_TRAINING_PARAMS,
                },
            },
        },
    },
    "text_classification": {
        "label": "文本 · 分类",
        "needs_target": True,
        "needs_text_column": True,
        "models": {
            "tfidf_logreg": {
                "label": "TF-IDF + 逻辑回归",
                "desc": "经典文本分类基线：词频-逆文档频率特征加线性分类器，训练快、效果稳。",
                "params": {
                    "max_features": {"type": "int", "default": 20000, "min": 1000, "max": 200000, "label": "词表上限"},
                    "ngram_max": {"type": "int", "default": 2, "min": 1, "max": 4, "label": "N-gram 上限"},
                    "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100, "label": "正则化系数 C"},
                },
            },
            "tfidf_svm": {
                "label": "TF-IDF + SVM",
                "desc": "TF-IDF 特征加线性支持向量机，文本分类长期强基线。",
                "params": {
                    "max_features": {"type": "int", "default": 20000, "min": 1000, "max": 200000, "label": "词表上限"},
                    "ngram_max": {"type": "int", "default": 2, "min": 1, "max": 4, "label": "N-gram 上限"},
                    "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100, "label": "惩罚系数 C"},
                },
            },
            "lstm": {
                "label": "LSTM 文本分类",
                "desc": "长短期记忆网络，捕捉序列长期依赖，适合短文本/评论/情感分类等任务。",
                "engine": "torch",
                "params": {
                    "embedding_dim": {"type": "int", "default": 128, "min": 16, "max": 1024, "label": "词向量维度"},
                    "hidden_dim": {"type": "int", "default": 128, "min": 8, "max": 4096, "label": "隐藏单元数"},
                    "num_layers": {"type": "int", "default": 1, "min": 1, "max": 8, "label": "RNN 层数"},
                    "bidirectional": {"type": "bool", "default": True, "label": "双向 LSTM"},
                    "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                    "max_seq_len": {"type": "int", "default": 128, "min": 4, "max": 2048, "label": "最大序列长度"},
                    "vocab_size": {"type": "int", "default": 5000, "min": 8, "max": 200000, "label": "词表上限"},
                    **_TRAINING_PARAMS,
                },
            },
            "gru": {
                "label": "GRU 文本分类",
                "desc": "门控循环单元，结构比 LSTM 轻量、训练更快，适合作为循环网络的对照模型。",
                "engine": "torch",
                "params": {
                    "embedding_dim": {"type": "int", "default": 128, "min": 16, "max": 1024, "label": "词向量维度"},
                    "hidden_dim": {"type": "int", "default": 128, "min": 8, "max": 4096, "label": "隐藏单元数"},
                    "num_layers": {"type": "int", "default": 1, "min": 1, "max": 8, "label": "RNN 层数"},
                    "bidirectional": {"type": "bool", "default": True, "label": "双向 GRU"},
                    "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                    "max_seq_len": {"type": "int", "default": 128, "min": 4, "max": 2048, "label": "最大序列长度"},
                    "vocab_size": {"type": "int", "default": 5000, "min": 8, "max": 200000, "label": "词表上限"},
                    **_TRAINING_PARAMS,
                },
            },
            "textcnn": {
                "label": "TextCNN 文本分类",
                "desc": "多尺寸卷积核并行提取局部 n-gram 特征，训练快，是文本分类经典深度基线。",
                "engine": "torch",
                "params": {
                    "embedding_dim": {"type": "int", "default": 128, "min": 16, "max": 1024, "label": "词向量维度"},
                    "num_filters": {"type": "int", "default": 64, "min": 8, "max": 1024, "label": "卷积核数量"},
                    "kernel_sizes": {"type": "string", "default": "2,3,4", "label": "卷积核尺寸(逗号分隔)"},
                    "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                    "max_seq_len": {"type": "int", "default": 128, "min": 4, "max": 2048, "label": "最大序列长度"},
                    "vocab_size": {"type": "int", "default": 5000, "min": 8, "max": 200000, "label": "词表上限"},
                    **_TRAINING_PARAMS,
                },
            },
            "transformer": {
                "label": "轻量 Transformer",
                "desc": "多头自注意力编码器（含位置编码与掩码池化），覆盖最新深度学习架构文本路线。",
                "engine": "torch",
                "params": {
                    "d_model": {"type": "int", "default": 128, "min": 16, "max": 1024, "label": "模型维度"},
                    "nhead": {"type": "int", "default": 4, "min": 1, "max": 64, "label": "注意力头数"},
                    "num_layers": {"type": "int", "default": 2, "min": 1, "max": 16, "label": "Encoder 层数"},
                    "dim_feedforward": {"type": "int", "default": 256, "min": 32, "max": 4096, "label": "前馈维度"},
                    "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                    "max_seq_len": {"type": "int", "default": 128, "min": 4, "max": 2048, "label": "最大序列长度"},
                    "vocab_size": {"type": "int", "default": 5000, "min": 8, "max": 200000, "label": "词表上限"},
                    **_TRAINING_PARAMS,
                },
            },
        },
    },
    "image_classification": {
        "label": "图像 · 分类",
        "needs_target": False,
        "needs_text_column": False,
        "models": {
            "cnn": {
                "label": "CNN (3层卷积网络)",
                "desc": "轻量卷积神经网络（Conv-BN-ReLU×3 + 全连接），训练快，适合小数据集与教学演示。",
                "engine": "torch",
                "params": {
                    "conv_channels": {"type": "string", "default": "32,64,128", "label": "卷积通道(逗号分隔)"},
                    "activation": {"type": "choice", "default": "relu",
                                   "options": ["relu", "tanh", "gelu", "leaky_relu"], "label": "激活函数"},
                    "dropout": {"type": "float", "default": 0.3, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                    "image_size": {"type": "int", "default": 64, "min": 16, "max": 224, "label": "图像尺寸"},
                    **_TRAINING_PARAMS,
                },
            },
            "resnet18": {
                "label": "ResNet18 (可迁移学习)",
                "desc": "经典残差网络，可加载 ImageNet 预训练权重做迁移学习，中小型图像数据集首选。",
                "engine": "torch",
                "params": {
                    "image_size": {"type": "int", "default": 64, "min": 16, "max": 224, "label": "图像尺寸"},
                    "pretrained": {"type": "bool", "default": True, "label": "加载ImageNet预训练"},
                    "freeze_backbone": {"type": "bool", "default": False, "label": "冻结骨干只训分类头"},
                    **_TRAINING_PARAMS,
                },
            },
        },
    },
}


def all_models_flat() -> list[dict]:
    out = []
    for task, t in CATALOG.items():
        for key, m in t["models"].items():
            out.append({"task": task, "task_label": t["label"], "key": key, "label": m["label"], "desc": m["desc"]})
    return out


def get_model_spec(task: str, model: str) -> dict | None:
    t = CATALOG.get(task)
    if not t:
        return None
    m = t["models"].get(model)
    if not m:
        return None
    return {"task": task, "model": model, "label": m["label"], "desc": m["desc"],
            "engine": m.get("engine") or "sklearn", "params": m["params"]}


def _coerce(pschema: dict, raw):
    """按 schema 转换单个参数：类型、有限性、min/max 夹取。

    越界一律夹到合法区间而不是抛错 —— 前端滑杆已经限定了范围，这里兜住的是
    直接打 API 或脚本误传，避免 epochs=99999 之类把机器挂死。
    """
    default = pschema.get("default")
    ptype = pschema.get("type")
    if ptype == "bool":
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if ptype == "choice":
        return raw if raw in (pschema.get("options") or []) else default
    if ptype not in ("int", "float"):
        return raw
    try:
        v = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(v):
        return default
    lo, hi = pschema.get("min"), pschema.get("max")
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return int(v) if ptype == "int" else v


def sanitize_params(spec: dict, raw: dict | None) -> dict:
    """白名单 + 类型 + 范围三重清洗：只保留 schema 里声明过的键。"""
    out: dict = {}
    raw = raw or {}
    for key, pschema in (spec.get("params") or {}).items():
        value = _coerce(pschema, raw.get(key, pschema.get("default")))
        if value is None:
            continue
        out[key] = value
    return out
