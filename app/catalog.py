"""模型目录：任务类型 → 模型 → 参数 schema（供前端动态表单与报告文案使用）。"""
from __future__ import annotations

import math

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
                "label": "多层感知机 (MLP)",
                "desc": "全连接神经网络，适合展示深度方法与传统机器学习的对比。",
                "params": {
                    "hidden_sizes": {"type": "string", "default": "128,64", "label": "隐藏层(逗号分隔)"},
                    "learning_rate_init": {"type": "float", "default": 0.001, "min": 0.00001, "max": 1.0, "label": "学习率"},
                    "max_iter": {"type": "int", "default": 300, "min": 20, "max": 5000, "label": "最大迭代轮数"},
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
                "params": {
                    "epochs": {"type": "int", "default": 15, "min": 1, "max": 300, "label": "训练轮数"},
                    "batch_size": {"type": "int", "default": 32, "min": 2, "max": 256, "label": "批大小"},
                    "lr": {"type": "float", "default": 0.001, "min": 0.00001, "max": 1.0, "label": "学习率"},
                    "image_size": {"type": "int", "default": 64, "min": 16, "max": 224, "label": "图像尺寸"},
                },
            },
            "resnet18": {
                "label": "ResNet18 (可迁移学习)",
                "desc": "经典残差网络，可加载 ImageNet 预训练权重做迁移学习，中小型图像数据集首选。",
                "params": {
                    "epochs": {"type": "int", "default": 15, "min": 1, "max": 300, "label": "训练轮数"},
                    "batch_size": {"type": "int", "default": 32, "min": 2, "max": 256, "label": "批大小"},
                    "lr": {"type": "float", "default": 0.001, "min": 0.00001, "max": 1.0, "label": "学习率"},
                    "image_size": {"type": "int", "default": 64, "min": 16, "max": 224, "label": "图像尺寸"},
                    "pretrained": {"type": "bool", "default": True, "label": "加载ImageNet预训练"},
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
    return {"task": task, "model": model, "label": m["label"], "desc": m["desc"], "params": m["params"]}


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
