"""向导数据预处理配置：默认值、清洗与表格特征转换器（torch/sklearn 共用）。"""
from __future__ import annotations


DATA_PREP_DEFAULTS: dict = {
    "missing": "impute",
    "impute": "median",
    "scale": "standard",
    "encode": "onehot",
    "augment": "flip_rotate",
    "split_first": True,
    "test_size": 0.2,
    "val_split": 0.2,
    "seed": 42,
}

_MISSING_CHOICES = {"impute", "drop", "keep"}
_IMPUTE_CHOICES = {"median", "mean", "most_frequent", "zero", "none"}
_SCALE_CHOICES = {"standard", "minmax", "robust", "none"}
_ENCODE_CHOICES = {"onehot", "label", "none"}
_AUGMENT_CHOICES = {"none", "flip_rotate", "crop", "color_jitter", "all"}


def clean_prep(raw) -> dict:
    """清洗前端提交的预处理配置，未知键丢弃，非法值回退默认。"""
    clean = dict(DATA_PREP_DEFAULTS)
    if not isinstance(raw, dict):
        return clean
    for key, choices in (
        ("missing", _MISSING_CHOICES),
        ("impute", _IMPUTE_CHOICES),
        ("scale", _SCALE_CHOICES),
        ("encode", _ENCODE_CHOICES),
        ("augment", _AUGMENT_CHOICES),
    ):
        value = raw.get(key)
        if isinstance(value, str) and value in choices:
            clean[key] = value
    if "split_first" in raw:
        v = str(raw["split_first"]).strip().lower()
        clean["split_first"] = v in ("1", "true", "yes", "on")
    for key in ("test_size", "val_split"):
        try:
            clean[key] = min(max(float(raw[key]), 0.05), 0.5)
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        clean["seed"] = max(0, min(int(raw["seed"]), 2**31 - 1))
    except (TypeError, ValueError, OverflowError):
        pass
    return clean


def build_tabular_transformer(prep: dict, num_cols: list, cat_cols: list):
    """按向导策略构造 sklearn 表格预处理 ColumnTransformer。

    返回的 transformer 同时可供 PyTorch 数据加载与 sklearn 训练脚本使用。
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler,
    )

    missing = prep.get("missing", "impute")
    impute = prep.get("impute", "median")

    def _imputer(kind: str):
        if missing == "drop":
            return None
        if missing == "keep" or impute == "none":
            return SimpleImputer(strategy="constant",
                                 fill_value=0.0 if kind == "num" else "missing")
        if impute == "zero":
            return SimpleImputer(strategy="constant",
                                 fill_value=0.0 if kind == "num" else "missing")
        if kind == "cat" and impute in ("median", "mean"):
            # 中位数/均值只适用于数值列，类别列统一改用众数填充。
            return SimpleImputer(strategy="most_frequent")
        if impute in ("median", "mean", "most_frequent"):
            return SimpleImputer(strategy=impute)
        return None

    def _scaler():
        key = prep.get("scale", "standard")
        if key == "minmax":
            return MinMaxScaler()
        if key == "robust":
            return RobustScaler()
        if key == "none":
            return None
        return StandardScaler()

    def _encoder():
        key = prep.get("encode", "onehot")
        if key == "label":
            try:
                return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            except TypeError:
                return OrdinalEncoder()
        if key == "none":
            return None
        return OneHotEncoder(handle_unknown="ignore")

    num_parts: list = []
    num_imp = _imputer("num")
    if num_imp is not None:
        num_parts.append(("imp", num_imp))
    num_sc = _scaler()
    if num_sc is not None:
        num_parts.append(("sc", num_sc))
    cat_parts: list = []
    cat_imp = _imputer("cat")
    if cat_imp is not None:
        cat_parts.append(("imp", cat_imp))
    cat_enc = _encoder()
    if cat_enc is not None:
        cat_parts.append(("enc", cat_enc))
    return ColumnTransformer([
        ("num", Pipeline(num_parts) if num_parts else "passthrough", num_cols),
        ("cat", Pipeline(cat_parts) if cat_parts else "passthrough", cat_cols),
    ])


def to_dense(matrix):
    """把 sklearn 稀疏输出转成 float32 稠密数组，非数值类型给出可操作提示。"""
    import numpy as np

    arr = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)
    try:
        return arr.astype(np.float32)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "预处理后仍含非数值特征：请把步骤 3 的类别编码改为「独热编码」或「标签编码」，"
            "或只对全数值数据选择「不编码」。"
        ) from e
