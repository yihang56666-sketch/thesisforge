"""数据中心：内置数据集一键载入、本地导入、URL 下载（SSRF 防护）、自动 EDA。"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS_DIR, UPLOADS_DIR
from .config import APP_VERSION
from . import plots
from .security import SafeURLError, validate_public_http_url

MAX_DOWNLOAD_BYTES = 1 * 1024**3  # 1GB

# 数据集 ID 白名单：字母/数字/中文/下划线/连字符（_slug 只会产出这一类）
_DS_ID_RE = re.compile(r"^[0-9a-zA-Z\u4e00-\u9fff_-]{1,80}$")


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", name.strip()).strip("-").lower()
    # 截断到 60 字符：ID 需匹配 _DS_ID_RE，也避免中文长文件名撞 Windows 路径长度上限
    return (s[:60].rstrip("-")) or "dataset"


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 内置数据集
BUILTIN_CATALOG: dict[str, dict] = {
    "iris": {
        "name": "鸢尾花 Iris（多分类·入门）",
        "loader": lambda: _from_sklearn("load_iris"),
        "task": "tabular_classification",
        "desc": "150 条样本、4 个特征、3 类。毕设入门/流程验证首选。",
    },
    "wine": {
        "name": "葡萄酒 Wine（多分类）",
        "loader": lambda: _from_sklearn("load_wine"),
        "task": "tabular_classification",
        "desc": "178 条样本、13 个理化特征、3 类。",
    },
    "breast_cancer": {
        "name": "乳腺癌威斯康星（二分类·医学）",
        "loader": lambda: _from_sklearn("load_breast_cancer"),
        "task": "tabular_classification",
        "desc": "569 条样本、30 个特征、良性/恶性二分类，医学场景经典。",
    },
    "digits": {
        "name": "手写数字 Digits（8x8 灰度图）",
        "loader": lambda: _from_sklearn("load_digits"),
        "task": "tabular_classification",
        "desc": "1797 条 8x8 手写数字位图展平特征，10 类，可当图像或表格任务。",
    },
    "california_housing": {
        "name": "加州房价（回归）",
        "loader": lambda: _from_sklearn("fetch_california_housing"),
        "task": "tabular_regression",
        "desc": "20640 条 census 街区数据，预测房价中位数，经典回归任务。",
    },
    "diabetes": {
        "name": "糖尿病进展（回归·小数据）",
        "loader": lambda: _from_sklearn("load_diabetes"),
        "task": "tabular_regression",
        "desc": "442 条样本、10 个基线变量，回归入门。",
    },
}

_SK_FUNCS = None


def _from_sklearn(fn_name: str) -> tuple[pd.DataFrame, str]:
    global _SK_FUNCS
    if _SK_FUNCS is None:
        import sklearn.datasets as sd

        _SK_FUNCS = {
            "load_iris": sd.load_iris,
            "load_wine": sd.load_wine,
            "load_breast_cancer": sd.load_breast_cancer,
            "load_digits": sd.load_digits,
            "load_diabetes": sd.load_diabetes,
            "fetch_california_housing": sd.fetch_california_housing,
        }
    data = _SK_FUNCS[fn_name]()
    if hasattr(data, "target_names"):
        target = [data.target_names[i] for i in data.target]
    else:
        target = data.target
    df = pd.DataFrame(data.data, columns=list(data.feature_names))
    df["target"] = target
    return df, "target"


# ---------------------------------------------------------------- 元数据
def dataset_dir(ds_id: str) -> Path:
    """数据集目录。ID 只允许 _slug 的产物字符集，模块内部自带穿越防护，
    不依赖调用方（main.ds_dir_of）先校验。"""
    ds_id = str(ds_id or "")
    if not _DS_ID_RE.match(ds_id):
        raise ValueError("非法数据集 ID")
    p = (DATASETS_DIR / ds_id).resolve()
    root = DATASETS_DIR.resolve()
    if p == root or not p.is_relative_to(root):
        raise ValueError("非法数据集路径")
    return p


def load_meta(ds_id: str) -> dict:
    p = dataset_dir(ds_id) / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"数据集不存在: {ds_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def _save_meta(ds_dir: Path, meta: dict) -> dict:
    (ds_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def list_datasets() -> list[dict]:
    out = []
    if DATASETS_DIR.exists():
        for d in sorted(DATASETS_DIR.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                try:
                    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                    out.append(meta)
                except Exception:
                    continue
    return out


# ---------------------------------------------------------------- EDA
def run_eda(ds_id: str) -> list[str]:
    """生成 EDA 图表与统计，返回图表文件名列表。"""
    meta = load_meta(ds_id)
    ds_dir = dataset_dir(ds_id)
    eda_dir = ds_dir / "eda"
    eda_dir.mkdir(exist_ok=True)
    files: list[str] = []

    if meta["type"] == "image":
        img_root = ds_dir / "images"
        counts, paths, labels = {}, [], []
        for cls_dir in sorted(img_root.iterdir()):
            if cls_dir.is_dir():
                imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
                counts[cls_dir.name] = len(imgs)
                for p in imgs[:2]:
                    paths.append(str(p))
                    labels.append(cls_dir.name)
        if counts:
            files.append(Path(plots.plot_class_balance(counts, eda_dir / "class_balance.png")).name)
            meta["n_classes"] = len(counts)
            meta["classes"] = list(counts)
            meta["n_images"] = int(sum(counts.values()))
            if paths:
                files.append(Path(plots.plot_sample_images(paths, labels, eda_dir / "samples.png")).name)
        meta["stats"] = {"class_counts": counts}
    else:
        df = pd.read_csv(ds_dir / "dataset.csv")
        target = meta.get("target")
        if target and target in df.columns:
            counts = df[target].astype(str).value_counts().to_dict()
            files.append(Path(plots.plot_class_balance(counts, eda_dir / "class_balance.png")).name)
            meta["n_classes"] = len(counts)
            meta["classes"] = list(map(str, counts))
            class_counts = counts
        else:
            class_counts = {}
        num_df = df.select_dtypes(include=[np.number])
        desc = {}
        if num_df.shape[1] >= 1:
            sample = num_df.sample(n=min(len(num_df), 3000), random_state=42)
            files.append(Path(plots.plot_numeric_hist({c: sample[c].dropna().values for c in sample.columns}, eda_dir / "histograms.png")).name)
            corr = num_df.corr(numeric_only=True)
            p = plots.plot_correlation_heatmap(corr, eda_dir / "correlation.png")
            if p:
                files.append(Path(p).name)
            desc = num_df.describe().round(4).to_dict()
        col_types = {c: ("numeric" if str(df[c].dtype) in ("int64", "float64", "int32", "float32") else "categorical") for c in df.columns}
        missing = {c: int(df[c].isna().sum()) for c in df.columns}
        meta["stats"] = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "column_types": col_types,
            "missing": missing,
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "describe": {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()} for k, v in desc.items()},
        }
        meta["n_rows"] = int(len(df))
        meta["columns"] = list(df.columns)
    meta["eda_files"] = files
    meta["eda_done"] = True
    _save_meta(ds_dir, meta)
    return files


# ---------------------------------------------------------------- 载入/导入/下载
def load_builtin(name: str) -> dict:
    if name not in BUILTIN_CATALOG:
        raise ValueError(f"未知内置数据集: {name}")
    df, target = BUILTIN_CATALOG[name]["loader"]()
    ds_id = _slug(name)
    ds_dir = dataset_dir(ds_id)
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True)
    df.to_csv(ds_dir / "dataset.csv", index=False)
    meta = {
        "id": ds_id,
        "name": BUILTIN_CATALOG[name]["name"],
        "source": "builtin",
        "type": "tabular",
        "task": BUILTIN_CATALOG[name]["task"],
        "target": target,
        "columns": list(df.columns),
        "n_rows": int(len(df)),
        "created_at": _now(),
        "desc": BUILTIN_CATALOG[name]["desc"],
    }
    _save_meta(ds_dir, meta)
    run_eda(ds_id)
    return load_meta(ds_id)


def _safe_extract_zip(zf_source, dest: Path) -> None:
    """防 zip-slip：逐个成员规范化校验，拒绝 ..、盘符/UNC、绝对路径，结果限制在 dest 内。"""
    dest = dest.resolve()
    with zipfile.ZipFile(zf_source) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            normalized = member.filename.replace("\\", "/")
            if ":" in normalized or normalized.startswith("/"):
                continue  # 拒绝盘符/UNC/绝对路径成员
            target = (dest / member.filename).resolve()
            if ".." in target.parts or not target.is_relative_to(dest):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def import_bytes(filename: str, content: bytes) -> dict:
    """兼容入口：内存字节流导入。新代码优先用 import_path，避免整包驻留内存。"""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".bin"
    tmp = UPLOADS_DIR / f"mem-{datetime.datetime.now():%Y%m%d%H%M%S%f}-{os.getpid()}{suffix}"
    tmp.write_bytes(content)
    try:
        return import_path(filename, tmp)
    finally:
        tmp.unlink(missing_ok=True)


def import_path(filename: str, stored: Path) -> dict:
    """从已落盘的临时文件导入 CSV/Excel（表格）或 zip（图像分类数据集，内部为 类别名/图片）。

    stored 必须是完整写好的文件，解析成功后由本函数负责删除。
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    base = _slug(Path(filename).stem) or "imported"
    ds_id, i = base, 1
    while dataset_dir(ds_id).exists():
        ds_id = f"{base}-{i}"
        i += 1
    ds_dir = dataset_dir(ds_id)
    ds_dir.mkdir(parents=True)

    def _fail(msg: str):
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise ValueError(msg)

    if suffix in (".csv", ".xlsx", ".xls"):
        try:
            df = pd.read_csv(stored) if suffix == ".csv" else pd.read_excel(stored)
        except Exception as e:
            _fail(f"表格解析失败: {e}")
        if df.empty or df.shape[1] < 2:
            _fail("表格至少需要 2 列（特征列 + 标签列）")
        df.to_csv(ds_dir / "dataset.csv", index=False)
        n_rows, n_cols = int(len(df)), int(df.shape[1])
        meta = {
            "id": ds_id, "name": Path(filename).stem, "source": "imported", "type": "tabular",
            "task": "tabular_classification", "target": df.columns[-1],
            "columns": list(df.columns), "n_rows": n_rows, "created_at": _now(),
            "desc": f"本地导入表格，{n_rows} 行 × {n_cols} 列。",
        }
        df = None  # 提前释放，EDA 阶段不必再持有整表
        _save_meta(ds_dir, meta)
    elif suffix == ".zip":
        img_root = ds_dir / "images"
        try:
            _safe_extract_zip(stored, img_root)
        except Exception as e:
            _fail(f"zip 解压失败: {e}")
        classes = [d.name for d in sorted(img_root.iterdir()) if d.is_dir()] if img_root.exists() else []
        if len(classes) < 2:
            _fail("zip 内需要按『类别文件夹/图片』组织，且至少 2 个类别")
        meta = {
            "id": ds_id, "name": Path(filename).stem, "source": "imported", "type": "image",
            "task": "image_classification", "target": None, "columns": [],
            "created_at": _now(), "desc": f"图像分类数据集，{len(classes)} 个类别。",
        }
        _save_meta(ds_dir, meta)
    else:
        _fail("仅支持 .csv/.xlsx/.xls 表格或 .zip 图像数据集")
    stored.unlink(missing_ok=True)
    run_eda(ds_id)
    return load_meta(ds_id)


def download_url(url: str, name: str | None = None) -> dict:
    """从公网 URL 下载数据集（http/https，逐跳 SSRF 校验，限 1GB）。"""
    url = validate_public_http_url(url)
    filename = Path(urlparse_path(url)).name or "download.zip"
    if not Path(filename).suffix.lower():
        filename += ".zip"
    dest = _fetch_to_tempfile(url, name or filename)
    try:
        return import_path(name or filename, dest)
    finally:
        dest.unlink(missing_ok=True)


def urlparse_path(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).path


def _fetch_to_tempfile(url: str, filename: str) -> Path:
    """流式下载到临时文件：边下边累计字节数，超限立即中断，不把整个响应读进内存。"""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".bin"
    tmp = UPLOADS_DIR / f"dl-{datetime.datetime.now():%Y%m%d%H%M%S%f}-{os.getpid()}{suffix}"
    try:
        _stream_download(url, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _stream_download(url: str, tmp: Path) -> None:
    import httpx
    from urllib.parse import urljoin

    with httpx.Client(follow_redirects=False, timeout=httpx.Timeout(30, read=180),
                      headers={"User-Agent": f"ThesisForge/{APP_VERSION}"}) as client:
        hops = 0
        current = url
        while True:
            validate_public_http_url(current)
            with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    hops += 1
                    if hops > 5:
                        raise SafeURLError("重定向次数过多")
                    loc = resp.headers.get("location", "")
                    if not loc:
                        raise SafeURLError("重定向缺少目标地址")
                    current = urljoin(current, loc)
                    continue
                resp.raise_for_status()
                declared = resp.headers.get("content-length", "")
                if declared.strip().isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("文件超过 1GB 下载上限")
                total = 0
                with tmp.open("wb") as out:
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise ValueError("文件超过 1GB 下载上限")
                        out.write(chunk)
                    out.flush()
                if total == 0:
                    raise ValueError("下载内容为空")
                return


def _fetch(url: str) -> bytes:
    """整包读入内存的下载（仅供测试与小文件使用）。"""
    import httpx
    from urllib.parse import urljoin

    with httpx.Client(follow_redirects=False, timeout=120,
                      headers={"User-Agent": f"ThesisForge/{APP_VERSION}"}) as client:
        hops = 0
        current = url
        while True:
            validate_public_http_url(current)
            resp = client.get(current)
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                hops += 1
                if hops > 5:
                    raise SafeURLError("重定向次数过多")
                loc = resp.headers.get("location", "")
                if not loc:
                    raise SafeURLError("重定向缺少目标地址")
                current = urljoin(current, loc)
                continue
            resp.raise_for_status()
            if len(resp.content) > MAX_DOWNLOAD_BYTES:
                raise ValueError("文件超过 1GB 下载上限")
            return resp.content


def preview(ds_id: str, rows: int = 20) -> dict:
    meta = load_meta(ds_id)
    if meta["type"] == "image":
        img_root = dataset_dir(ds_id) / "images"
        sample = []
        for cls_dir in sorted(img_root.iterdir()):
            if cls_dir.is_dir():
                for p in list(cls_dir.iterdir())[:3]:
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                        sample.append({"class": cls_dir.name, "file": p.name})
        return {"type": "image", "sample": sample[:rows]}
    df = pd.read_csv(dataset_dir(ds_id) / "dataset.csv")
    return {
        "type": "tabular",
        "columns": list(df.columns),
        "rows": df.head(rows).fillna("").astype(str).values.tolist(),
        "total": int(len(df)),
    }


def delete_dataset(ds_id: str) -> None:
    d = dataset_dir(ds_id)
    if not d.exists():
        raise FileNotFoundError(ds_id)
    shutil.rmtree(d)
