"""本机模型权重扫描：识别用户已下载/拷贝的检测与分割权重。"""
from __future__ import annotations

import copy
import hashlib
import os
import re
from pathlib import Path

from .config import DATA_DIR

from . import catalog

_WEIGHT_PATTERNS = {
    "rtdetr-l": (r"(?:^|[-_])rtdetr-l(?:\.|-)", "object_detection"),
    "yolov8n-seg": (r"(?:^|[-_])yolov8n-seg(?:\.|-)", "semantic_segmentation"),
    "yolov8s-seg": (r"(?:^|[-_])yolov8s-seg(?:\.|-)", "semantic_segmentation"),
    "yolov8m-seg": (r"(?:^|[-_])yolov8m-seg(?:\.|-)", "semantic_segmentation"),
    "yolov8l-seg": (r"(?:^|[-_])yolov8l-seg(?:\.|-)", "semantic_segmentation"),
    "yolov8x-seg": (r"(?:^|[-_])yolov8x-seg(?:\.|-)", "semantic_segmentation"),
    "yolov8n": (r"(?:^|[-_])yolov8n(?=(?:\.|-)(?!seg))", "object_detection"),
    "yolov8s": (r"(?:^|[-_])yolov8s(?=(?:\.|-)(?!seg))", "object_detection"),
    "yolov8m": (r"(?:^|[-_])yolov8m(?=(?:\.|-)(?!seg))", "object_detection"),
    "yolov8l": (r"(?:^|[-_])yolov8l(?=(?:\.|-)(?!seg))", "object_detection"),
    "yolov8x": (r"(?:^|[-_])yolov8x(?=(?:\.|-)(?!seg))", "object_detection"),
    "yolo26n-seg": (r"(?:^|[-_])yolo26n-seg(?:\.|-)", "semantic_segmentation"),
    "yolo26s-seg": (r"(?:^|[-_])yolo26s-seg(?:\.|-)", "semantic_segmentation"),
    "yolo26n": (r"(?:^|[-_])yolo26n(?=(?:\.|-)(?!seg))", "object_detection"),
    "yolo26s": (r"(?:^|[-_])yolo26s(?=(?:\.|-)(?!seg))", "object_detection"),
}

YOLO_PATTERNS = _WEIGHT_PATTERNS


def default_model_dirs() -> list[Path]:
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    dirs = [
        cwd,
        cwd / "models",
        cwd / "data" / "models",
        DATA_DIR / "models",
        home / "models",
        home / ".cache" / "ultralytics",
        home / ".config" / "Ultralytics",
        home / "AppData" / "Roaming" / "Ultralytics",
    ]
    env_dir = os.environ.get("THESISFORGE_MODEL_DIR", "").strip()
    if env_dir:
        dirs.insert(0, Path(env_dir).expanduser().resolve())
    return dirs


def _model_for(filename: str) -> tuple[str | None, str | None, bool]:
    lowered = Path(filename).name.lower()
    for model, (pattern, task) in _WEIGHT_PATTERNS.items():
        if re.search(pattern, lowered):
            return model, task, True
    if "-seg" in lowered or "_seg" in lowered or "mask" in lowered:
        return _local_key(filename), "semantic_segmentation", False
    return _local_key(filename), "object_detection", False


def _local_key(filename: str) -> str:
    return "local-" + hashlib.sha1(str(filename).encode("utf-8")).hexdigest()[:8]


def scan_local_models(extra_dirs: list[Path] | None = None) -> list[dict]:
    """扫描传入目录下的 .pt 权重；未传入时扫描常用目录。"""
    seen: set[Path] = set()
    out: list[dict] = []
    dirs = list(extra_dirs) if extra_dirs else default_model_dirs()
    for root in dirs:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        for p in root.rglob("*.pt"):
            if p in seen or not p.is_file():
                continue
            name = p.name.lower()
            if name in {"best.pt", "last.pt"}:
                continue
            seen.add(p)
            model, task, known = _model_for(p.name)
            if model and task:
                out.append({
                    "filename": p.name,
                    "path": str(p),
                    "model": model,
                    "task": task,
                    "available": True,
                    "known": known,
                    "local": True,
                    "weights_path": str(p),
                })
    return out


def find_model_path(model_key: str, extra_dirs: list[Path] | None = None) -> Path | None:
    """优先返回本机已存在的权重；找不到时返回 None，交由训练管线自动下载。"""
    key = str(model_key)
    for p in scan_local_models(extra_dirs):
        if p["model"] == key:
            return Path(p["path"])
    return None


def _base_model_for_task(task: str) -> str:
    if task == "semantic_segmentation":
        return "yolov8n-seg"
    return "yolov8n"


def _local_model_entry(item: dict) -> dict:
    task = item["task"]
    base = catalog.CATALOG.get(task, {}).get("models", {}).get(_base_model_for_task(task), {})
    return {
        "label": f"{item['filename']} (本机权重)",
        "desc": f"扫描到本机权重文件：{item['path']}。使用 YOLO 格式数据集训练，训练后仍会生成 best.pt。",
        "engine": "ultralytics",
        "params": copy.deepcopy(base.get("params", {})),
        "local": True,
        "weights_path": item["path"],
    }


def build_local_catalog() -> dict:
    """静态目录 + 本机扫描到的模型目录，供前端与训练流程共用。"""
    out = copy.deepcopy(catalog.CATALOG)
    for item in scan_local_models():
        out.setdefault(item["task"], {"models": {}})
        out[item["task"]].setdefault("models", {})
        out[item["task"]]["models"][item["model"]] = _local_model_entry(item)
    return out


def get_local_model_spec(task: str, model: str, extra_dirs: list[Path] | None = None) -> dict | None:
    for item in scan_local_models(extra_dirs):
        if item["task"] == task and item["model"] == model:
            entry = _local_model_entry(item)
            return {
                "task": task,
                "model": model,
                "label": entry["label"],
                "desc": entry["desc"],
                "engine": entry["engine"],
                "params": entry["params"],
                "weights_path": entry["weights_path"],
            }
    return None
