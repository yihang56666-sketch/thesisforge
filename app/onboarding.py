"""毕设向导进度：步骤定义、持久化与完成状态。"""
from __future__ import annotations

import json
import time

from .config import DATA_DIR

PROJECT_FILE = DATA_DIR / "project.json"
STEP_NAMES = {
    "1": "立项",
    "2": "数据",
    "3": "数据预处理与划分",
    "4": "任务与网络架构",
    "5": "优化与训练策略",
    "6": "训练与监控",
    "7": "评估分析",
    "8": "消融实验",
    "9": "实验对比",
    "10": "报告工坊",
}
VALID_STATES = {"todo", "doing", "done", "skipped"}

STEP_TEMPLATES = {
    "3": "data_prep",
    "4": "network",
    "5": "training",
}

TEMPLATE_KEYS = {
    "data_prep": {"missing", "impute", "scale", "encode", "augment", "split_first",
                  "test_size", "val_split", "seed"},
    "network": {"task", "arch", "params"},
    "training": {"optimizer", "lr", "batch_size", "epochs", "scheduler",
                 "weight_decay", "early_stop_patience", "grad_clip", "device"},
}


def _blank_step() -> dict:
    return {"state": "todo", "completed": False, "saved_at": None, "template": None}


def default_template(template: str | None) -> dict | None:
    """向导步骤配置模板：只在步骤 3/4/5 使用，旧项目自动补默认推荐值。"""
    if template == "data_prep":
        return {
            "missing": "impute", "impute": "median", "scale": "standard",
            "encode": "onehot", "augment": "flip_rotate", "split_first": True,
            "test_size": 0.2, "val_split": 0.2, "seed": 42,
        }
    if template == "network":
        return {
            "task": "tabular_classification",
            "arch": "mlp",
            "params": {},
        }
    if template == "training":
        return {
            "optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 15,
            "scheduler": "cosine", "weight_decay": 0.0,
            "early_stop_patience": 0, "grad_clip": 0.0, "device": "auto",
        }
    return None


def _clean_template(template: str | None, value) -> dict | None:
    keys = TEMPLATE_KEYS.get(template or "")
    if not keys:
        return None
    base = default_template(template) or {}
    if isinstance(value, dict):
        base.update({k: value[k] for k in keys if k in value})
    return base


def _step_template_name(key: str) -> str | None:
    return STEP_TEMPLATES.get(str(key))


def default_progress() -> dict:
    return {
        "version": 1,
        "project": {"title": "", "direction": "", "author": "", "advisor": "", "goal": ""},
        "steps": {key: _blank_step() for key in STEP_NAMES},
        "active_step": 1,
        "updated_at": "",
    }


def load_progress() -> dict:
    try:
        raw = json.loads(PROJECT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    data = default_progress()
    if isinstance(raw, dict) and raw.get("version") == 1:
        for key in STEP_NAMES:
            step = raw.get("steps", {}).get(key, {})
            if isinstance(step, dict) and step.get("state") in VALID_STATES:
                template_name = _step_template_name(key) or step.get("template")
                data["steps"][key] = {
                    "state": step.get("state", "todo"),
                    "completed": bool(step.get("completed", False)),
                    "saved_at": step.get("saved_at"),
                    "template": _clean_template(template_name, step.get("template")),
                }
        if isinstance(raw.get("project"), dict):
            data["project"].update({k: str(v) for k, v in raw["project"].items()
                                    if k in data["project"]})
        if isinstance(raw.get("active_step"), int) and 1 <= raw["active_step"] <= 10:
            data["active_step"] = raw["active_step"]
    return data


def save_progress(data: dict) -> dict:
    clean = default_progress()
    clean["project"].update({k: str(v) for k, v in data.get("project", {}).items()
                             if k in clean["project"]})
    existing = load_progress()
    for key, step in data.get("steps", {}).items():
        if key in STEP_NAMES and isinstance(step, dict) and step.get("state") in VALID_STATES:
            template_name = _step_template_name(key) or step.get("template")
            template_value = step.get("template")
            if template_name in TEMPLATE_KEYS and template_value is None:
                template_value = existing["steps"][key].get("template")
            clean["steps"][key] = {"state": step["state"],
                                   "completed": bool(step.get("completed", False)),
                                   "saved_at": step.get("saved_at"),
                                   "template": _clean_template(template_name, template_value)}
    if isinstance(data.get("active_step"), int) and 1 <= data["active_step"] <= 10:
        clean["active_step"] = data["active_step"]
    clean["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    PROJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean


def update_step(step_key: str | int, state: str) -> bool:
    data = load_progress()
    key = str(step_key)
    if key not in STEP_NAMES or state not in VALID_STATES:
        return False
    data["steps"][key] = {
        "state": state,
        "completed": state == "done",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S") if state in ("done", "skipped") else None,
        "template": data["steps"][key].get("template") if key in ("3", "4", "5") else None,
    }
    save_progress(data)
    return True


def looks_completed_manually(data: dict) -> bool:
    return all(step.get("completed", False) for step in data.get("steps", {}).values())
