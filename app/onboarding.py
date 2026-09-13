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


def _blank_step() -> dict:
    return {"state": "todo", "completed": False, "saved_at": None}


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
                data["steps"][key] = {
                    "state": step.get("state", "todo"),
                    "completed": bool(step.get("completed", False)),
                    "saved_at": step.get("saved_at"),
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
    for key, step in data.get("steps", {}).items():
        if key in STEP_NAMES and isinstance(step, dict) and step.get("state") in VALID_STATES:
            clean["steps"][key] = {"state": step["state"],
                                   "completed": bool(step.get("completed", False)),
                                   "saved_at": step.get("saved_at")}
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
    }
    save_progress(data)
    return True


def looks_completed_manually(data: dict) -> bool:
    return all(step.get("completed", False) for step in data.get("steps", {}).values())
