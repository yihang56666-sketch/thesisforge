"""训练作业管理器：每个实验一个独立子进程，日志/状态落盘，可取消。

安全约定：
- run_id 由服务端生成，形如 20260913-101500-a1b2c3，所有外部传入的 run_id
  必须通过 run_dir_of() 白名单 + 归一化校验，防止路径穿越；
- 动态路径一律先 resolve()、显式拒绝 ".."、并限制在实验目录内。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .config import RUNS_DIR, ROOT

JOBS: dict[str, subprocess.Popen] = {}

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_RUN_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_RUNS_ROOT = str(RUNS_DIR.resolve())
_ALLOWED_SCRIPTS = {"train_sklearn.py", "train_torch.py"}

GROUP_BASELINE = "baseline"
GROUP_IMPROVED = "improved"
GROUP_ABLATION = "ablation"
GROUP_CUSTOM = "custom"
GROUPS = (GROUP_BASELINE, GROUP_IMPROVED, GROUP_ABLATION, GROUP_CUSTOM)

GROUP_LABELS = {
    GROUP_BASELINE: "基线",
    GROUP_IMPROVED: "改进",
    GROUP_ABLATION: "消融",
    GROUP_CUSTOM: "自定义",
}


def _safe_child(base: Path, name: str) -> Path:
    """规范化 base/name：拒绝 '..' 并确保结果限制在 base 目录内。"""
    base = base.resolve()
    target = base / name
    if ".." in target.parts or not target.resolve().is_relative_to(base):
        raise ValueError("非法路径")
    return target.resolve()


def run_dir_of(run_id: str) -> Path:
    """校验 run_id 并返回其实验目录；非法 ID 直接拒绝。"""
    run_id = str(run_id or "")
    if not _RUN_ID_RE.match(run_id):
        raise ValueError("非法的实验 ID")
    p = (RUNS_DIR / run_id).resolve()
    inside = str(p).lower().startswith(_RUNS_ROOT.lower() + os.sep) if os.name == "nt" else str(p).startswith(_RUNS_ROOT + os.sep)
    if not inside:
        raise ValueError("非法的实验路径")
    return p


def _write_status(run_dir: Path, state: str, error: str | None = None) -> None:
    _safe_child(run_dir, "status.json").write_text(
        json.dumps({"state": state, "error": error, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False),
        encoding="utf-8",
    )


def read_status(run_id: str) -> dict:
    try:
        p = _safe_child(run_dir_of(run_id), "status.json")
    except ValueError:
        return {"state": STATUS_FAILED, "error": "非法实验 ID"}
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"state": STATUS_RUNNING, "error": None}


def create_run(config: dict, script: str) -> str:
    if script not in _ALLOWED_SCRIPTS:
        raise ValueError("非法的训练脚本")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = run_dir_of(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["run_id"] = run_id
    _safe_child(run_dir, "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_status(run_dir, STATUS_RUNNING)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    sync_jobs()  # 先回收已退出的作业，避免陈旧 running 状态与句柄堆积

    base = run_dir.resolve()
    log_path = base / "log.txt"
    if ".." in log_path.parts or not log_path.resolve().is_relative_to(base):
        raise ValueError("非法日志路径")
    log_fp = log_path.open("w", encoding="utf-8", buffering=1)

    script_path = (ROOT / "app" / script).resolve()
    if ".." in script_path.parts or not script_path.is_relative_to((ROOT / "app").resolve()):
        raise ValueError("非法脚本路径")
    try:
        proc = subprocess.Popen(
            worker_cmd(Path(script).name, script_path, run_dir),
            cwd=str(ROOT), env=env, stdout=log_fp, stderr=subprocess.STDOUT,
        )
    finally:
        log_fp.close()  # 子进程已持有自己的句柄，父进程不必再留着
    JOBS[run_id] = proc
    return run_id


def worker_cmd(script_name: str, script_path: Path, run_dir: Path) -> list[str]:
    """训练子进程命令行。

    普通 Python 下直接 `python app/train_x.py`；打包成 EXE（PyInstaller）后
    sys.executable 就是主程序本身，必须回到自身入口用 --tf-worker 分发，
    否则会把自己再启一次 Web 服务。
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--tf-worker", script_name, "--run-dir", str(run_dir)]
    return [sys.executable, "-u", str(script_path), "--run-dir", str(run_dir)]


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=15)
        else:
            proc.terminate()
    except Exception:
        pass


def cancel_run(run_id: str) -> bool:
    proc = JOBS.get(run_id)
    run_dir = run_dir_of(run_id)
    if proc is not None and proc.poll() is None:
        _kill_tree(proc)
        # 先等子进程真正退出，再落最终状态，避免被垂死子进程覆盖
        try:
            proc.wait(timeout=15)
        except Exception:
            pass
        current = read_status(run_id).get("state")
        if current in (STATUS_DONE, STATUS_FAILED):
            JOBS.pop(run_id, None)
            return current == STATUS_DONE
        _write_status(run_dir, STATUS_CANCELLED, "用户手动取消")
        JOBS.pop(run_id, None)
        return True
    if run_dir.exists() and read_status(run_id).get("state") == STATUS_RUNNING:
        _write_status(run_dir, STATUS_CANCELLED, "用户手动取消")
        JOBS.pop(run_id, None)
        return True
    return False


def sync_jobs() -> None:
    """把已退出但状态仍为 running 的进程标记为失败（正常完成由训练脚本自己写状态）。"""
    for run_id, proc in list(JOBS.items()):
        rc = proc.poll()
        if rc is not None:
            state = read_status(run_id).get("state")
            if state == STATUS_RUNNING:
                if rc == 0:
                    _write_status(run_dir_of(run_id), STATUS_DONE)
                else:
                    _write_status(run_dir_of(run_id), STATUS_FAILED, f"训练进程异常退出，退出码 {rc}，请查看日志")
            JOBS.pop(run_id, None)


def read_summary(run_id: str) -> dict | None:
    p = _safe_child(run_dir_of(run_id), "summary.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def read_meta(run_id: str) -> dict:
    """读取实验元数据；旧实验没有元数据时补齐默认值。"""
    try:
        cfg = json.loads(_safe_child(run_dir_of(run_id), "config.json").read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    return {
        "name": cfg.get("name") or "",
        "group": cfg.get("group") if cfg.get("group") in GROUPS else GROUP_BASELINE,
        "note": cfg.get("note") or "",
    }


def update_meta(run_id: str, update: dict) -> dict:
    """更新实验名称/分组/备注；分组非法时直接拒绝。"""
    run_dir = run_dir_of(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(run_id)
    cfg_path = _safe_child(run_dir, "config.json")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    meta = read_meta(run_id)
    for key in ("name", "group", "note"):
        if key not in update:
            continue
        value = update[key]
        if key == "group":
            value = (value or "").strip()
            if value not in GROUPS:
                raise ValueError("实验分组必须是 baseline/improved/ablation/custom 之一")
        else:
            value = "" if value is None else str(value).strip()
        if key == "name":
            value = value[:80]
        elif key == "note":
            value = value[:500]
        cfg[key] = value
        meta[key] = value
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def read_metrics_log(run_id: str) -> list[dict]:
    p = _safe_child(run_dir_of(run_id), "metrics.jsonl")
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def tail_log(run_id: str, max_bytes: int = 20000) -> str:
    p = _safe_child(run_dir_of(run_id), "log.txt")
    if not p.exists():
        return ""
    data = p.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="replace")


def list_runs() -> list[dict]:
    sync_jobs()
    out = []
    if not RUNS_DIR.exists():
        return out
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        try:
            run_dir_of(d.name)  # 跳过任何不符合命名规则的目录
        except ValueError:
            continue
        cfg = {}
        try:
            cfg = json.loads(_safe_child(d, "config.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        status = read_status(d.name)
        summary = read_summary(d.name) or {}
        meta = read_meta(d.name)
        out.append({
            "run_id": d.name,
            "state": status.get("state"),
            "error": status.get("error"),
            "name": meta["name"],
            "group": meta["group"],
            "note": meta["note"],
            "dataset_name": cfg.get("dataset_name"),
            "task": cfg.get("task"),
            "model": cfg.get("model"),
            "model_label": cfg.get("model_label"),
            "created_at": cfg.get("created_at"),
            "primary_metric": summary.get("primary_metric"),
        })
    return out


def run_detail(run_id: str) -> dict:
    sync_jobs()
    run_dir = run_dir_of(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(run_id)
    try:
        cfg = json.loads(_safe_child(run_dir, "config.json").read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    status = read_status(run_id)
    summary = read_summary(run_id)
    meta = read_meta(run_id)
    artifacts = sorted([p.name for p in run_dir.iterdir() if p.suffix == ".png"])
    return {
        "run_id": run_id,
        "state": status.get("state"),
        "error": status.get("error"),
        "name": meta["name"],
        "group": meta["group"],
        "note": meta["note"],
        "config": cfg,
        "summary": summary,
        "metrics_log": read_metrics_log(run_id),
        "log_tail": tail_log(run_id),
        "artifacts": artifacts,
    }
