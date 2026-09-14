"""Optuna 自动调参：复用现有训练子进程，避免与普通实验产生不同评估口径。

当前版本支持：
- TPE / 网格 / Hyperband 三种搜索模式；
- 从模型目录自动构建搜索空间；
- 每个候选参数都真实跑一次训练，并读取 summary.json；
- 结果落盘到 data/tuning/<search_id>/，便于后续报告与复现。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import catalog


SEARCH_MODES = ("tpe", "grid", "hyperband")


def build_search_space(task: str, model: str, options: dict | None = None) -> dict:
    """从模型目录构建安全的搜索空间；只包含目录中定义的参数。"""
    spec = catalog.get_model_spec(task, model)
    if not spec:
        raise ValueError("未知任务或模型")

    opts = dict(options or {})
    n_trials = int(opts.get("n_trials") or 10)
    n_trials = max(2, min(n_trials, 200))
    metric = str(opts.get("metric") or "primary_metric").strip()
    mode = str(opts.get("search_mode") or "tpe").strip().lower()
    if mode not in SEARCH_MODES:
        raise ValueError(f"未知搜索模式: {mode}（可选 {list(SEARCH_MODES)}）")

    space: dict = {}
    for key, p in (spec.get("params") or {}).items():
        # 训练参数里 epochs/device 一类不适合完全交给搜索；这里仍然保留，但给前端提示。
        if p.get("type") in ("float", "int", "choice", "bool", "string"):
            space[key] = dict(p)
    if not space:
        raise ValueError("该模型没有可调参数")
    return {
        "task": task,
        "model": model,
        "model_label": spec["label"],
        "n_trials": n_trials,
        "metric": metric,
        "search_mode": mode,
        "space": space,
    }


def suggest_params(space: dict, trial, base_params: dict | None = None) -> dict:
    """把 Optuna trial 映射成目录允许的参数。"""
    base = dict(base_params or {})
    for key, p in (space.get("space") or {}).items():
        ptype = p.get("type")
        if ptype == "float":
            base[key] = trial.suggest_float(key, float(p.get("min", 0.0)), float(p.get("max", 1.0)))
        elif ptype == "int":
            base[key] = int(trial.suggest_int(key, int(p.get("min", 1)), int(p.get("max", 10))))
        elif ptype == "choice":
            base[key] = trial.suggest_categorical(key, list(p.get("options") or []))
        elif ptype == "bool":
            base[key] = trial.suggest_categorical(key, [True, False])
        elif ptype == "string":
            choices = [str(x) for x in _string_choices(p)]
            base[key] = trial.suggest_categorical(key, choices)
    return base


def _string_choices(p: dict) -> list:
    """字符串参数给搜索时用一组保守推荐值；避免任意字符串带来的风险。"""
    default = str(p.get("default") or "")
    if not default:
        return [default]
    values = [default]
    # 逗号分隔的层/通道参数，搜索时做轻量加倍/减半，保持结构合法。
    parts = [x.strip() for x in default.split(",") if x.strip()]
    if len(parts) > 1:
        try:
            nums = [int(x) for x in parts]
            values.append(",".join(str(max(1, int(v / 2))) for v in nums))
            values.append(",".join(str(min(2048, int(v * 2))) for v in nums))
        except ValueError:
            pass
    return sorted(set(values))


def objective_value(summary: dict, metric: str):
    """优先读取指定指标，否则用 primary_metric；没有值时剪枝。"""
    metrics = summary.get("metrics") or {}
    if metric and metric != "primary_metric" and metric in metrics:
        return metrics[metric]
    pm = summary.get("primary_metric") or {}
    if metric and metric != "primary_metric":
        return pm.get(metric)
    return pm.get("value")


def _safe_search_dir(search_id: str) -> Path:
    from .config import DATA_DIR

    base = (DATA_DIR / "tuning").resolve()
    target = (base / search_id).resolve()
    if ".." in target.parts or not target.is_relative_to(base):
        raise ValueError("非法调参目录")
    return target


def _run_trial(config: dict, out_dir: Path, script: str, timeout_seconds: int = 3600):
    """启动现有训练脚本，等待 summary.json 生成后返回结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    script_path = (Path(__file__).resolve().parent / script).resolve()
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + __import__("os").pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    with (out_dir / "log.txt").open("w", encoding="utf-8", buffering=1) as fp:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script_path), "--run-dir", str(out_dir)],
            cwd=str(Path(__file__).resolve().parent.parent), env=env,
            stdout=fp, stderr=subprocess.STDOUT,
        )
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            raise RuntimeError(f"调参 trial 超时（{timeout_seconds}s）")
    if proc.returncode != 0:
        raise RuntimeError(f"训练失败，退出码 {proc.returncode}，请查看 {out_dir / 'log.txt'}")
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"训练完成但没有 summary.json: {out_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_search(
    task: str,
    model: str,
    base_config: dict,
    n_trials: int = 10,
    metric: str = "primary_metric",
    search_mode: str = "tpe",
    seed: int = 42,
):
    """运行一次本地 Optuna 搜索。"""
    import optuna

    from .runner import JOBS  # noqa: F401  # 保持兼容；当前直接使用子进程。

    space = build_search_space(task, model, {
        "n_trials": n_trials, "metric": metric, "search_mode": search_mode,
    })
    search_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    root = _safe_search_dir(search_id)
    root.mkdir(parents=True, exist_ok=True)

    pruner = None
    if search_mode == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(2, space["n_trials"]),
            reduction_factor=2,
        )
    sampler = optuna.samplers.TPESampler(seed=seed)
    if search_mode == "grid":
        # Optuna 的 GridSampler 需要完整网格；这里改为轻量候选网格，避免爆炸。
        candidates = {k: list(_string_choices(v)) if v.get("type") == "string"
                      else list(v.get("options") or []) for k, v in space["space"].items()}
        # 只取每个数值参数的默认/上下界三个候选，控制搜索规模。
        for k, v in space["space"].items():
            if v.get("type") in ("float", "int"):
                lo, hi = float(v.get("min", 0)), float(v.get("max", 1))
                candidates[k] = [lo, (lo + hi) / 2, hi]
        combos = 1
        for values in candidates.values():
            if values:
                combos *= len(values)
        space["n_trials"] = min(space["n_trials"], combos)
        sampler = optuna.samplers.GridSampler(candidates)

    def objective(trial):
        cfg = dict(base_config)
        cfg["task"] = task
        cfg["model"] = model
        cfg["model_label"] = space["model_label"]
        cfg["params"] = suggest_params(space, trial, cfg.get("params"))
        trial_dir = root / f"trial-{trial.number:03d}"
        script = "train_torch.py" if (catalog.get_model_spec(task, model) or {}).get("engine") == "torch" else "train_sklearn.py"
        summary = _run_trial(cfg, trial_dir, script)
        value = objective_value(summary, metric)
        if value is None:
            raise optuna.TrialPruned("训练没有返回可用指标")
        value = float(value)
        trial.set_user_attr("summary", {k: summary.get(k) for k in (
            "primary_metric", "metrics", "n_train", "n_val", "n_test", "elapsed"
        )})
        trial.report(value, step=0)
        return value

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=space["n_trials"], show_progress_bar=False)
    if not study.trials:
        raise RuntimeError("自动调参没有产生任何有效 trial，请检查数据集或模型配置")
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = float(best_trial.value)
    out = {
        "ok": True,
        "search_id": search_id,
        "task": task,
        "model": model,
        "model_label": space["model_label"],
        "metric": metric,
        "search_mode": search_mode,
        "n_trials": space["n_trials"],
        "best_value": best_value,
        "best_params": best_params,
        "best_trial": best_trial.number,
        "history": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
        "output_dir": str(root),
    }
    (root / "best.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
