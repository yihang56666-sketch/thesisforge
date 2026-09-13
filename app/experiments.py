"""实验对比：统一构建指标对比表、CSV 与 Markdown 文本。

对比口径只有一个来源（summary.json），前端只负责渲染，避免同一组实验
在列表页、详情页、报告页出现不同数值。主指标固定为对比表第一行。
"""
from __future__ import annotations

import csv
import io

from . import runner
from .runner import GROUP_ABLATION, GROUP_BASELINE, GROUP_CUSTOM, GROUP_IMPROVED, GROUP_LABELS

GROUP_ORDER = (GROUP_BASELINE, GROUP_IMPROVED, GROUP_ABLATION, GROUP_CUSTOM)

_METRIC_LABELS = {
    "accuracy": "准确率",
    "balanced_accuracy": "平衡准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1",
    "auc": "AUC",
    "roc_auc": "ROC AUC",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R²",
    "loss": "损失",
    "val_loss": "验证损失",
    "test_loss": "测试损失",
}

_HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "auc": True,
    "roc_auc": True,
    "r2": True,
    "mae": False,
    "mse": False,
    "rmse": False,
    "loss": False,
    "val_loss": False,
    "test_loss": False,
}


def group_rank(group: str) -> int:
    return GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)


def group_label(group: str) -> str:
    return GROUP_LABELS.get(group, group or GROUP_LABELS[GROUP_BASELINE])


def sort_runs(runs, by_metric: bool = False) -> list:
    """按 基线→改进→消融→自定义 排序；by_metric 时组内按主指标降序。"""
    def metric_value(r):
        pm = (r.get("summary") or r).get("primary_metric") or {}
        v = pm.get("value")
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def key(r):
        cfg = r.get("config") or r
        group = cfg.get("group") or r.get("group") or GROUP_BASELINE
        m = metric_value(r)
        metric_key = (-m) if (m is not None and by_metric) else 0
        name = cfg.get("name") or r.get("name") or ""
        return (group_rank(group), metric_key, name, r.get("run_id") or "")

    return sorted(runs, key=key)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.8g}"
    return str(v)


def _metric_label(key: str) -> str:
    return _METRIC_LABELS.get(key, key)


def _higher_is_better(key: str) -> bool:
    return _HIGHER_IS_BETTER.get(key, True)


def _numeric_metrics(summary: dict) -> dict:
    out = {}
    for k, v in (summary.get("metrics") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
    return out


def build_comparison(run_ids: list[str]) -> dict:
    """只纳入已完成且带 summary 的实验；返回按分组排序的对比数据结构。"""
    experiments = []
    for rid in (run_ids or [])[:20]:
        try:
            d = runner.run_detail(str(rid))
        except (ValueError, FileNotFoundError):
            continue
        if d["state"] != "done" or not d["summary"]:
            continue
        cfg = d["config"]
        s = d["summary"]
        pm = s.get("primary_metric") or {}
        experiments.append({
            "run_id": rid,
            "name": cfg.get("name") or "",
            "group": cfg.get("group") if cfg.get("group") in runner.GROUPS else runner.GROUP_BASELINE,
            "group_label": group_label(cfg.get("group") or runner.GROUP_BASELINE),
            "dataset_name": cfg.get("dataset_name"),
            "model": cfg.get("model"),
            "model_label": cfg.get("model_label"),
            "params": cfg.get("params"),
            "created_at": cfg.get("created_at"),
            "primary_metric": pm,
            "metrics": _numeric_metrics(s),
            "split_scheme": s.get("split_scheme"),
            "n_train": s.get("n_train"),
            "n_val": s.get("n_val"),
            "n_test": s.get("n_test"),
        })
    experiments = sort_runs(experiments, by_metric=True)

    columns = [{
        "run_id": e["run_id"],
        "label": e["name"] or e["model_label"] or e["run_id"],
        "group": e["group"],
        "group_label": e["group_label"],
        "model_label": e["model_label"] or e["model"] or "-",
    } for e in experiments]

    metric_rows = []
    if experiments:
        first = experiments[0]
        first_pm = first["primary_metric"] or {}
        metric_rows.append({
            "key": "primary_metric",
            "label": f"主指标：{first_pm.get('name') or '主指标'}",
            "is_primary": True,
            "higher_is_better": _higher_is_better(str(first_pm.get("name") or "")),
            "values": [(e.get("primary_metric") or {}).get("value") for e in experiments],
        })
        pm_key = (first_pm or {}).get("name")
        extra_keys = []
        for e in experiments:
            for k in (e.get("metrics") or {}):
                if k != pm_key and k not in extra_keys:
                    extra_keys.append(str(k))
        for k in extra_keys:
            metric_rows.append({
                "key": k,
                "label": _metric_label(k),
                "is_primary": False,
                "higher_is_better": _higher_is_better(k),
                "values": [(e.get("metrics") or {}).get(k) for e in experiments],
            })

    # CSV：指标 × 实验，第一行是主指标
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["指标"] + [c["label"] for c in columns])
    for mr in metric_rows:
        w.writerow([mr["label"]] + [_fmt(v) for v in mr["values"]])
    csv_text = buf.getvalue()

    # Markdown：与 CSV 同构，方便直接粘进论文/笔记
    if columns:
        header = "| 指标 | " + " | ".join(c["label"] for c in columns) + " |"
        sep = "| --- |" + " --- |" * len(columns)
        lines = [header, sep]
        for mr in metric_rows:
            cells = [mr["label"]] + [_fmt(v) for v in mr["values"]]
            lines.append("| " + " | ".join(cells) + " |")
        md_text = "\n".join(lines) + "\n"
    else:
        md_text = ""

    return {
        "experiments": experiments,
        "columns": columns,
        "metric_rows": metric_rows,
        "csv": csv_text,
        "markdown": md_text,
        "count": len(experiments),
    }
