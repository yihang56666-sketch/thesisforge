"""实验对比：统一构建指标对比表、CSV 与 Markdown 文本。

对比口径只有一个来源（summary.json），前端只负责渲染，避免同一组实验
在列表页、详情页、报告页出现不同数值。主指标固定为对比表第一行。
"""
from __future__ import annotations

import csv
import io
import statistics
import uuid

from . import catalog, runner
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


def _mean_std(values: list[float]) -> str:
    """均值±标准差，精度按数值量级自适应，正文/CSV 共用同一字符串。"""
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    prec = 0 if abs(mean) >= 100 else (1 if abs(mean) >= 10 else (2 if abs(mean) >= 1 else 3))
    return f"{mean:.{prec}f}±{std:.{prec}f}" if len(values) > 1 else f"{mean:.{prec}f}"


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
            "batch_id": cfg.get("batch_id"),
            "batch_kind": cfg.get("batch_kind"),
            "repeat_index": cfg.get("repeat_index"),
        })
    experiments = sort_runs(experiments, by_metric=True)

    repeats: dict[str, list[dict]] = {}
    for e in experiments:
        if e.get("batch_kind") == BATCH_KIND_REPEATS and e.get("batch_id"):
            repeats.setdefault(e["batch_id"], []).append(e)
    for members in repeats.values():
        members.sort(key=lambda m: (m.get("repeat_index") is None, m.get("repeat_index") or 0, m.get("run_id") or ""))

    columns = []
    emitted: set[str] = set()
    for e in experiments:
        bid = e.get("batch_id")
        if (e.get("batch_kind") == BATCH_KIND_REPEATS and bid in repeats and len(repeats[bid]) > 1):
            if bid in emitted:
                continue
            emitted.add(bid)
            members = repeats[bid]
            columns.append({
                "run_id": members[0]["run_id"],
                "run_ids": [m["run_id"] for m in members],
                "label": (members[0]["name"] or members[0]["model_label"] or members[0]["run_id"]) + f" ×{len(members)}次",
                "group": members[0]["group"],
                "group_label": members[0]["group_label"],
                "model_label": members[0]["model_label"] or members[0]["model"] or "-",
                "repeat_count": len(members),
            })
            continue
        columns.append({
            "run_id": e["run_id"],
            "label": e["name"] or e["model_label"] or e["run_id"],
            "group": e["group"],
            "group_label": e["group_label"],
            "model_label": e["model_label"] or e["model"] or "-",
            "repeat_count": 1,
        })

    metric_rows = []
    if experiments:
        first = experiments[0]
        first_pm = first["primary_metric"] or {}
        pm_key = first_pm.get("name")
        keys_order = [pm_key] if pm_key else []
        for e in experiments:
            for k in (e.get("metrics") or {}):
                if k != pm_key and k not in keys_order:
                    keys_order.append(str(k))
        for idx, k in enumerate(keys_order):
            is_primary = idx == 0 and bool(pm_key)
            row_values = []
            n_per_col = []
            std_only = []
            for c in columns:
                rids = c.get("run_ids") or [c.get("run_id")]
                vals = []
                for e in experiments:
                    if e["run_id"] not in rids:
                        continue
                    v = (e.get("primary_metric") or {}).get("value") if is_primary else (e.get("metrics") or {}).get(k)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.append(float(v))
                n_per_col.append(len(vals))
                if not vals:
                    row_values.append(None)
                    std_only.append(None)
                elif len(vals) > 1:
                    row_values.append(_mean_std(vals))
                    std_only.append(statistics.pstdev(vals))
                else:
                    row_values.append(vals[0])
                    std_only.append(None)
            metric_rows.append({
                "key": "primary_metric" if is_primary else k,
                "label": (f"主指标：{pm_key}" if is_primary else _metric_label(k)),
                "is_primary": is_primary,
                "higher_is_better": _higher_is_better(str(pm_key if is_primary else k)),
                "values": row_values,
                "n": n_per_col,
                "std": std_only,
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


BATCH_KIND_REPEATS = "repeats"
BATCH_KIND_ABLATION = "ablation"
_BATCH_BASE_SEED = 42
_BATCH_MAX_SEED = 2**31 - 1


def _read_source_config(run_id: str) -> dict:
    """读取已完成实验的完整配置；未完成/不存在时报错。"""
    d = runner.run_detail(run_id)
    if d["state"] != runner.STATUS_DONE:
        raise ValueError("请先等待源实验训练完成")
    cfg = d["config"]
    if not cfg or not cfg.get("task") or not cfg.get("model") or not cfg.get("dataset_id"):
        raise ValueError("源实验配置不完整，无法批量派生")
    return dict(cfg)


def _coerce_param(spec: dict, key: str, value):
    """消融参数也走目录白名单与类型约束，和新建实验同一套口径。"""
    pschema = (spec.get("params") or {}).get(key)
    if not pschema:
        raise ValueError(f"参数 {key} 不在模型目录中，无法消融")
    clean = catalog.sanitize_params({"params": {key: pschema}}, {key: value}).get(key)
    if clean is None:
        raise ValueError(f"参数 {key} 的值不合法")
    return clean


def _next_derived_seed(params: dict, batch_id: str, offset: int) -> int:
    """派生实验保持独立随机性：在批内用 batch_id 派生确定性偏移。"""
    base = params.get("seed")
    if base is None:
        base = _BATCH_BASE_SEED
    try:
        base = max(0, min(int(base), _BATCH_MAX_SEED))
    except (TypeError, ValueError):
        base = _BATCH_BASE_SEED
    extra = (int(batch_id, 16) + offset) % 1000
    return min(base + extra, _BATCH_MAX_SEED)


def _next_repeat_seed(seed, offset: int) -> int:
    """重复实验只在随机种子上加偏移，保证唯一且仍在 int 范围内。"""
    try:
        base = max(0, min(int(seed), _BATCH_MAX_SEED))
    except (TypeError, ValueError):
        base = _BATCH_BASE_SEED
    return min(base + offset, _BATCH_MAX_SEED)


def _clip_name(name: str, limit: int = 80) -> str:
    return (name or "").strip()[:limit]


def _run_kind(engine: str) -> str:
    return "train_torch.py" if engine == "torch" else "train_sklearn.py"


def create_batch_repeats(source_run_id: str, count: int) -> list[str]:
    """克隆一个已完成实验，按同一配置跑 count 次，仅随机种子不同。"""
    count = int(count)
    if count < 2 or count > 10:
        raise ValueError("重复次数需在 2-10 之间")
    base = _read_source_config(source_run_id)
    spec = catalog.get_model_spec(base.get("task", ""), base.get("model", ""))
    engine = (spec or {}).get("engine") or "sklearn"
    base_name = _clip_name(base.get("name") or base.get("model_label") or "重复实验", 62)
    seed = base.get("random_state") or (base.get("params") or {}).get("seed") or _BATCH_BASE_SEED
    batch_id = uuid.uuid4().hex
    run_ids = []
    for i in range(1, count + 1):
        cfg = dict(base)
        cfg.pop("batch_id", None)
        cfg["name"] = _clip_name(f"{base_name}-重复-{i}")
        cfg["group"] = runner.GROUP_IMPROVED
        cfg["note"] = _clip_name((base.get("note") or "") + "；重复实验，仅随机种子不同", 500)
        cfg["batch_id"] = batch_id
        cfg["batch_kind"] = BATCH_KIND_REPEATS
        cfg["repeat_index"] = i
        cfg["random_state"] = _next_repeat_seed(seed, i - 1)
        params = dict(base.get("params") or {})
        if "seed" in params:
            params["seed"] = _next_repeat_seed(params.get("seed"), i - 1)
        cfg["params"] = params
        run_ids.append(runner.create_run(cfg, _run_kind(engine)))
    return run_ids


def _fmt_ablation_value(schema, value) -> str:
    if schema and schema.get("type") == "bool":
        return "开启" if value else "关闭"
    if isinstance(value, float) and value == int(value):
        return f"{int(value)}"
    return str(value)


def create_batch_ablation(source_run_id: str, overrides: dict[str, list]) -> list[str]:
    """基于已完成实验，对每个 (参数, 值) 各派生一个只改该参数的新实验。"""
    if not overrides:
        raise ValueError("请至少选择一个消融参数")
    base = _read_source_config(source_run_id)
    spec = catalog.get_model_spec(base.get("task", ""), base.get("model", ""))
    engine = (spec or {}).get("engine") or "sklearn"
    schema = (spec or {}).get("params") or {}
    base_name = _clip_name(base.get("name") or base.get("model_label") or "消融实验", 50)
    batch_id = uuid.uuid4().hex
    run_ids = []
    for param, values in overrides.items():
        clean_values = []
        for value in values:
            clean = _coerce_param(spec, param, value)
            if clean not in clean_values:
                clean_values.append(clean)
        if not clean_values:
            raise ValueError(f"参数 {param} 没有可用的消融值")
        for position, clean in enumerate(clean_values):
            cfg = dict(base)
            cfg.pop("batch_id", None)
            label = (schema.get(param) or {}).get("label") or param
            pretty = _fmt_ablation_value(schema.get(param), clean)
            cfg["name"] = _clip_name(f"{base_name}-消融-{label}-{pretty}")
            cfg["group"] = runner.GROUP_ABLATION
            cfg["note"] = _clip_name(
                (base.get("note") or "") + f"；消融：仅修改 {label}={clean}，其余与源实验一致", 500)
            cfg["batch_id"] = batch_id
            cfg["batch_kind"] = BATCH_KIND_ABLATION
            cfg.pop("repeat_index", None)
            params = dict(base.get("params") or {})
            params[param] = clean
            if "seed" in schema:
                params["seed"] = _next_derived_seed(params, batch_id, position * 100 + len(run_ids))
            cfg["params"] = params
            run_ids.append(runner.create_run(cfg, _run_kind(engine)))
    return run_ids
