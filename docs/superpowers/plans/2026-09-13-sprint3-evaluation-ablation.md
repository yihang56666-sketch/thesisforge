# Sprint 3: 重复实验、一键消融与实验对比增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户可以从一个已完成实验一键发起重复实验（同配置换随机种子）与消融实验（每次只改一个超参数），并在实验对比页自动合并重复批次、展示「均值±标准差」。

**Architecture:** `app/experiments.py` 新增 `create_batch_repeats` / `create_batch_ablation` 两个纯后端函数，直接克隆已完成 run 的 config 并调用 `runner.create_run`；`runner.read_meta` 增加 `batch_id` / `batch_kind` 字段；`build_comparison` 按 `batch_kind == "repeats"` && 相同 `batch_id` 聚合列；FastAPI 新增两个薄路由；前端实验详情页增加「重复 3 次 / 一键消融」入口，对比页渲染 `mean±std` 与重复次数角标。

**Tech Stack:** Python 3.13 / FastAPI / pytest / statistics / 原生 JS。

---

## File Structure

- Modify: `app/runner.py` — `read_meta` 返回 `batch_id`/`batch_kind`
- Modify: `app/experiments.py` — 批量创建 + 聚合对比
- Modify: `app/main.py` — 两个新 API 路由
- Modify: `web/app.js` — 详情页按钮、对比页聚合展示
- Test: `tests/test_experiments_batch.py` — 批量/消融/聚合单测
- Test: `tests/test_main.py` — 两个新 API 冒烟

---

### Task 1: 批处理/消融后端核心函数

**Files:**
- Modify: `app/experiments.py`
- Test: `tests/test_experiments_batch.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_experiments_batch.py`:

```python
# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import json
import time
import unittest
import uuid

from app import experiments, runner


def _run_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _make_done_run(rid, params=None, seed=42, model="mlp", model_label="MLP",
                   group="baseline", name="基线-MLP"):
    d = runner.run_dir_of(rid)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dataset_id": "iris", "dataset_name": "鸢尾花", "task": "tabular_classification",
        "model": model, "model_label": model_label,
        "params": params or {"optimizer": "adam", "lr": 0.001, "seed": seed},
        "name": name, "group": group, "note": "来源", "target": "target",
        "test_size": 0.2, "val_split": 0.2,
        "random_state": seed, "created_at": "2026-09-13 10:00:00",
    }
    (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({
        "metrics": {"accuracy": 0.91, "f1": 0.89},
        "primary_metric": {"name": "accuracy", "value": 0.91},
    }), encoding="utf-8")
    return cfg


class BatchRepeatsTest(unittest.TestCase):
    def test_repeats_share_batch_and_vary_seed_only(self):
        src = _run_id()
        _make_done_run(src)
        ids = experiments.create_batch_repeats(src, count=3)
        self.assertEqual(len(ids), 3)
        cfg0 = json.loads((runner.run_dir_of(ids[0]) / "config.json").read_text(encoding="utf-8"))
        cfg2 = json.loads((runner.run_dir_of(ids[2]) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg0["batch_id"], cfg2["batch_id"])
        self.assertEqual(cfg0["batch_kind"], "repeats")
        self.assertEqual(cfg0["repeat_index"], 1)
        self.assertEqual(cfg2["repeat_index"], 3)
        self.assertEqual(cfg0["params"]["seed"], 42)
        self.assertEqual(cfg2["params"]["seed"], 44)
        self.assertEqual(cfg0["random_state"], 42)
        self.assertEqual(cfg2["random_state"], 44)
        self.assertIn("重复-1", cfg0["name"])
        self.assertEqual(cfg0["group"], "improved")
        self.assertEqual(len(set(ids)), 3)

    def test_repeats_reject_bad_source_and_count(self):
        with self.assertRaises(ValueError):
            experiments.create_batch_repeats("bad-id", 3)
        src = _run_id()
        _make_done_run(src)
        with self.assertRaises(ValueError):
            experiments.create_batch_repeats(src, count=1)


class BatchAblationTest(unittest.TestCase):
    def test_each_ablation_changes_exactly_one_param(self):
        src = _run_id()
        _make_done_run(src, params={"n_estimators": 200, "max_depth": 0, "seed": 7})
        ids = experiments.create_batch_ablation(src, {"n_estimators": [100, 400], "max_depth": [8]})
        self.assertEqual(len(ids), 3)
        cfg0 = json.loads((runner.run_dir_of(ids[0]) / "config.json").read_text(encoding="utf-8"))
        cfg2 = json.loads((runner.run_dir_of(ids[2]) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg0["batch_id"], cfg2["batch_id"])
        self.assertEqual(cfg0["batch_kind"], "ablation")
        self.assertEqual(cfg0["params"]["n_estimators"], 100)
        self.assertEqual(cfg0["params"]["max_depth"], 0)
        self.assertEqual(cfg2["params"]["n_estimators"], 200)
        self.assertEqual(cfg2["params"]["max_depth"], 8)
        self.assertEqual(cfg0["group"], "ablation")
        self.assertIn("n_estimators", cfg0["name"])
        self.assertIn("100", cfg0["name"])

    def test_ablation_rejects_empty_overrides(self):
        src = _run_id()
        _make_done_run(src)
        with self.assertRaises(ValueError):
            experiments.create_batch_ablation(src, {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest tests/test_experiments_batch.py -q`

Expected: FAIL（`AttributeError: module 'app.experiments' has no attribute 'create_batch_repeats'`）

- [ ] **Step 3: 在 `app/experiments.py` 末尾新增实现**

```python
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


def _next_repeat_seed(seed: object, offset: int) -> int:
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
        cfg["random_state"] = _next_repeat_seed(seed, i)
        params = dict(base.get("params") or {})
        if "seed" in params:
            params["seed"] = _next_repeat_seed(params.get("seed"), i)
        cfg["params"] = params
        run_ids.append(runner.create_run(cfg, _run_kind(base.get("engine", "sklearn"))))
    return run_ids


def _fmt_ablation_value(schema: dict | None, value) -> str:
    if schema and schema.get("type") == "bool":
        return "开启" if value else "关闭"
    if isinstance(value, float):
        return f"{value:.12g}".rstrip(".") if value == int(value) else f"{value:.12g}"
    return str(value)


def create_batch_ablation(source_run_id: str, overrides: dict[str, list]) -> list[str]:
    """基于已完成实验，对每个 (参数, 值) 各派生一个只改该参数的新实验。"""
    if not overrides:
        raise ValueError("请至少选择一个消融参数")
    base = _read_source_config(source_run_id)
    base_name = _clip_name(base.get("name") or base.get("model_label") or "消融实验", 50)
    spec = catalog.get_model_spec(base.get("task", ""), base.get("model", ""))
    schema = (spec or {}).get("params") or {}
    batch_id = uuid.uuid4().hex
    run_ids = []
    for param, values in overrides.items():
        for value in values:
            cfg = dict(base)
            cfg.pop("batch_id", None)
            label = (schema.get(param) or {}).get("label") or param
            pretty = _fmt_ablation_value(schema.get(param), value)
            cfg["name"] = _clip_name(f"{base_name}-消融-{label}-{pretty}")
            cfg["group"] = runner.GROUP_ABLATION
            cfg["note"] = _clip_name(
                (base.get("note") or "") + f"；消融：仅修改 {label}={value}，其余与源实验一致", 500)
            cfg["batch_id"] = batch_id
            cfg["batch_kind"] = BATCH_KIND_ABLATION
            params = dict(base.get("params") or {})
            params[param] = value
            cfg["params"] = params
            run_ids.append(runner.create_run(cfg, _run_kind(base.get("engine", "sklearn"))))
    return run_ids
```

Also update imports at the top of `app/experiments.py`:

```python
import uuid

from . import catalog, runner
```

- [ ] **Step 4: 运行确认通过**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest tests/test_experiments_batch.py -q`

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_experiments_batch.py app/experiments.py
git commit -m "feat: 支持重复实验与消融实验批量创建"
```

---

### Task 2: 对比页重复批次聚合（mean±std）

**Files:**
- Modify: `app/experiments.py`
- Test: `tests/test_experiments.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_experiments.py`:

```python
class RepeatAggregationTest(unittest.TestCase):
    def test_repeats_same_batch_merge_and_show_mean_std(self):
        batch = "repbatch-0001"
        for i, acc in enumerate([0.91, 0.93, 0.95], start=1):
            make_run(f"20260913-101500-0{i:06x}", name=f"改进-MLP-重复-{i}", group="improved",
                     state="done", model="mlp", model_label="MLP",
                     metrics={"accuracy": acc, "f1": round(acc - 0.01, 2)},
                     primary={"name": "accuracy", "value": acc})
            cfg_path = runner.run_dir_of(f"20260913-101500-0{i:06x}") / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["batch_id"] = batch
            cfg["batch_kind"] = "repeats"
            cfg["repeat_index"] = i
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        r = experiments.build_comparison([f"20260913-101500-0{i:06x}" for i in range(1, 4)])
        self.assertEqual(r["count"], 3)
        self.assertEqual(len(r["columns"]), 1)
        self.assertEqual(r["columns"][0]["label"], "改进-MLP-重复-1 ×3次")
        self.assertEqual(r["columns"][0]["run_ids"], [f"20260913-101500-0{i:06x}" for i in range(1, 4)])
        row = r["metric_rows"][0]
        self.assertTrue(row["is_primary"])
        self.assertEqual(row["values"], ["0.93±0.016"])
        self.assertEqual(row["n"], [3])
        self.assertEqual(row["std_key"], "accuracy")
```

- [ ] **Step 2: 运行确认失败**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest tests/test_experiments.py::RepeatAggregationTest -q`

Expected: FAIL（`KeyError: 'run_ids'`）

- [ ] **Step 3: 实现重复批次聚合**

In `app/experiments.py`, add to imports:

```python
import statistics
```

Add helpers before `build_comparison`:

```python
def _mean_std(values: list[float]) -> str:
    """均值±标准差，精度按数值量级自适应，正文/CSV 共用同一字符串。"""
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    prec = 0 if abs(mean) >= 100 else (1 if abs(mean) >= 10 else (2 if abs(mean) >= 1 else 4))
    return f"{mean:.{prec}f}±{std:.{prec}f}" if len(values) > 1 else f"{mean:.{prec}f}"
```

In `build_comparison`, change the experiment append block to include batch fields:

```python
        experiments.append({
            "run_id": rid,
            "name": cfg.get("name") or "",
            "group": ...,
            "group_label": ...,
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
```

Replace `experiments = sort_runs(...)` and the `columns = [...]` list comprehension with:

```python
    experiments = sort_runs(experiments, by_metric=True)
    repeats: dict[str, list[dict]] = {}
    for e in experiments:
        if e.get("batch_kind") == "repeats" and e.get("batch_id"):
            repeats.setdefault(e["batch_id"], []).append(e)

    columns = []
    emitted: set[str] = set()
    for e in experiments:
        bid = e.get("batch_id")
        if (e.get("batch_kind") == "repeats" and bid and bid in repeats and len(repeats[bid]) > 1):
            if bid in emitted:
                continue
            emitted.add(bid)
            members = repeats[bid]
            columns.append({
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
```

Replace the `metric_rows` builder so repeat columns aggregate (keep all existing `metric_rows.append` shape but with `values`/`n`/`std_key` and member-aware values). Exact replacement from `metric_rows = []` through `return {`:

```python
    metric_rows = []
    if experiments:
        first = experiments[0]
        members_of = {c["run_ids"][0] if c.get("run_ids") else c.get("run_id"): c.get("run_ids") or [c.get("run_id")]
                      for c in columns}
        first_pm = first["primary_metric"] or {}
        pm_key = first_pm.get("name")
        keys_order = [pm_key] if pm_key else []
        for e in experiments:
            for k in (e.get("metrics") or {}):
                if k != pm_key and k not in keys_order:
                    keys_order.append(str(k))
        for idx, k in enumerate(keys_order):
            row_values = []
            n_per_col = []
            std_only = []
            for c in columns:
                rids = c.get("run_ids") or [c.get("run_id")]
                vals = []
                for e in experiments:
                    if e["run_id"] not in rids:
                        continue
                    v = (e.get("primary_metric") or {}).get("value") if idx == 0 and pm_key else (e.get("metrics") or {}).get(k)
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
                "key": "primary_metric" if idx == 0 and pm_key else k,
                "label": (f"主指标：{pm_key}" if idx == 0 and pm_key else _metric_label(k)),
                "is_primary": idx == 0 and pm_key,
                "higher_is_better": _higher_is_better(str(pm_key if idx == 0 and pm_key else k)),
                "values": row_values,
                "n": n_per_col,
                "std": std_only,
            })

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["指标"] + [c["label"] for c in columns])
    for mr in metric_rows:
        w.writerow([mr["label"]] + [_fmt(v) for v in mr["values"]])
    csv_text = buf.getvalue()

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
```

- [ ] **Step 4: 运行确认通过**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest tests/test_experiments.py tests/test_experiments_batch.py -q`

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/experiments.py tests/test_experiments.py
git commit -m "feat: 实验对比自动聚合重复批次并展示均值±标准差"
```

---

### Task 3: runner 元数据 + FastAPI 批量端点

**Files:**
- Modify: `app/runner.py`
- Modify: `app/main.py`
- Test: `tests/test_runner_meta.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: 写失败测试**

In `tests/test_runner_meta.py`, add:

```python
    def test_meta_exposes_batch_fields(self):
        rid = "20260913-111111-batch01"
        d = runner.run_dir_of(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({
            "name": "B", "group": "improved", "note": "n",
            "batch_id": "abc123", "batch_kind": "repeats", "repeat_index": 3,
        }, ensure_ascii=False), encoding="utf-8")
        (d / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        meta = runner.read_meta(rid)
        self.assertEqual(meta["batch_id"], "abc123")
        self.assertEqual(meta["batch_kind"], "repeats")
```

In `tests/test_main.py`, add after `test_compare_returns_sorted_metric_table`:

```python
    def test_batch_repeats_endpoint_forks_source(self):
        rid = "20260913-101500-ca1a01"
        _make_run(rid, name="基线-MLP", group="baseline", model="mlp", model_label="MLP",
                  metrics={"accuracy": 0.9}, primary={"name": "accuracy", "value": 0.9})
        cfg_path = runner.run_dir_of(rid) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["params"] = {"seed": 42}
        cfg["random_state"] = 42
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        made = []

        def fake_create(config, script):
            new = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            d = runner.run_dir_of(new)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            made.append((config, script))
            return new

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = self.client.post("/api/experiments/repeats", json={"run_id": rid, "count": 2},
                                 headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(made), 2)
        self.assertEqual(made[0][1], "train_torch.py")
        self.assertEqual(made[0][0]["batch_kind"], "repeats")
        self.assertEqual(made[0][0]["params"]["seed"], 42)
        self.assertEqual(made[1][0]["params"]["seed"], 43)

    def test_batch_ablation_endpoint_forks_source(self):
        rid = "20260913-101500-ca1a02"
        _make_run(rid, name="基线-MLP", group="baseline", model="mlp", model_label="MLP",
                  metrics={"accuracy": 0.9}, primary={"name": "accuracy", "value": 0.9})
        cfg_path = runner.run_dir_of(rid) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["params"] = {"dropout": 0.2}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        made = []

        def fake_create(config, script):
            new = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            d = runner.run_dir_of(new)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            made.append((config, script))
            return new

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = self.client.post("/api/experiments/ablation",
                                 json={"run_id": rid, "overrides": {"dropout": [0.5]}},
                                 headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(made[0][0]["batch_kind"], "ablation")
        self.assertEqual(made[0][0]["params"]["dropout"], 0.5)
        self.assertEqual(made[0][0]["group"], "ablation")
```

Note: `_make_run` currently hardcodes `model="random_forest"`; add `model`/`model_label` kwargs to it:

```python
def _make_run(rid, *, name="", group="baseline", note="", state="done",
              metrics=None, primary=None, model="random_forest", model_label="随机森林"):
    ...
        "model": model, "model_label": model_label, "params": {"n_estimators": 100},
```

- [ ] **Step 2: 运行确认失败**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest tests/test_runner_meta.py tests/test_main.py -q`

Expected: FAIL

- [ ] **Step 3: 实现**

In `app/runner.py` `read_meta`, change return dict:

```python
    return {
        "name": cfg.get("name") or "",
        "group": cfg.get("group") if cfg.get("group") in GROUPS else GROUP_BASELINE,
        "note": cfg.get("note") or "",
        "batch_id": str(cfg.get("batch_id") or ""),
        "batch_kind": str(cfg.get("batch_kind") or ""),
        "repeat_index": cfg.get("repeat_index"),
    }
```

In `list_runs`, add batch keys to each item; in `run_detail`, add the same three keys:

```python
            "batch_id": meta["batch_id"],
            "batch_kind": meta["batch_kind"],
            "repeat_index": meta["repeat_index"],
```

In `app/main.py`, add after `compare_experiments`:

```python
class BatchRepeatsReq(BaseModel):
    run_id: str
    count: int = 3


class BatchAblationReq(BaseModel):
    run_id: str
    overrides: dict[str, list] = {}


@app.post("/api/experiments/repeats")
def batch_repeats(req: BatchRepeatsReq):
    try:
        run_ids = experiments.create_batch_repeats(req.run_id, req.count)
        return {"ok": True, "run_ids": run_ids}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/experiments/ablation")
def batch_ablation(req: BatchAblationReq):
    try:
        run_ids = experiments.create_batch_ablation(req.run_id, req.overrides)
        return {"ok": True, "run_ids": run_ids}
    except ValueError as e:
        raise HTTPException(400, str(e))
```

- [ ] **Step 4: 运行确认通过**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest tests/test_runner_meta.py tests/test_main.py tests/test_experiments.py tests/test_experiments_batch.py -q`

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/runner.py app/main.py tests/test_runner_meta.py tests/test_main.py
git commit -m "feat: 实验元数据支持批次字段，新增批量端点"
```

---

### Task 4: 前端实验详情页一键重复 / 一键消融

**Files:**
- Modify: `web/app.js`
- Verify: `scripts/verify_wizard.py`

- [ ] **Step 1: 在 `openRun` 头部按钮区加入两个按钮（仅 done 显示）**

In `web/app.js`, inside `openRun`, change the buttons row to:

```js
          <div class="row-flex">
            ${running ? '<button class="btn danger small" id="btn-cancel">取消训练</button>' : ""}
            ${d.state === "done" ? '<button class="btn small" id="btn-repeat">重复 3 次</button><button class="btn small" id="btn-ablation">一键消融</button>' : ""}
            <button class="btn small" id="btn-reuse">复用参数</button>
            <button class="btn small" id="btn-run-ai">AI 分析结果</button>
            <button class="btn danger small" id="btn-del">删除</button>
            <button class="btn small" id="btn-back">返回列表</button>
          </div>
```

- [ ] **Step 2: 加入批次徽标与批量启动处理**

In `openRun`, after the `数据划分` row, add batch info when present:

```js
        ${d.batch_kind ? `<div class="row-flex mt8 small">${groupBadge(d.group)}<span class="batch-chip">${d.batch_kind === "repeats" ? `重复批次 #${esc(d.repeat_index || "")}` : "消融批次"}</span><span class="muted">${esc(d.batch_id || "")}</span></div>` : ""}
```

After `$("#btn-reuse").onclick = ...}, add:

```js
    const launchBatch = async (count, showToast = true) => {
      const btn = $("#btn-repeat");
      if (btn) { btn.disabled = true; btn.textContent = "提交中…"; }
      try {
        const r = await api("/api/experiments/repeats", { method: "POST", body: { run_id: id, count } });
        if (showToast) toast(`已创建 ${r.run_ids.length} 个重复实验`, "success");
        location.hash = "#/runs";
        return r.run_ids;
      } catch (e) { toast(e.message, "error"); if (btn) { btn.disabled = false; btn.textContent = "重复 3 次"; } }
    };
    if ($("#btn-repeat")) $("#btn-repeat").onclick = () => {
      const c = prompt("重复次数（2-10，默认 3）：", "3");
      if (c === null) return;
      const n = parseInt(c, 10);
      if (!n || n < 2 || n > 10) return toast("次数需在 2-10 之间", "error");
      launchBatch(n);
    };
```

- [ ] **Step 3: 一键消融动态面板**

After the repeat button handler in `openRun`, add:

```js
    if ($("#btn-ablation")) $("#btn-ablation").onclick = () => {
      if (d.state !== "done") return;
      const params = d.config.params || {};
      const spec = (((state.models || {})[d.config.task]) || {}).models?.[d.config.model] || {};
      const rows = Object.entries(params).map(([k, base]) => {
        const ps = (spec.params || {})[k] || {};
        let opts;
        if (ps.type === "bool") opts = [true, false].filter((v) => v !== base);
        else if (ps.type === "choice") opts = (ps.options || []).filter((v) => v !== base);
        else if (typeof base === "number") {
          opts = [0, 0.5, 1.5, 2].map((f) => Math.round(base * f * 100) / 100)
            .filter((v) => v !== base && Number.isFinite(v));
        } else opts = [];
        if (!opts.length) return "";
        return `<div class="form-row"><label>${esc(ps.label || k)}（当前 ${esc(String(base))}）</label>
          <div class="ablation-opts">${opts.map((v) => `<label class="chip-check"><input type="checkbox" data-p="${esc(k)}" value="${esc(v)}">${esc(String(v))}</label>`).join("")}</div></div>`;
      }).filter(Boolean).join("");
      if (!rows) return toast("该实验没有可消融的超参数", "error");
      const panel = document.createElement("div");
      panel.className = "panel mt14";
      panel.innerHTML = `<b>一键消融（每个勾选项派生一个新实验，其余参数与源实验完全一致）</b>
        <div class="form-grid mt8">${rows}</div>
        <div class="row-flex mt8"><button class="btn accent small" id="btn-ablation-go">启动消融实验</button><button class="btn small" id="btn-ablation-cancel">取消</button></div>`;
      const old = box.querySelector(".ablation-panel");
      if (old) old.remove();
      panel.classList.add("ablation-panel");
      box.querySelector(".meta-box").after(panel);
      $("#btn-ablation-cancel").onclick = () => panel.remove();
      $("#btn-ablation-go").onclick = async () => {
        const overrides = {};
        panel.querySelectorAll("input[type=checkbox]:checked").forEach((c) => {
          let v = c.value;
          if (v === "true") v = true; else if (v === "false") v = false;
          else if (v !== "" && !Number.isNaN(Number(v))) v = Number(v);
          (overrides[c.dataset.p] = overrides[c.dataset.p] || []).push(v);
        });
        if (!Object.keys(overrides).length) return toast("请至少勾选一个消融值", "error");
        const btn = $("#btn-ablation-go");
        btn.disabled = true; btn.textContent = "提交中…";
        try {
          const r = await api("/api/experiments/ablation", { method: "POST", body: { run_id: id, overrides } });
          toast(`已创建 ${r.run_ids.length} 个消融实验`, "success");
          location.hash = "#/runs";
        } catch (e) { toast(e.message, "error"); btn.disabled = false; btn.textContent = "启动消融实验"; }
      };
    };
```

- [ ] **Step 4: 在 `runsTable` 名称行/`openRun` 头部显示重复批次徽标**

In `web/app.js` `runsTable`, change the name cell subtitle:

```js
      <td><div class="cell-main">${esc(r.name || r.model_label || r.model || r.run_id)}</div><div class="cell-sub mt4">${groupBadge(r.group)}${r.batch_kind ? `<span class="batch-chip small">${r.batch_kind === "repeats" ? "重复" : "消融"}</span>` : ""}</div></td>
```

Append CSS in `web/style.css`:

```css
.batch-chip{display:inline-block;margin-left:6px;padding:1px 7px;border:1px solid var(--border,#cbd5e1);border-radius:999px;font-size:11px;color:var(--muted,#475569);background:#f8fafc;white-space:nowrap}
.ablation-opts{display:flex;flex-wrap:wrap;gap:8px}
.chip-check{display:inline-flex;align-items:center;gap:5px;padding:4px 8px;border:1px solid var(--border,#cbd5e1);border-radius:6px;font-size:12px;cursor:pointer}
.chip-check input{accent-color:var(--accent,#2563eb)}
```

- [ ] **Step 5: 静态回归 + 启动服务截图检查**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest tests/test_main.py tests/test_experiments.py tests/test_experiments_batch.py tests/test_runner_meta.py -q`

Expected: PASS

Run: `node --check web/app.js`（如 Node 可用；不可用则跳过并在提交说明中注明）

Expected: 无语法错误

- [ ] **Step 6: 提交**

```bash
git add web/app.js web/style.css
git commit -m "feat: 实验详情页支持一键重复与消融，列表显示批次徽标"
```

---

### Task 5: 全量回归 + 端到端冒烟

**Files:**
- Command only

- [ ] **Step 1: 全量测试**

Run: `D:\ZONGHESHEJI\tools\Python313\Scripts\python.exe -m pytest -q`

Expected: 全部通过

- [ ] **Step 2: 冒烟 API 调用**

先构造一个已完成 run（用现有 smoke 数据或做一个最小 sklearn run），再调用：

```powershell
$body = @{ run_id = "<done-run-id>"; count = 2 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/experiments/repeats" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
```

Expected: 返回 `ok: true` 与两个 `run_ids`

- [ ] **Step 3: 提交计划勾选**

`docs/superpowers/plans/2026-09-13-sprint3-evaluation-ablation.md` 中把本计划全部勾选，提交：

```bash
git add docs/superpowers/plans/2026-09-13-sprint3-evaluation-ablation.md
git commit -m "docs: Sprint 3 计划完成"
```
