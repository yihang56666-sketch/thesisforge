# Auto Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a local Optuna-backed automatic hyperparameter search that submits real training runs, supports Hyperband pruning, and is usable from the UI.

**Architecture:** A new `app/tuning.py` builds a safe search space from `catalog.py`, launches the existing training scripts as per-trial subprocesses, reads each trial's `summary.json`, and uses Optuna to search and prune. A FastAPI endpoint exposes tuning actions; the experiment page gets a simple guided search panel.

**Tech Stack:** Optuna, FastAPI, Pydantic, existing runner and training scripts.

---

### Task 1: Tuning Engine

**Files:**
- Create: `app/tuning.py`
- Test: `tests/test_tuning.py`

- [x] **Step 1: Write the failing test**

```python
from unittest.mock import patch
from app import tuning

def test_build_search_space_supported():
    spec = tuning.build_search_space(
        "tabular_classification", "logistic_regression",
        {"n_trials": 8, "metric": "accuracy", "search_mode": "tpe"},
    )
    assert spec["n_trials"] == 8
    assert spec["metric"] == "accuracy"
    assert spec["space"]["C"]["type"] == "float"
    assert spec["space"]["C"]["low"] == 0.001
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tuning.py::test_build_search_space_supported -v`
Expected: FAIL, module not found.

- [x] **Step 3: Implement search space builder**

`build_search_space()` should copy numeric and choice parameters from the model catalog, preserve defaults, cap trial count between 2 and 200, and reject unsupported tasks or models.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tuning.py::test_build_search_space_supported -v`
Expected: PASS.

### Task 2: Optuna Objective

**Files:**
- Modify: `app/tuning.py`
- Test: `tests/test_tuning.py`

- [x] **Step 1: Add objective test**

```python
def test_objective_reports_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "_run_trial", lambda *a, **k: {"primary_metric": {"name": "accuracy", "value": 0.9}})
    assert tuning.objective_value({"primary_metric": {"name": "accuracy", "value": 0.9}}, "accuracy") == 0.9
```

- [x] **Step 2: Add metric helper**

`objective_value()` should use the requested metric when present, otherwise the run's primary metric. Raise a `TrialPruned` on missing metrics after the warm-up trial.

- [x] **Step 3: Add trial runner**

`_run_trial()` should create one temporary run directory, call the existing training subprocess, and wait for `summary.json`.

- [x] **Step 4: Add search test**

```python
def test_search_invokes_runner(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tuning, "_run_trial", lambda cfg, out, **k: calls.append(cfg) or {"primary_metric": {"name": "accuracy", "value": 0.8}})
    result = tuning.run_search(
        task="tabular_classification", model="logistic_regression",
        base_config={"dataset_dir": str(tmp_path)}, n_trials=2, metric="accuracy", search_mode="tpe", seed=7,
    )
    assert len(calls) == 2
    assert result["best_value"] >= 0
```

- [x] **Step 5: Run tuning tests**

Run: `python -m pytest tests/test_tuning.py -v`
Expected: PASS.

### Task 3: API Integration

**Files:**
- Modify: `app/main.py`
- Modify: `app/tuning.py`
- Test: `tests/test_tuning.py`

- [x] **Step 1: Add request models**

Create `TuningReq` with `task`, `model`, `params`, `dataset_id`, `n_trials`, `metric`, `search_mode`, and `seed`.

- [x] **Step 2: Add endpoint**

`POST /api/tuning/search` should validate the model, derive dataset defaults, run the search, and return the best params, value, study history, and saved search file.

- [x] **Step 3: Add API test**

Use FastAPI TestClient and monkeypatch `tuning.run_search` to return a fixed result.

- [x] **Step 4: Run API test**

Run: `python -m pytest tests/test_tuning.py -v`
Expected: PASS.

### Task 4: UI Entry

**Files:**
- Modify: `web/app.js`
- Modify: `web/style.css`
- Test: `scripts/verify_wizard.py`

- [x] **Step 1: Add guided panel**

Add a tuning card on the model training page with search trials, metric, search mode, and a “开始自动调参” button.

- [x] **Step 2: Render recommendations**

After search completes, show best parameters, best score, and history.

- [x] **Step 3: Smoke check**

Run: `python scripts/verify_wizard.py`
Expected: PASS.

### Task 5: Docs and Packaging

**Files:**
- Create: `docs/自动调参指南.md`
- Modify: `README.md`
- Modify: `requirements.txt`

- [x] **Step 1: Add dependency**

Append `optuna>=3.6` to `requirements.txt`.

- [x] **Step 2: Write user guide**

Explain search modes, metric choice, trial cost, Hyperband pruning, and how to promote the best run.

- [x] **Step 3: Run full tests**

Run: `python -m pytest -q`
Expected: PASS.
