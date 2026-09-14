# Time Series Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a complete time-series forecasting workflow using chronological splitting, sliding windows, LSTM/GRU/Transformer networks, real evaluation artifacts, and full UI/API coverage.

**Architecture:** Register a new `time_series_forecasting` task in the model catalog. Extend the torch trainer with a dedicated `_load_time_series()` branch that reads tabular data, scales numeric columns using train-only statistics, splits chronologically, and creates `(lookback, horizon)` windows. Reuse the existing run runner, artifacts, comparison, tuning, and desktop/UI wiring; add a time-series network branch and final forecast plot.

**Tech Stack:** FastAPI, PyTorch, pandas, scikit-learn, matplotlib, pytest, vanilla JavaScript.

---

### Task 1: Catalog and Model Registry

**Files:**
- Modify: `app/catalog.py`
- Modify: `app/networks.py`
- Test: `tests/test_networks.py`

- [x] **Step 1: Add the task catalog**

```python
"time_series_forecasting": {
    "label": "时间序列 · 预测",
    "needs_target": True,
    "needs_text_column": False,
    "models": {
        "lstm": {
            "label": "LSTM 时间序列预测",
            "desc": "长短期记忆网络，按历史滑窗预测未来序列，适合单变量/多变量时间序列。",
            "engine": "torch",
            "params": {
                "lookback": {"type": "int", "default": 12, "min": 2, "max": 512, "label": "回看窗口"},
                "horizon": {"type": "int", "default": 1, "min": 1, "max": 128, "label": "预测步数"},
                "hidden_dim": {"type": "int", "default": 64, "min": 8, "max": 1024, "label": "隐藏单元数"},
                "num_layers": {"type": "int", "default": 1, "min": 1, "max": 8, "label": "层数"},
                "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                **_TRAINING_PARAMS,
            },
        },
        "gru": { "...": "GRU mirrors LSTM params" },
        "transformer": {
            "label": "Transformer 时间序列预测",
            "desc": "自注意力序列模型，能建模较长时间依赖，适合作为深度对照模型。",
            "engine": "torch",
            "params": {
                "lookback": {"type": "int", "default": 12, "min": 2, "max": 512, "label": "回看窗口"},
                "horizon": {"type": "int", "default": 1, "min": 1, "max": 128, "label": "预测步数"},
                "d_model": {"type": "int", "default": 64, "min": 16, "max": 1024, "label": "模型维度"},
                "nhead": {"type": "int", "default": 4, "min": 1, "max": 16, "label": "注意力头数"},
                "num_layers": {"type": "int", "default": 2, "min": 1, "max": 16, "label": "层数"},
                "dim_feedforward": {"type": "int", "default": 128, "min": 32, "max": 4096, "label": "前馈维度"},
                "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.95, "label": "丢弃率"},
                **_TRAINING_PARAMS,
            },
        },
    },
}
```

- [x] **Step 2: Add torch networks**

```python
class TimeSeriesRNN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers, horizon, dropout=0.1, rnn_type="lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(in_features, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=float(dropout) if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))


class TimeSeriesTransformer(nn.Module):
    def __init__(self, in_features, d_model, nhead, num_layers, dim_feedforward, horizon,
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        self.input = nn.Linear(in_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                           dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        out = self.input(x) + self.pos[:, :x.size(1), :]
        out = self.encoder(out).mean(dim=1)
        return self.fc(self.dropout(out))
```

- [x] **Step 3: Wire `build_model()`**

```python
if task == "time_series_forecasting":
    in_features = _int(dims.get("num_features"), 1, lo=1, hi=100000)
    horizon = _int(p.get("horizon"), 1, lo=1, hi=128)
    dropout = _float(p.get("dropout"), 0.1, 0.0, 0.95)
    if model in ("lstm", "gru"):
        return TimeSeriesRNN(in_features, _int(p.get("hidden_dim"), 64, lo=8, hi=1024),
                             _int(p.get("num_layers"), 1, lo=1, hi=8), horizon,
                             dropout=dropout, rnn_type=model)
    if model == "transformer":
        return TimeSeriesTransformer(in_features, _int(p.get("d_model"), 64, lo=16, hi=1024),
                                     _int(p.get("nhead"), 4, lo=1, hi=16),
                                     _int(p.get("num_layers"), 2, lo=1, hi=16),
                                     _int(p.get("dim_feedforward"), 128, lo=32, hi=4096),
                                     horizon, dropout=dropout,
                                     max_seq_len=_int(p.get("lookback"), 12, lo=2, hi=512))
    raise ValueError(f"时间序列不支持的模型: {model}")
```

- [x] **Step 4: Verify network shapes**

Run: `python -m pytest tests/test_networks.py -q`
Expected: PASS

---

### Task 2: Chronological Data and Training

**Files:**
- Modify: `app/trainer.py`
- Test: `tests/test_time_series.py`

- [x] **Step 1: Write a failing full-pipeline test**

```python
def test_time_series_full_pipeline():
    d = _run_dir("time-series")
    values = np.sin(np.linspace(0, 20, 90)) + np.arange(90) * 0.01
    pd.DataFrame({"value": values}).to_csv(d / "dataset.csv", index=False)
    cfg = _base_cfg(d / "dataset.csv", task="time_series_forecasting", model="lstm",
                    target="value", params={"lookback": 8, "horizon": 1, "hidden_dim": 8,
                    "num_layers": 1, "dropout": 0.0, "epochs": 1, "batch_size": 8})
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    rc = trainer.train_from_run_dir(d)
    assert rc == 0
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "time_series_forecasting"
    assert {"r2", "mae", "rmse"} <= set(summary["metrics"])
    assert (d / "curves.png").exists()
    assert (d / "forecast.png").exists()
```

- [x] **Step 2: Add the loader**

```python
def _load_time_series(config, st, device, emit):
    path = Path(config["dataset_dir"]) / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    frame = pd.read_csv(path)
    target = config.get("target") or (frame.columns or [None])[-1]
    if target not in frame.columns:
        raise ValueError(f"目标列 {target} 不在数据集中")
    y_values = pd.to_numeric(frame[target], errors="coerce")
    if y_values.isna().any():
        raise ValueError("时间序列目标列必须全部为数值")
    features = frame.select_dtypes(include=[np.number])
    if target not in features.columns:
        features[target] = y_values
    features = features.loc[:, features.columns.drop_duplicates()]
    values = features.to_numpy(dtype=np.float32)
    lookback = int(st.get("lookback", 12))
    horizon = int(st.get("horizon", 1))
    if len(values) < lookback + horizon + 2:
        raise ValueError("时间序列样本不足，无法完成回看窗口、预测步数和训练/验证/测试划分")
    test_n = max(1, int(round(len(values) * float(st["test_size"]))))
    val_n = max(1, int(round(len(values) * float(st["val_split"]))))
    train_end = len(values) - test_n - val_n
    train, val, test = values[:train_end], values[train_end:train_end + val_n], values[-test_n:]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    train = ((train - mean) / std).astype(np.float32)
    val = ((val - mean) / std).astype(np.float32)
    test = ((test - mean) / std).astype(np.float32)
    def windows(arr):
        xs, ys = [], []
        for i in range(len(arr) - lookback - horizon + 1):
            xs.append(arr[i:i + lookback])
            ys.append(arr[i + lookback:i + lookback + horizon, -1])
        return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))
    datasets = [torch.utils.data.TensorDataset(*windows(arr)) for arr in (train, val, test)]
    loaders = [torch.utils.data.DataLoader(d, batch_size=int(st["batch_size"]), shuffle=False) for d in datasets]
    return {"train": loaders[0], "val": loaders[1], "test": loaders[2],
            "feature_names": list(features.columns), "target_name": target,
            "num_features": values.shape[1], "lookback": lookback, "horizon": horizon,
            "scale_mean": mean.tolist(), "scale_std": std.tolist(),
            "split_scheme": "chronological train/val/test"}
```

- [x] **Step 3: Dispatch it**

```python
if task == "time_series_forecasting":
    return _load_time_series(config, st, device, emit)
```

- [x] **Step 4: Treat time series as regression**

```python
classification = task in ("tabular_classification", "text_classification", "image_classification")
```

- [x] **Step 5: Add a forecast plot and save time-series checkpoint fields**

```python
if task == "time_series_forecasting":
    plt = _plot_time_series(yt, y_pred_full, data["target_name"],
                            data["scale_mean"][-1], data["scale_std"][-1])
    _plot_pred_true(yt, y_pred_full, _safe(run_dir, "pred_vs_true.png"))
    artifacts.extend(["pred_vs_true.png", "forecast.png"])
```

- [x] **Step 6: Run tests**

Run: `python -m pytest tests/test_time_series.py -q`
Expected: PASS

---

### Task 3: API and UI

**Files:**
- Modify: `app/main.py`
- Modify: `web/app.js`
- Modify: `web/wizard.js`
- Test: `tests/test_main.py`

- [x] **Step 1: Accept tabular datasets for time-series runs**

```python
if task == "time_series_forecasting":
    if ds_meta["type"] != "tabular":
        raise HTTPException(400, "时间序列预测需要表格数据")
```

- [x] **Step 2: Show time-series task in the training page**

```javascript
const tasks = ds.type === "image"
  ? ["image_classification"]
  : ["tabular_classification", "tabular_regression", "text_classification", "time_series_forecasting"];
```

- [x] **Step 3: Let tabular datasets select time-series tasks**

```javascript
return keys.filter((k) => k.startsWith("tabular_") || k === "time_series_forecasting");
```

- [x] **Step 4: Add API validation test**

```python
def test_create_run_accepts_time_series():
    body = {"dataset_id": ds_id, "task": "time_series_forecasting", "model": "lstm",
            "params": {"lookback": 4, "horizon": 1, "epochs": 1}, "target": "value"}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 200
```

---

### Task 4: Docs and Regression Gate

**Files:**
- Create: `docs/时间序列预测指南.md`
- Modify: `README.md`

- [x] **Step 1: Document data format, splitting, parameters, and outputs**

```markdown
# 时间序列预测指南

时间序列任务使用 CSV/Excel 表格数据。至少包含一列连续数值目标，例如 `value`。

## 推荐设置
- 回看窗口：先设为 12，观察周期后调整。
- 预测步数：单步预测设为 1，多步预测可设为 3-12。
- 模型：先用 LSTM/GRU 跑通，再用 Transformer 对照。
- 验证：最终指标来自时间顺序划分后的测试段，不做随机打乱。
```

- [x] **Step 2: Run all checks**

```bash
python -m pytest -q
python scripts/verify_wizard.py
```

Expected: all tests and wizard checks pass.
