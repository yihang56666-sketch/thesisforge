# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import importlib.util
import json
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _run_dir(name: str):
    d = _isolate.scratch(f"time-series-{name}") / uuid.uuid4().hex[:10]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _series_csv(path, n: int = 90) -> None:
    values = np.sin(np.linspace(0, 20, n)) + np.arange(n) * 0.01
    pd.DataFrame({"value": values}).to_csv(path, index=False)


def _base_cfg(dataset_path, target="value", model="lstm") -> dict:
    return {
        "name": "time-series-test", "dataset_id": "time-series-test-ds",
        "dataset_name": "时间序列测试", "dataset_dir": str(dataset_path.parent),
        "task": "time_series_forecasting", "model": model,
        "model_label": "LSTM 时间序列预测",
        "params": {
            "lookback": 8, "horizon": 1, "hidden_dim": 8, "num_layers": 1,
            "dropout": 0.0, "optimizer": "adam", "lr": 0.01,
            "batch_size": 8, "epochs": 1,
        },
        "target": target, "text_column": None,
        "test_size": 0.2, "val_split": 0.2, "random_state": 42,
    }


@unittest.skipUnless(HAS_TORCH, "需要 PyTorch")
class TimeSeriesTest(unittest.TestCase):
    def _fit(self, cfg: dict, run_dir):
        from app import trainer

        return trainer.fit(cfg, run_dir)

    def test_time_series_full_pipeline(self):
        d = _run_dir("lstm")
        csv = d / "dataset.csv"
        _series_csv(csv)
        cfg = _base_cfg(csv)
        (d / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from app import trainer

        self.assertEqual(trainer.train_from_run_dir(d), 0)
        status = json.loads((d / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "done")
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["task"], "time_series_forecasting")
        self.assertEqual(summary["split_scheme"], "chronological train/val/test")
        self.assertGreater(summary["n_train"], 0)
        self.assertGreater(summary["n_val"], 0)
        self.assertGreater(summary["n_test"], 0)
        self.assertGreater(summary["metrics"]["r2"], -2)
        self.assertTrue((d / "curves.png").exists())
        self.assertTrue((d / "pred_vs_true.png").exists())
        self.assertTrue((d / "forecast.png").exists())

    def test_all_time_series_models_build_and_run(self):
        for model in ("lstm", "gru", "transformer"):
            with self.subTest(model=model):
                d = _run_dir(model)
                csv = d / "dataset.csv"
                _series_csv(csv)
                cfg = _base_cfg(csv, model=model)
                if model == "transformer":
                    cfg["params"].update({"d_model": 16, "nhead": 2, "dim_feedforward": 32})
                cfg["params"]["epochs"] = 1
                (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
                self._fit(cfg, d)
                summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["model"], model)
                self.assertTrue((d / "forecast.png").exists())

    def test_insufficient_data_raises(self):
        d = _run_dir("short")
        csv = d / "dataset.csv"
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)
        cfg = _base_cfg(csv)
        cfg["params"]["lookback"] = 8
        with self.assertRaises(ValueError):
            self._fit(cfg, d)


if __name__ == "__main__":
    unittest.main()
