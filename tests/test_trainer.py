# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import importlib.util
import json
import math
import unittest
import uuid

import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _run_dir(name: str):
    d = _isolate.scratch(f"trainer-{name}") / uuid.uuid4().hex[:10]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_run(cfg: dict, run_dir) -> None:
    (run_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _classification_csv(path, n: int = 60) -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "cat": np.tile(["a", "b", "c"], n // 3 + 1)[:n],
    })
    df["target"] = np.where(df["x1"] + 0.4 * df["x2"] > 0, "pos", "neg")
    df.to_csv(path, index=False)


def _regression_csv(path, n: int = 60) -> None:
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 2 * df["x1"] - 1.5 * df["x2"] + rng.normal(scale=0.1, size=n)
    df.to_csv(path, index=False)


def _text_csv(path, n: int = 40) -> None:
    rows = []
    for i in range(n):
        label = 1 if i % 2 == 0 else 0
        text = "很好 很棒 优秀" if label else "很差 糟糕 失败"
        rows.append((text, label))
    pd.DataFrame(rows, columns=["text", "target"]).to_csv(path, index=False)


def _base_cfg(dataset_path, *, task="tabular_classification", model="mlp",
              params=None, ds_name="测试数据", target="target") -> dict:
    ds_dir = dataset_path.parent
    return {
        "name": "trainer-test", "dataset_id": "trainer-test-ds",
        "dataset_name": ds_name, "dataset_dir": str(ds_dir),
        "task": task, "model": model, "model_label": "MLP",
        "params": params or {}, "target": target,
        "text_column": "text", "test_size": 0.2, "val_split": 0.2,
        "random_state": 42,
    }


@unittest.skipUnless(HAS_TORCH, "需要 PyTorch")
class TrainerTest(unittest.TestCase):
    def _fit(self, cfg: dict, run_dir):
        from app import trainer

        return trainer.fit(cfg, run_dir)

    def test_tabular_mlp_full_pipeline(self):
        d = _run_dir("tabular")
        csv = d / "dataset.csv"
        _classification_csv(csv)
        cfg = _base_cfg(csv, params={
            "hidden_sizes": "8,4", "activation": "relu", "dropout": 0.0,
            "optimizer": "adam", "lr": 0.01, "batch_size": 8, "epochs": 2,
        })
        _write_run(cfg, d)
        from app import trainer

        rc = trainer.train_from_run_dir(d)
        self.assertEqual(rc, 0)
        status = json.loads((d / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "done")
        metrics = [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x]
        self.assertTrue(any(x["type"] == "epoch" for x in metrics))
        self.assertTrue(any(x["type"] == "summary" for x in metrics))
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        self.assertGreater(summary["metrics"]["accuracy"], 0)
        for k, v in summary["metrics"].items():
            self.assertIsInstance(v, (int, float), k)
            self.assertFalse(math.isnan(v), k)
            self.assertTrue(math.isfinite(v), k)
        self.assertNotIn("NaN", (d / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("primary_metric", summary)
        self.assertTrue((d / "best.pt").exists())
        self.assertTrue((d / "curves.png").exists())
        self.assertTrue((d / "confusion_matrix.png").exists())

    def test_all_optimizers_available(self):
        for opt in ("sgd", "sgd_momentum", "adam", "adamw", "rmsprop"):
            with self.subTest(optimizer=opt):
                d = _run_dir(f"opt-{opt}")
                csv = d / "dataset.csv"
                _classification_csv(csv)
                cfg = _base_cfg(csv, params={
                    "hidden_sizes": "6", "dropout": 0.0, "optimizer": opt,
                    "lr": 0.01, "batch_size": 16, "epochs": 1,
                })
                _write_run(cfg, d)
                self._fit(cfg, d)
                status = json.loads((d / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["state"], "done", opt)

    def test_all_schedulers_available(self):
        for sch in ("cosine", "step", "plateau", "none"):
            with self.subTest(scheduler=sch):
                d = _run_dir(f"sch-{sch}")
                csv = d / "dataset.csv"
                _classification_csv(csv)
                cfg = _base_cfg(csv, params={
                    "hidden_sizes": "6", "dropout": 0.0, "optimizer": "adam",
                    "lr": 0.01, "batch_size": 16, "epochs": 1,
                    "scheduler": sch, "step_size": 1, "gamma": 0.5,
                })
                _write_run(cfg, d)
                self._fit(cfg, d)
                status = json.loads((d / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["state"], "done", sch)

    def test_early_stop_breaks_early(self):
        d = _run_dir("early-stop")
        csv = d / "dataset.csv"
        rng = np.random.default_rng(5)
        n = 24
        df = pd.DataFrame({
            "x1": np.repeat([5.0, -5.0], n // 2),
            "x2": np.repeat([1.0, -1.0], n // 2),
            "target": np.repeat(["p", "n"], n // 2),
        })
        df.to_csv(csv, index=False)
        cfg = _base_cfg(csv, params={
            "hidden_sizes": "16", "dropout": 0.0, "optimizer": "adam",
            "lr": 0.05, "batch_size": 8, "epochs": 20,
            "early_stop_patience": 1,
        })
        _write_run(cfg, d)
        self._fit(cfg, d)
        metrics = [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x]
        epoch_logs = [x for x in metrics if x["type"] == "epoch"]
        self.assertLess(len(epoch_logs), 20)
        self.assertLessEqual(epoch_logs[-1]["epoch"], 5)

    def test_text_and_regression(self):
        d = _run_dir("text")
        csv = d / "dataset.csv"
        _text_csv(csv)
        cfg = _base_cfg(csv, task="text_classification", model="textcnn", params={
            "embedding_dim": 8, "num_filters": 4, "kernel_sizes": "2,3",
            "dropout": 0.0, "optimizer": "adam", "lr": 0.01,
            "batch_size": 8, "epochs": 1, "max_seq_len": 16, "vocab_size": 200,
        })
        _write_run(cfg, d)
        self._fit(cfg, d)
        self.assertEqual(json.loads((d / "status.json").read_text(encoding="utf-8"))["state"], "done")

        d2 = _run_dir("reg")
        csv2 = d2 / "dataset.csv"
        _regression_csv(csv2)
        cfg2 = _base_cfg(csv2, task="tabular_regression", target="y", params={
            "hidden_sizes": "8", "dropout": 0.0, "optimizer": "adam",
            "lr": 0.01, "batch_size": 8, "epochs": 1,
        })
        _write_run(cfg2, d2)
        self._fit(cfg2, d2)
        summary = json.loads((d2 / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("r2", summary["metrics"])
        self.assertTrue((d2 / "pred_vs_true.png").exists())

    def test_bad_config_writes_failed(self):
        d = _run_dir("bad")
        csv = d / "dataset.csv"
        _classification_csv(csv)
        cfg = _base_cfg(csv, params={"optimizer": "no_such_opt", "epochs": 1})
        _write_run(cfg, d)
        from app import trainer

        rc = trainer.train_from_run_dir(d)
        self.assertNotEqual(rc, 0)
        status = json.loads((d / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "failed")

        d2 = _run_dir("bad-model")
        csv2 = d2 / "dataset.csv"
        _classification_csv(csv2)
        cfg2 = _base_cfg(csv2, task="tabular_classification", model="nope")
        _write_run(cfg2, d2)
        rc2 = trainer.train_from_run_dir(d2)
        self.assertNotEqual(rc2, 0)
        status2 = json.loads((d2 / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status2["state"], "failed")


if __name__ == "__main__":
    unittest.main()
