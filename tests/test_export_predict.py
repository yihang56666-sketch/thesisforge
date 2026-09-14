# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _run_dir(name: str) -> Path:
    d = _isolate.scratch(f"export-predict-{name}") / uuid.uuid4().hex[:10]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_sklearn_config(run_dir: Path, dataset_dir: Path) -> None:
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_dir.name,
        "name": "sklearn-export-test",
        "group": "baseline",
        "dataset_name": "表格分类测试",
        "dataset_dir": str(dataset_dir),
        "task": "tabular_classification",
        "model": "logistic_regression",
        "model_label": "逻辑回归",
        "params": {"C": 1.0, "max_iter": 1000},
        "target": "target",
        "text_column": "text",
        "prep": {},
        "test_size": 0.25,
        "val_split": 0.2,
        "random_state": 42,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


class ExportedPredictScriptTest(unittest.TestCase):
    def test_sklearn_export_can_infer_new_row(self):
        run_dir = _run_dir("sklearn")
        dataset_dir = run_dir / "dataset"
        dataset_dir.mkdir()
        rng = np.random.default_rng(13)
        frame = pd.DataFrame({
            "x1": rng.normal(size=60),
            "x2": rng.normal(size=60),
            "cat": np.tile(["a", "b", "c"], 20),
        })
        frame["target"] = np.where(frame["x1"] - 0.3 * frame["x2"] > 0, "yes", "no")
        frame.to_csv(dataset_dir / "dataset.csv", index=False)
        _write_sklearn_config(run_dir, dataset_dir)

        train = subprocess.run(
            [sys.executable, str(ROOT / "app" / "train_sklearn.py"), "--run-dir", str(run_dir)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(train.returncode, 0, train.stdout[-800:] + train.stderr[-800:])
        self.assertTrue((run_dir / "model.pkl").exists())
        self.assertTrue((run_dir / "predict.py").exists())

        sample = run_dir / "sample.csv"
        pd.DataFrame([{"x1": 1.2, "x2": -0.4, "cat": "a"}]).to_csv(sample, index=False)
        pred = subprocess.run(
            [sys.executable, str(run_dir / "predict.py"), "--input", str(sample)],
            cwd=str(run_dir), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(pred.returncode, 0, pred.stderr[-800:])
        payload = json.loads(pred.stdout.strip())
        self.assertIn(payload["prediction"], {"yes", "no"})
        self.assertEqual(len(payload.get("probabilities", [])), 2)

    def test_torch_export_finds_offline_package_root_via_runtime_python(self):
        from app.export_predict import write_predict_script

        run_dir = _run_dir("torch-root")
        script = write_predict_script(run_dir, "torch", {
            "task": "tabular_classification",
            "model": "mlp",
            "params": {"hidden_sizes": "4"},
            "target": "target",
        })

        package = _run_dir("offline-package")
        (package / "runtime").mkdir()
        (package / "app").mkdir()
        (package / "app" / "__init__.py").write_text("", encoding="utf-8")
        (package / "app" / "networks.py").write_text(
            "def build_model(*args, **kwargs):\n"
            "    raise RuntimeError('help should not build a model')\n",
            encoding="utf-8",
        )
        fake_python = package / "runtime" / "python.exe"

        injection = _run_dir("sitecustomize")
        (injection / "sitecustomize.py").write_text(
            "import sys\nsys.executable = " + repr(str(fake_python)) + "\n",
            encoding="utf-8",
        )
        outside = _run_dir("outside-cwd")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(injection)
        env.pop("THESISFORGE_ROOT", None)
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(outside), env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-1200:])
        self.assertIn("--input", proc.stdout)


if __name__ == "__main__":
    unittest.main()
