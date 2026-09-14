# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import importlib.util
import json
import unittest
import uuid
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ExplainHelperTest(unittest.TestCase):
    def test_normalize_cam(self):
        from app.explain import normalize_cam

        result = normalize_cam([0.0, 1.0])
        self.assertAlmostEqual(float(result[-1]), 1.0)
        self.assertTrue(np.allclose(normalize_cam([2.0, 2.0]), [0.0, 0.0]))

    def test_grad_cam_image(self):
        import torch

        from app.explain import grad_cam_image
        from app.networks import ImageCNN

        net = ImageCNN(num_classes=3, conv_channels=(4, 8))
        x = torch.randn(2, 3, 32, 32)
        cam, pred = grad_cam_image(net, x)
        self.assertEqual(cam.ndim, 2)
        self.assertGreater(min(cam.shape), 0)
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)
        self.assertIn(pred, (0, 1, 2))

    def test_plot_token_attribution_skips_empty(self):
        import tempfile

        from app.explain import plot_token_attribution

        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(plot_token_attribution([], Path(td) / "out.png"))


@unittest.skipUnless(HAS_TORCH, "需要 PyTorch")
class ExplainIntegrationTest(unittest.TestCase):
    def _fit_tabular(self):
        d = _isolate.scratch("explain-tabular") / uuid.uuid4().hex[:10]
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        n = 40
        df = __import__("pandas").DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        })
        df["target"] = np.where(df["x1"] > 0, "pos", "neg")
        df.to_csv(d / "dataset.csv", index=False)
        cfg = {
            "name": "explain", "dataset_id": "explain", "dataset_name": "测试",
            "dataset_dir": str(d), "task": "tabular_classification", "model": "mlp",
            "model_label": "MLP", "params": {
                "hidden_sizes": "8", "dropout": 0.0, "optimizer": "adam",
                "lr": 0.02, "batch_size": 8, "epochs": 2,
            }, "target": "target", "test_size": 0.2, "val_split": 0.2,
        }
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        from app import trainer

        summary = trainer.fit(cfg, d)
        return d, summary

    def test_tabular_mlp_writes_permutation_importance(self):
        d, summary = self._fit_tabular()
        self.assertTrue((d / "permutation_importance.png").exists())
        self.assertIn("permutation_importance.png", summary["artifacts"])
        self.assertIn("permutation_importance.json", summary["artifacts"])

    def test_text_classification_writes_token_attribution(self):
        d = _isolate.scratch("explain-text") / uuid.uuid4().hex[:10]
        d.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        rows = [("很好 很棒 优秀", "pos"), ("很差 糟糕 失败", "neg")] * 20
        pd.DataFrame(rows, columns=["text", "target"]).to_csv(d / "dataset.csv", index=False)
        cfg = {
            "name": "explain-text", "dataset_id": "explain-text", "dataset_name": "文本",
            "dataset_dir": str(d), "task": "text_classification", "model": "textcnn",
            "model_label": "TextCNN", "text_column": "text", "target": "target",
            "params": {
                "embedding_dim": 8, "num_filters": 4, "kernel_sizes": "2,3",
                "dropout": 0.0, "optimizer": "adam", "lr": 0.01,
                "batch_size": 8, "epochs": 1, "max_seq_len": 16, "vocab_size": 200,
            }, "test_size": 0.2, "val_split": 0.2,
        }
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        from app import trainer

        summary = trainer.fit(cfg, d)
        self.assertTrue((d / "token_attribution.json").exists())
        payload = json.loads((d / "token_attribution.json").read_text(encoding="utf-8"))
        self.assertIn("attributions", payload)
        self.assertIn("token_attribution.json", summary["artifacts"])


if __name__ == "__main__":
    unittest.main()
