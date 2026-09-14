# -*- coding: utf-8 -*-
import unittest
import _isolate  # noqa: F401

from pathlib import Path

from app import catalog, model_scanner
from app.train_detection import find_model_weights, yolo_weights


class Yolo26CatalogTest(unittest.TestCase):
    def test_detection_catalog_has_yolo26(self):
        models = catalog.CATALOG["object_detection"]["models"]
        self.assertIn("yolo26n", models)
        self.assertIn("yolo26s", models)
        self.assertEqual(models["yolo26n"]["engine"], "ultralytics")

    def test_segmentation_catalog_has_yolo26(self):
        models = catalog.CATALOG["semantic_segmentation"]["models"]
        self.assertIn("yolo26n-seg", models)
        self.assertIn("yolo26s-seg", models)
        self.assertEqual(models["yolo26n-seg"]["engine"], "ultralytics")


class Yolo26WeightMappingTest(unittest.TestCase):
    def test_model_weight_mapping(self):
        cases = {
            "yolo26n": "yolo26n.pt",
            "yolo26s": "yolo26s.pt",
            "yolo26n-seg": "yolo26n-seg.pt",
            "yolo26s-seg": "yolo26s-seg.pt",
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(yolo_weights(model), expected)

    def test_find_model_weights_scans_user_model_dir(self, *_):
        root = _isolate.scratch("yolo26-weights")
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        expected = model_dir / "yolo26n.pt"
        expected.write_bytes(b"stub")
        self.assertEqual(Path(find_model_weights("yolo26n", [model_dir])), expected)

    def test_scan_local_models(self, *_):
        root = _isolate.scratch("model-scanner")
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "yolo26n.pt").write_bytes(b"stub")
        (model_dir / "my-yolo26n-seg.pt").write_bytes(b"stub")
        found = model_scanner.scan_local_models([model_dir])
        names = [m["filename"] for m in found]
        self.assertIn("yolo26n.pt", names)
        self.assertIn("my-yolo26n-seg.pt", names)
        self.assertTrue(any(m["model"] == "yolo26n" and m["task"] == "object_detection" for m in found))
        self.assertTrue(any(m["model"] == "yolo26n-seg" and m["task"] == "semantic_segmentation" for m in found))

    def test_scan_local_models_uses_env_dir(self, *_):
        root = _isolate.scratch("model-scanner-env")
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "yolo26n.pt").write_bytes(b"stub")
        found = model_scanner.scan_local_models([model_dir])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["model"], "yolo26n")

    def test_scan_local_models_supports_custom_pt(self, *_):
        root = _isolate.scratch("custom-model-scanner")
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "my-detector.pt").write_bytes(b"stub")
        (model_dir / "my-mask.pt").write_bytes(b"stub")
        found = model_scanner.scan_local_models([model_dir])
        self.assertEqual(len(found), 2)
        self.assertTrue(all(m["model"].startswith("local-") for m in found))
        self.assertTrue(any(m["task"] == "object_detection" for m in found))
        self.assertTrue(any(m["task"] == "semantic_segmentation" for m in found))

    def test_local_model_spec_returns_weights_path(self, *_):
        root = _isolate.scratch("custom-model-spec")
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "my-detector.pt"
        path.write_bytes(b"stub")
        found = model_scanner.scan_local_models([model_dir])
        model = found[0]["model"]
        spec = model_scanner.get_local_model_spec("object_detection", model, [model_dir])
        self.assertEqual(spec["engine"], "ultralytics")
        self.assertEqual(spec["weights_path"], str(path))
        self.assertIn("epochs", spec["params"])


if __name__ == "__main__":
    unittest.main()
