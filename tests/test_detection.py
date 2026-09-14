# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import io
import json
import random
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from app import datasets_hub, runner
from app.train_detection import yolo_weights


def _png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _make_yolo_zip(per_class: int = 4) -> bytes:
    rng = random.Random(42)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for split, n in (("train", per_class * 2), ("val", per_class)):
            for i in range(n):
                cls = i % 2
                img = Image.new("RGB", (64, 64), (30, 34, 40))
                d = ImageDraw.Draw(img)
                x0 = rng.randint(8, 28)
                y0 = rng.randint(8, 28)
                if cls == 0:
                    d.rectangle([x0, y0, x0 + 20, y0 + 14], fill=(220, 210, 190))
                else:
                    d.ellipse([x0, y0, x0 + 16, y0 + 16], fill=(120, 180, 220))
                zf.writestr(f"images/{split}/{i:03d}.png", _png_bytes(img))
                zf.writestr(f"labels/{split}/{i:03d}.txt",
                            f"{cls} 0.5 0.5 0.3 0.3\n")
        zf.writestr("data.yaml",
                    "path: .\ntrain: images/train\nval: images/val\n"
                    "names:\n  0: box\n  1: ball\n")
    return buf.getvalue()


class DetectionImportTest(unittest.TestCase):
    def test_import_yolo_zip(self):
        meta = datasets_hub.import_bytes("yolo-demo.zip", _make_yolo_zip())
        self.assertEqual(meta["task"], "object_detection")
        self.assertEqual(meta["n_images"], 12)
        self.assertEqual(meta["classes"], ["box", "ball"])
        self.assertGreater(meta.get("n_boxes", 0), 0)
        eda = meta.get("eda_files") or []
        self.assertIn("class_balance.png", eda)


class SegmentationImportTest(unittest.TestCase):
    def test_import_yolo_segment_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for split, n in (("train", 3), ("val", 2)):
                for i in range(n):
                    img = Image.new("RGB", (64, 64), (30, 34, 40))
                    zf.writestr(f"images/{split}/{i:03d}.png", _png_bytes(img))
                    zf.writestr(f"labels/{split}/{i:03d}.txt", "0 0.2 0.2 0.5 0.2 0.5 0.6\n")
            zf.writestr("data.yaml",
                        "task: segment\npath: .\ntrain: images/train\nval: images/val\n"
                        "names:\n  0: target\n")
        meta = datasets_hub.import_bytes("seg-demo.zip", buf.getvalue())
        self.assertEqual(meta["task"], "semantic_segmentation")
        self.assertEqual(meta["n_images"], 5)
        self.assertEqual(meta["classes"], ["target"])


class DetectionWeightMappingTest(unittest.TestCase):
    def test_model_weight_mapping(self):
        cases = {
            "yolov8n": "yolov8n.pt",
            "yolov8s": "yolov8s.pt",
            "yolov8n-seg": "yolov8n-seg.pt",
            "yolov8s-seg": "yolov8s-seg.pt",
            "rtdetr-l": "rtdetr-l.pt",
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(yolo_weights(model), expected)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            yolo_weights("unknown")


class DetectionApiTest(unittest.TestCase):
    def test_create_detection_run(self):
        meta = datasets_hub.import_bytes("yolo-api.zip", _make_yolo_zip())
        ds_id = meta["id"]
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        scripts = []

        def fake_create(config, script):
            scripts.append(script)
            rid = "20260914-000000-dead01"
            d = runner.run_dir_of(rid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False),
                                           encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}),
                                           encoding="utf-8")
            return rid

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = client.post("/api/runs", json={
                "dataset_id": ds_id, "task": "object_detection",
                "model": "yolov8n", "params": {"epochs": 1, "imgsz": 160},
                "name": "检测基线", "group": "baseline",
            }, headers=headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(scripts, ["train_detection.py"])
