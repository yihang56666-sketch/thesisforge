# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import io
import json
import unittest
import zipfile
from unittest import mock

from PIL import Image

from app import datasets_hub, runner
from app.catalog import CATALOG


def _png_bytes() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 34, 40)).save(b, format="PNG")
    return b.getvalue()


def _mask_bytes() -> bytes:
    b = io.BytesIO()
    Image.new("L", (64, 64), 1).save(b, format="PNG")
    return b.getvalue()


def _mask_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for split, n in (("train", 3), ("val", 2)):
            for i in range(n):
                zf.writestr(f"images/{split}/{i:03d}.png", _png_bytes())
                zf.writestr(f"masks/{split}/{i:03d}.png", _mask_bytes())
        zf.writestr("classes.txt", "background\nwater\n")
    return buf.getvalue()


class SemanticSegmentationTest(unittest.TestCase):
    def test_import_pixel_mask_zip(self):
        meta = datasets_hub.import_bytes("semantic-demo.zip", _mask_zip())
        self.assertEqual(meta["task"], "semantic_segmentation")
        self.assertEqual(meta["data_format"], "pixel_masks")
        self.assertEqual(meta["n_images"], 5)
        self.assertEqual(meta["classes"], ["background", "water"])

    def test_catalog_has_unet(self):
        model = CATALOG["semantic_segmentation"]["models"]["unet"]
        self.assertEqual(model["engine"], "torch")

    def test_create_unet_run_routes_to_semantic_script(self):
        meta = datasets_hub.import_bytes("semantic-api.zip", _mask_zip())
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        scripts = []

        def fake_create(config, script):
            scripts.append(script)
            rid = "20260914-000000-dead02"
            d = runner.run_dir_of(rid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False),
                                           encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}),
                                           encoding="utf-8")
            return rid

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = client.post("/api/runs", json={
                "dataset_id": meta["id"], "task": "semantic_segmentation",
                "model": "unet", "params": {"epochs": 1, "imgsz": 64},
                "name": "语义分割", "group": "baseline",
            }, headers=headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(scripts, ["train_semantic.py"])

    def test_unet_output_shape(self):
        import torch
        from app.networks import build_model

        net = build_model("semantic_segmentation", "unet", {"base_channels": 8}, num_classes=2)
        out = net(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 2, 64, 64))
