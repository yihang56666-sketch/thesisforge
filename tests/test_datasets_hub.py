# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import io
import unittest
import zipfile
from pathlib import Path

from app import datasets_hub

try:
    from PIL import Image  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class DatasetDirTest(unittest.TestCase):
    def test_valid_id_accepted(self):
        self.assertEqual(datasets_hub.dataset_dir("iris"), datasets_hub.DATASETS_DIR / "iris")

    def test_rejects_path_traversal_and_empty(self):
        for bad in ["", " ", "..", "../evil", "a/../b", "a\\..\\b"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    datasets_hub.dataset_dir(bad)

    def test_rejects_absolute_path(self):
        with self.assertRaises(ValueError):
            datasets_hub.dataset_dir(str(Path("C:/Windows").resolve()))

    def test_rejects_overlong_id(self):
        with self.assertRaises(ValueError):
            datasets_hub.dataset_dir("a" * 81)


class ImportBytesTest(unittest.TestCase):
    def test_csv_roundtrip(self):
        csv_bytes = "a,b,target\n1,2,pos\n3,4,neg\n5,6,pos\n".encode("utf-8")
        meta = datasets_hub.import_bytes("sample.csv", csv_bytes)
        self.assertEqual(meta["type"], "tabular")
        self.assertEqual(meta["n_rows"], 3)
        self.assertEqual(meta["target"], "target")
        csv_path = datasets_hub.dataset_dir(meta["id"]) / "dataset.csv"
        self.assertEqual(csv_path.read_text(encoding="utf-8").count("\n"), 4)


class ZipSlipTest(unittest.TestCase):
    def test_parent_traversal_member_is_skipped(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../escape.txt", "evil")
            zf.writestr("a/ok.txt", "ok")
        dest = _isolate.scratch("zip-slip-parent")
        datasets_hub._safe_extract_zip(io.BytesIO(buf.getvalue()), dest)
        self.assertFalse((dest.parent / "escape.txt").exists())
        self.assertTrue((dest / "a" / "ok.txt").is_file())

    def test_absolute_members_are_skipped(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(zipfile.ZipInfo("C:/windows/win.ini"), "x")
            zf.writestr(zipfile.ZipInfo("/etc/passwd"), "x")
        dest = _isolate.scratch("zip-slip-absolute")
        datasets_hub._safe_extract_zip(io.BytesIO(buf.getvalue()), dest)
        self.assertEqual(list(dest.rglob("*")), [])


class BuiltinDemoTest(unittest.TestCase):
    def test_builtin_text_demo_loads(self):
        meta = datasets_hub.load_builtin("pseudo_text")
        self.assertEqual(meta["type"], "tabular")
        self.assertEqual(meta["task"], "text_classification")
        self.assertEqual(meta["n_rows"], 320)
        self.assertEqual(meta["text_column"], "text")
        self.assertEqual(meta["target"], "label")
        self.assertIn("duplicates", meta["stats"])

    @unittest.skipUnless(HAS_PIL, "需要 Pillow 才能生成内置图像示例")
    def test_builtin_image_demo_loads(self):
        meta = datasets_hub.load_builtin("pseudo_image")
        self.assertEqual(meta["type"], "image")
        self.assertEqual(meta["task"], "image_classification")
        self.assertEqual(meta["n_classes"], 5)
        self.assertEqual(meta["n_images"], 175)

    def test_eda_counts_duplicates(self):
        csv_bytes = "a,b,target\n1,2,x\n1,2,x\n3,4,y\n5,6,y\n5,6,y\n".encode("utf-8")
        meta = datasets_hub.import_bytes("dup.csv", csv_bytes)
        self.assertEqual(meta["stats"]["duplicates"], 2)
        self.assertAlmostEqual(meta["stats"]["duplicate_rate"], 0.4)

    def test_eda_generates_real_data_quality_warnings(self):
        rows = ["feature,label"]
        # 9 个重复特征样本，label 分布 7:1，模拟小样本且类别不平衡。
        for i in range(7):
            rows.append(f"{i},major")
        rows.append("8,minor")
        rows.append(",major")
        rows.append("0,major")
        rows.append("9,major")
        meta = datasets_hub.import_bytes("quality.csv", "\n".join(rows).encode("utf-8"))

        warnings = meta.get("quality_warnings") or []
        text = "\n".join(warnings)
        self.assertIn("缺失值", text)
        self.assertIn("重复样本", text)
        self.assertIn("类别分布不均", text)
        self.assertIn("样本规模较小", text)


if __name__ == "__main__":
    unittest.main()
