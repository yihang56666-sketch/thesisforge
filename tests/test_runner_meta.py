# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import json
import unittest

from app import runner


class ReadMetaTest(unittest.TestCase):
    def test_missing_config_returns_defaults(self):
        rid = "20260913-101500-a1b2c3"
        d = runner.run_dir_of(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").unlink(missing_ok=True)
        self.assertEqual(runner.read_meta(rid),
                         {"name": "", "group": "baseline", "note": ""})

    def test_invalid_group_falls_back_to_baseline(self):
        rid = "20260913-101500-a1b2c4"
        d = runner.run_dir_of(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"group": "oops"}), encoding="utf-8")
        self.assertEqual(runner.read_meta(rid)["group"], "baseline")


class UpdateMetaTest(unittest.TestCase):
    def setUp(self):
        self.rid = "20260913-101500-b2c3d4"
        self.d = runner.run_dir_of(self.rid)
        self.d.mkdir(parents=True, exist_ok=True)

    def test_write_meta_and_persist(self):
        meta = runner.update_meta(self.rid, {"name": "基线实验", "group": "improved", "note": "备注"})
        self.assertEqual(meta["name"], "基线实验")
        self.assertEqual(meta["group"], "improved")
        self.assertEqual(meta["note"], "备注")
        cfg = json.loads((self.d / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["group"], "improved")
        self.assertEqual(runner.read_meta(self.rid), meta)

    def test_trims_and_truncates_values(self):
        meta = runner.update_meta(self.rid, {"name": "  " + "名" * 90 + "  ", "note": "  备注" * 200})
        self.assertEqual(len(meta["name"]), 80)
        self.assertEqual(len(meta["note"]), 500)
        self.assertFalse(meta["name"].startswith(" "))

    def test_invalid_group_rejected_without_write(self):
        with self.assertRaises(ValueError):
            runner.update_meta(self.rid, {"group": "weird"})
        self.assertFalse((self.d / "config.json").exists())

    def test_missing_run_raises(self):
        with self.assertRaises(FileNotFoundError):
            runner.update_meta("20260913-101500-b2c3d5", {"name": "x"})


class ListRunsMetaTest(unittest.TestCase):
    def test_list_runs_includes_meta(self):
        rid = "20260913-101500-c3d4e5"
        d = runner.run_dir_of(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(
            json.dumps({"name": "列表实验", "group": "ablation", "note": "n"}),
            encoding="utf-8")
        (d / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps({}), encoding="utf-8")
        item = next(x for x in runner.list_runs() if x["run_id"] == rid)
        self.assertEqual(item["name"], "列表实验")
        self.assertEqual(item["group"], "ablation")
        self.assertEqual(item["note"], "n")


if __name__ == "__main__":
    unittest.main()
