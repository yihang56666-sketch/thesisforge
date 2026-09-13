# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import json
import unittest

from app import experiments, runner


def _make_done_run(rid, params=None, seed=42, model="mlp", model_label="MLP",
                   group="baseline", name="基线-MLP"):
    d = runner.run_dir_of(rid)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dataset_id": "iris", "dataset_name": "鸢尾花", "task": "tabular_classification",
        "model": model, "model_label": model_label,
        "params": params or {"optimizer": "adam", "lr": 0.001, "seed": seed},
        "name": name, "group": group, "note": "来源", "target": "target",
        "test_size": 0.2, "val_split": 0.2,
        "random_state": seed, "created_at": "2026-09-13 10:00:00",
    }
    (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({
        "metrics": {"accuracy": 0.91, "f1": 0.89},
        "primary_metric": {"name": "accuracy", "value": 0.91},
    }), encoding="utf-8")
    return cfg


class BatchRepeatsTest(unittest.TestCase):
    def test_repeats_share_batch_and_vary_seed_only(self):
        src = "20260913-101500-aaaa01"
        _make_done_run(src)
        ids = experiments.create_batch_repeats(src, count=3)
        self.assertEqual(len(ids), 3)
        cfg0 = json.loads((runner.run_dir_of(ids[0]) / "config.json").read_text(encoding="utf-8"))
        cfg2 = json.loads((runner.run_dir_of(ids[2]) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg0["batch_id"], cfg2["batch_id"])
        self.assertEqual(cfg0["batch_kind"], "repeats")
        self.assertEqual(cfg0["repeat_index"], 1)
        self.assertEqual(cfg2["repeat_index"], 3)
        self.assertEqual(cfg0["params"]["seed"], 42)
        self.assertEqual(cfg2["params"]["seed"], 44)
        self.assertEqual(cfg0["random_state"], 42)
        self.assertEqual(cfg2["random_state"], 44)
        self.assertIn("重复-1", cfg0["name"])
        self.assertEqual(cfg0["group"], "improved")
        self.assertEqual(len(set(ids)), 3)

    def test_repeats_reject_bad_source_and_count(self):
        with self.assertRaises(ValueError):
            experiments.create_batch_repeats("bad-id", 3)
        src = "20260913-101500-aaaa02"
        _make_done_run(src)
        with self.assertRaises(ValueError):
            experiments.create_batch_repeats(src, count=1)


class BatchAblationTest(unittest.TestCase):
    def test_each_ablation_changes_exactly_one_param(self):
        src = "20260913-101500-bbbb01"
        _make_done_run(src, params={"dropout": 0.3, "lr": 0.001, "seed": 7})
        ids = experiments.create_batch_ablation(src, {"dropout": [0.0, 0.6], "lr": [0.01]})
        self.assertEqual(len(ids), 3)
        cfg0 = json.loads((runner.run_dir_of(ids[0]) / "config.json").read_text(encoding="utf-8"))
        cfg2 = json.loads((runner.run_dir_of(ids[2]) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg0["batch_id"], cfg2["batch_id"])
        self.assertEqual(cfg0["batch_kind"], "ablation")
        self.assertEqual(cfg0["params"]["dropout"], 0.0)
        self.assertEqual(cfg0["params"]["lr"], 0.001)
        self.assertEqual(cfg2["params"]["dropout"], 0.3)
        self.assertEqual(cfg2["params"]["lr"], 0.01)
        self.assertEqual(cfg0["params"]["seed"], 7)
        self.assertEqual(cfg2["params"]["seed"], 7)
        self.assertEqual(cfg0["random_state"], 42)
        self.assertEqual(cfg2["random_state"], 42)
        self.assertEqual(cfg0["group"], "ablation")
        self.assertIn("丢弃率", cfg0["name"])
        self.assertIn("0", cfg0["name"])

    def test_ablation_rejects_empty_overrides(self):
        src = "20260913-101500-bbbb02"
        _make_done_run(src)
        with self.assertRaises(ValueError):
            experiments.create_batch_ablation(src, {})


if __name__ == "__main__":
    unittest.main()
