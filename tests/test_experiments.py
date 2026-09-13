# -*- coding: utf-8 -*-
import _isolate  # noqa: F401   必须先于 app.* 导入

import json
import unittest

from app import experiments, runner


def make_run(run_id, *, state="done", group="baseline", name="", note="",
             metrics=None, primary=None, model="random_forest", model_label="随机森林"):
    d = runner.run_dir_of(run_id)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dataset_id": "iris", "dataset_name": "鸢尾花", "task": "tabular_classification",
        "model": model, "model_label": model_label, "params": {"n_estimators": 100},
        "name": name, "group": group, "note": note, "created_at": "2026-09-13 10:00:00",
    }
    (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    if metrics is not None:
        summary = {"metrics": metrics}
        if primary is not None:
            summary["primary_metric"] = primary
        (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


class GroupOrderTest(unittest.TestCase):
    def test_rank_follows_baseline_improved_ablation_custom(self):
        order = [runner.GROUP_BASELINE, runner.GROUP_IMPROVED, runner.GROUP_ABLATION, runner.GROUP_CUSTOM]
        self.assertEqual([experiments.group_rank(g) for g in order], [0, 1, 2, 3])
        self.assertEqual(experiments.group_rank("unknown"), 4)

    def test_label_falls_back_to_chinese(self):
        self.assertEqual(experiments.group_label("ablation"), "消融")
        self.assertEqual(experiments.group_label(""), "基线")

    def test_sort_is_group_then_metric_then_name(self):
        runs = [
            {"run_id": "b", "config": {"group": "ablation", "name": "B"},
             "summary": {"primary_metric": {"name": "accuracy", "value": 0.6}}},
            {"run_id": "a", "config": {"group": "baseline", "name": "A"},
             "summary": {"primary_metric": {"name": "accuracy", "value": 0.9}}},
            {"run_id": "c", "config": {"group": "improved", "name": "C"},
             "summary": {"primary_metric": {"name": "accuracy", "value": 0.8}}},
        ]
        out = experiments.sort_runs(runs, by_metric=True)
        self.assertEqual([r["run_id"] for r in out], ["a", "c", "b"])


class BuildComparisonTest(unittest.TestCase):
    def setUp(self):
        make_run("20260913-101500-000001", name="基线-随机森林", group="baseline",
                 metrics={"accuracy": 0.85, "f1": 0.83},
                 primary={"name": "accuracy", "value": 0.85})
        make_run("20260913-101500-000002", name="改进-梯度提升树", group="improved",
                 metrics={"accuracy": 0.91, "f1": 0.9},
                 primary={"name": "accuracy", "value": 0.91})
        make_run("20260913-101500-000003", name="消融-去掉增强", group="ablation",
                 metrics={"accuracy": 0.87, "f1": 0.84},
                 primary={"name": "accuracy", "value": 0.87})
        make_run("20260913-101500-000004", state="running", group="improved",
                 metrics={"accuracy": 0.5}, primary={"name": "accuracy", "value": 0.5})

    def test_only_done_runs_and_group_order(self):
        r = experiments.build_comparison([
            "20260913-101500-000004", "20260913-101500-000003",
            "20260913-101500-000002", "20260913-101500-000001",
        ])
        self.assertEqual(r["count"], 3)
        self.assertEqual([c["run_id"] for c in r["columns"]],
                         ["20260913-101500-000001", "20260913-101500-000002",
                          "20260913-101500-000003"])
        self.assertEqual([c["group"] for c in r["columns"]],
                         ["baseline", "improved", "ablation"])

    def test_primary_metric_is_first_row_and_values_match(self):
        r = experiments.build_comparison([
            "20260913-101500-000001", "20260913-101500-000002",
            "20260913-101500-000003",
        ])
        first = r["metric_rows"][0]
        self.assertTrue(first["is_primary"])
        self.assertEqual(first["label"], "主指标：accuracy")
        self.assertEqual(first["values"], [0.85, 0.91, 0.87])
        self.assertEqual(r["metric_rows"][1]["key"], "f1")

    def test_csv_and_markdown_are_symmetric(self):
        r = experiments.build_comparison([
            "20260913-101500-000001", "20260913-101500-000002",
            "20260913-101500-000003",
        ])
        self.assertIn("基线-随机森林", r["csv"])
        self.assertIn("主指标：accuracy", r["csv"])
        self.assertTrue(r["markdown"].startswith("| 指标 |"))
        # 表头 + 分隔行 + 每个指标一行
        self.assertEqual(len(r["markdown"].splitlines()), len(r["metric_rows"]) + 2)

    def test_missing_and_invalid_ids_are_ignored(self):
        r = experiments.build_comparison(["bad-id", "20260913-101500-999999",
                                          "20260913-101500-000001"])
        self.assertEqual(r["count"], 1)
        self.assertEqual(r["columns"][0]["run_id"], "20260913-101500-000001")

    def test_bad_group_normalizes_to_baseline(self):
        make_run("20260913-101500-000005", name="脏分组", group="weird",
                 metrics={"accuracy": 0.7}, primary={"name": "accuracy", "value": 0.7})
        r = experiments.build_comparison(["20260913-101500-000005"])
        self.assertEqual(r["columns"][0]["group"], "baseline")


class RepeatAggregationTest(unittest.TestCase):
    def test_repeats_same_batch_merge_and_show_mean_std(self):
        batch = "repbatch-0001"
        for i, acc in enumerate([0.91, 0.93, 0.95], start=1):
            rid = f"20260913-101500-{i:06x}"
            make_run(rid, name=f"改进-MLP-重复-{i}", group="improved", state="done",
                     model="mlp", model_label="MLP",
                     metrics={"accuracy": acc, "f1": round(acc - 0.01, 2)},
                     primary={"name": "accuracy", "value": acc})
            cfg_path = runner.run_dir_of(rid) / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["batch_id"] = batch
            cfg["batch_kind"] = "repeats"
            cfg["repeat_index"] = i
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        r = experiments.build_comparison([f"20260913-101500-{i:06x}" for i in range(1, 4)])
        self.assertEqual(r["count"], 3)
        self.assertEqual(len(r["columns"]), 1)
        self.assertEqual(r["columns"][0]["label"], "改进-MLP-重复-1 ×3次")
        self.assertEqual(r["columns"][0]["run_ids"],
                         [f"20260913-101500-{i:06x}" for i in range(1, 4)])
        row = r["metric_rows"][0]
        self.assertTrue(row["is_primary"])
        self.assertEqual(row["values"], ["0.930±0.016"])
        self.assertEqual(row["n"], [3])
        self.assertAlmostEqual(row["std"][0], 0.01632993161856052, places=12)

    def test_partial_repeat_batch_keeps_plain_single_column(self):
        batch = "repbatch-0002"
        rid = "20260913-101500-aaaa01"
        make_run(rid, name="改进-X-重复-1", group="improved", state="done",
                 metrics={"accuracy": 0.9}, primary={"name": "accuracy", "value": 0.9})
        cfg_path = runner.run_dir_of(rid) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["batch_id"] = batch
        cfg["batch_kind"] = "repeats"
        cfg["repeat_index"] = 1
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        r = experiments.build_comparison([rid])
        self.assertEqual(len(r["columns"]), 1)
        self.assertEqual(r["columns"][0]["label"], "改进-X-重复-1")
        self.assertEqual(r["metric_rows"][0]["values"], [0.9])


if __name__ == "__main__":
    unittest.main()
