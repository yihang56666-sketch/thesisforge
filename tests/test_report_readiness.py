# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.report_readiness import report_readiness


def _run(run_id="base", group="baseline", accuracy=0.80):
    return {
        "run_id": run_id,
        "name": run_id,
        "group": group,
        "config": {"task": "tabular_classification", "params": {"seed": 42}},
        "summary": {
            "primary_metric": {"name": "accuracy", "value": accuracy},
            "metrics": {"accuracy": accuracy, "macro_f1": 0.72},
            "eval_source": "test",
            "epochs": [{"train_acc": accuracy + 0.01, "val_acc": accuracy}],
        },
    }


class ReportReadinessTest(unittest.TestCase):
    def _by_label(self, checks):
        return {c["label"]: c for c in checks}

    def test_complete_project_is_ready_with_suggestions_only(self):
        author = {"name": "苏一航", "school": "XX大学", "student_id": "23010001", "advisor": "李老师"}
        dataset = {"name": "示例分类集", "n_rows": 500, "quality_warnings": []}
        result = report_readiness("基于深度学习的水面障碍物检测研究", author, dataset, [
            _run("base", "baseline"), _run("imp", "improved"), _run("abl", "ablation"),
            _run("repeat-1", "improved"), _run("repeat-2", "improved"), _run("repeat-3", "improved"),
        ])
        checks = self._by_label(result["checks"])

        self.assertFalse(result["blocking"])
        self.assertTrue(all(c["status"] == "ok" for c in checks.values()))

    def test_missing_title_author_dataset_and_runs_block_delivery(self):
        result = report_readiness("基于机器学习的毕业设计研究", {}, None, [])
        checks = self._by_label(result["checks"])

        self.assertTrue(result["blocking"])
        self.assertEqual(checks["论文题目"]["status"], "fail")
        self.assertEqual(checks["作者信息"]["status"], "fail")
        self.assertEqual(checks["数据集"]["status"], "fail")
        self.assertEqual(checks["实验完成"]["status"], "fail")

    def test_missing_baseline_improved_ablation_and_repeats_warns(self):
        result = report_readiness("基于深度学习的可靠分类研究", {"name": "苏一航"}, {"name": "示例"}, [_run()])
        checks = self._by_label(result["checks"])

        self.assertEqual(checks["实验设计"]["status"], "warn")
        self.assertEqual(checks["重复实验"]["status"], "warn")
        self.assertIn("基线", checks["实验设计"]["message"])
        self.assertIn("改进", checks["实验设计"]["message"])
        self.assertIn("消融", checks["实验设计"]["message"])

    def test_dataset_quality_and_small_sample_are_reported(self):
        result = report_readiness("基于深度学习的可靠分类研究", {"name": "苏一航"}, {
            "name": "小样本集", "n_rows": 80,
            "quality_warnings": ["存在 3 个缺失值，需在预处理阶段明确填充或删除策略。"],
        }, [_run()])
        checks = self._by_label(result["checks"])

        self.assertEqual(checks["数据集"]["status"], "warn")
        self.assertIn("80 条", checks["数据集"]["message"])
        self.assertIn("缺失值", checks["数据集"]["message"])

    def test_missing_test_metrics_and_epoch_history_warn(self):
        run = _run()
        run["summary"]["eval_source"] = "val"
        run["summary"]["epochs"] = []
        result = report_readiness("基于深度学习的可靠分类研究", {"name": "苏一航"}, {"name": "示例"}, [run])
        checks = self._by_label(result["checks"])

        self.assertEqual(checks["最终评估"]["status"], "warn")
        self.assertEqual(checks["训练曲线"]["status"], "warn")


class ReportReadinessApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_readiness_endpoint_returns_blocking_checks(self):
        r = self.client.post("/api/report/readiness", json={
            "title": "基于机器学习的毕业设计研究", "run_ids": [], "dataset_id": None, "author": {},
        }, headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"})

        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["blocking"])
        labels = {c["label"] for c in body["checks"]}
        self.assertIn("论文题目", labels)
        self.assertIn("数据集", labels)
        self.assertIn("实验完成", labels)


if __name__ == "__main__":
    unittest.main()
