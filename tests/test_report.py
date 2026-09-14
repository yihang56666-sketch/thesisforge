# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import tempfile
import unittest
from pathlib import Path

from docx import Document

from app.report import build_report
from app.main import safe_report_filename
from app.report import json_params_short


class SafeReportFilenameTest(unittest.TestCase):
    def test_keeps_word_and_chinese_chars(self):
        self.assertEqual(safe_report_filename("基于ML的 研究:报告!? v1"), "基于ML的研究报告v1")

    def test_strips_windows_reserved_characters(self):
        self.assertEqual(safe_report_filename('题目 a/b\\c*d?"<>|'), "题目abcd")

    def test_empty_falls_back(self):
        self.assertEqual(safe_report_filename("///"), "报告")
        self.assertEqual(safe_report_filename("   "), "报告")

    def test_long_title_truncated(self):
        self.assertEqual(len(safe_report_filename("测" * 60)), 40)


class JsonParamsShortTest(unittest.TestCase):
    def test_truncates_to_four_params(self):
        out = json_params_short({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        self.assertEqual(out, "a=1, b=2, c=3, d=4")

    def test_empty_returns_default(self):
        self.assertEqual(json_params_short({}), "默认")
        self.assertEqual(json_params_short(None), "默认")


class ReportResearchDepthTest(unittest.TestCase):
    def _build(self, runs, dataset_meta):
        out = Path(tempfile.mkdtemp()) / "report.docx"
        build_report(out, "基于深度学习的分类研究", {}, dataset_meta, None, runs, {})
        doc = Document(str(out))
        return [p.text for p in doc.paragraphs]

    def test_without_ai_draft_includes_research_diagnostics(self):
        runs = [
            {
                "run_id": "base", "name": "基线", "group": "baseline",
                "config": {"task": "tabular_classification", "model": "logistic_regression",
                           "model_label": "逻辑回归", "group": "baseline", "params": {}},
                "summary": {"model_label": "逻辑回归", "group": "baseline",
                            "primary_metric": {"name": "accuracy", "value": 0.80},
                            "metrics": {"accuracy": 0.80}, "eval_source": "test"},
            },
            {
                "run_id": "imp", "name": "改进", "group": "improved",
                "config": {"task": "tabular_classification", "model": "mlp",
                           "model_label": "MLP", "group": "improved", "params": {}},
                "summary": {"model_label": "MLP", "group": "improved",
                            "primary_metric": {"name": "accuracy", "value": 0.86},
                            "metrics": {"accuracy": 0.86}, "eval_source": "test"},
            },
            {
                "run_id": "abl", "name": "消融", "group": "ablation",
                "config": {"task": "tabular_classification", "model": "mlp",
                           "model_label": "MLP", "group": "ablation", "params": {}},
                "summary": {"model_label": "MLP", "group": "ablation",
                            "primary_metric": {"name": "accuracy", "value": 0.82},
                            "metrics": {"accuracy": 0.82}, "eval_source": "test"},
            },
        ]
        dataset_meta = {
            "name": "示例分类集", "desc": "用于验证报告深度。", "columns": ["x1", "x2"],
            "target": "label", "stats": {"class_counts": {"yes": 80, "no": 20}},
        }
        texts = self._build(runs, dataset_meta)

        self.assertIn("4.2  综合对比与研究性解读", texts)
        self.assertTrue(any("0.8000" in t and "0.8600" in t and "7.5%" in t for t in texts))
        self.assertTrue(any("消融" in t and "0.8200" in t and "0.8600" in t for t in texts))
        self.assertTrue(any("类别不平衡" in t and "宏平均 F1" in t for t in texts))
        self.assertTrue(any("独立测试集" in t for t in texts))

    def test_report_flags_overfitting_from_epoch_history(self):
        runs = [
            {
                "run_id": "overfit", "name": "过拟合", "group": "baseline",
                "config": {"task": "image_classification", "model": "resnet18",
                           "model_label": "ResNet18", "group": "baseline", "params": {}},
                "summary": {"model_label": "ResNet18", "group": "baseline",
                            "primary_metric": {"name": "accuracy", "value": 0.86},
                            "metrics": {"accuracy": 0.86}, "eval_source": "test",
                            "epochs": [{"train_acc": 0.99, "val_acc": 0.80}]},
            }
        ]
        dataset_meta = {"name": "示例分类集", "columns": [], "target": "label"}
        texts = self._build(runs, dataset_meta)

        self.assertTrue(any("过拟合迹象" in t and "0.9900" in t and "0.8000" in t for t in texts))


if __name__ == "__main__":
    unittest.main()
