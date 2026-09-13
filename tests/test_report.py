# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import unittest

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


if __name__ == "__main__":
    unittest.main()
