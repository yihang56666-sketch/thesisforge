# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import math
import unittest

from app import catalog


class CatalogShapeTest(unittest.TestCase):
    def test_all_tasks_have_models(self):
        self.assertIn("tabular_classification", catalog.CATALOG)
        self.assertIn("tabular_regression", catalog.CATALOG)
        self.assertIn("text_classification", catalog.CATALOG)
        self.assertIn("image_classification", catalog.CATALOG)
        for task, t in catalog.CATALOG.items():
            self.assertTrue(t["models"], task)

    def test_unknown_spec_is_none(self):
        self.assertIsNone(catalog.get_model_spec("tabular_classification", "no_such_model"))
        self.assertIsNone(catalog.get_model_spec("no_such_task", "logistic_regression"))

    def test_flat_list_covers_catalog(self):
        flat = catalog.all_models_flat()
        total = sum(len(t["models"]) for t in catalog.CATALOG.values())
        self.assertEqual(len(flat), total)


class SanitizeParamsTest(unittest.TestCase):
    def setUp(self):
        self.spec = catalog.get_model_spec("tabular_classification", "logistic_regression")

    def test_clamps_float_to_schema_bounds(self):
        out = catalog.sanitize_params(self.spec, {"C": 1e9, "max_iter": 50})
        self.assertEqual(out["C"], 100.0)
        self.assertEqual(out["max_iter"], 100)

    def test_rejects_out_of_schema_keys(self):
        out = catalog.sanitize_params(self.spec, {"C": 0.5, "model_choice": "evil"})
        self.assertNotIn("model_choice", out)

    def test_bool_parsing(self):
        spec = catalog.get_model_spec("image_classification", "resnet18")
        self.assertTrue(catalog.sanitize_params(spec, {"pretrained": "yes"})["pretrained"])
        self.assertFalse(catalog.sanitize_params(spec, {"pretrained": "0"})["pretrained"])

    def test_inf_and_nan_fall_back_to_default(self):
        spec = catalog.get_model_spec("image_classification", "cnn")
        self.assertEqual(catalog.sanitize_params(spec, {"lr": math.inf})["lr"], spec["params"]["lr"]["default"])
        self.assertEqual(catalog.sanitize_params(spec, {"lr": math.nan})["lr"], spec["params"]["lr"]["default"])

    def test_choice_falls_back_on_bad_value(self):
        spec = catalog.get_model_spec("tabular_classification", "svm")
        out = catalog.sanitize_params(spec, {"kernel": "evil"})
        if "kernel" in spec["params"]:
            self.assertEqual(out["kernel"], spec["params"]["kernel"]["default"])


if __name__ == "__main__":
    unittest.main()
