# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import os
import unittest
from unittest import mock

from app import config


class DataDirIsolationTest(unittest.TestCase):
    def test_data_dir_points_to_test_tmp(self):
        self.assertEqual(config.DATA_DIR, _isolate.TEST_TMP)

    def test_ensure_dirs_creates_full_tree(self):
        config.ensure_dirs()
        for d in (config.DATA_DIR, config.DATASETS_DIR, config.RUNS_DIR,
                  config.EXPORTS_DIR, config.UPLOADS_DIR):
            self.assertTrue(d.is_dir(), d)


class ApiKeyTest(unittest.TestCase):
    def test_env_beats_stored_key(self):
        cfg = {"llm_api_key": "stored-secret", "llm_api_key_env": "LLM_API_KEY"}
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "env-secret"}, clear=False):
            self.assertEqual(config.resolve_api_key(cfg), "env-secret")

    def test_falls_back_to_stored_key(self):
        cfg = {"llm_api_key": "stored-secret", "llm_api_key_env": "LLM_API_KEY"}
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.resolve_api_key(cfg), "stored-secret")

    def test_custom_env_name(self):
        cfg = {"llm_api_key": "stored", "llm_api_key_env": "ZHIPU_KEY"}
        with mock.patch.dict(os.environ, {"ZHIPU_KEY": "custom-env"}, clear=False):
            self.assertEqual(config.resolve_api_key(cfg), "custom-env")

    def test_no_key_is_empty(self):
        cfg = {"llm_api_key": "", "llm_api_key_env": "LLM_API_KEY"}
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.resolve_api_key(cfg), "")


class SaveConfigTest(unittest.TestCase):
    def test_ignores_unknown_keys(self):
        config.save_runtime_config({"llm_model": "glm-4-flash", "totally_unknown_key": "x"})
        loaded = config.load_runtime_config()
        self.assertEqual(loaded["llm_model"], "glm-4-flash")
        self.assertNotIn("totally_unknown_key", loaded)

    def test_roundtrip_persists(self):
        config.save_runtime_config({"llm_temperature": 0.9})
        self.assertEqual(config.load_runtime_config()["llm_temperature"], 0.9)


if __name__ == "__main__":
    unittest.main()
