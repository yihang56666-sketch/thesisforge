# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import sys
import unittest
from pathlib import Path
from unittest import mock

from app import runner


class RunDirValidationTest(unittest.TestCase):
    def test_valid_id_roundtrip(self):
        rid = "20260913-101500-a1b2c3"
        self.assertEqual(runner.run_dir_of(rid), runner.RUNS_DIR / rid)

    def test_rejects_traversal_and_garbage(self):
        for bad in ["", "abc", "../x", "20260913-101500-a1b2c3/../x",
                    "20260913-101500-a1b2c3/escape", "..\\evil"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    runner.run_dir_of(bad)


class WorkerCmdTest(unittest.TestCase):
    def test_dev_mode_uses_system_python(self):
        cmd = runner.worker_cmd("train_sklearn.py", Path("app/train_sklearn.py"), Path("run"))
        self.assertEqual(cmd[0], sys.executable)
        self.assertIn("--run-dir", cmd)
        self.assertNotIn("--tf-worker", cmd)

    def test_frozen_mode_dispatches_to_entrypoint(self):
        with mock.patch("sys.frozen", True, create=True):
            cmd = runner.worker_cmd("train_torch.py", Path("app/train_torch.py"), Path("run"))
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], "--tf-worker")
        self.assertEqual(cmd[2], "train_torch.py")
        self.assertIn("--run-dir", cmd)


if __name__ == "__main__":
    unittest.main()
