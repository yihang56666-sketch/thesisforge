# -*- coding: utf-8 -*-
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ENTRY = Path(__file__).resolve().parent.parent / "packaging" / "ThesisForge.py"


def _load_entry() -> object:
    spec = importlib.util.spec_from_file_location("tf_launcher_entry", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class LauncherRuntimeArgsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_entry()

    def test_forwards_no_browser_flag(self) -> None:
        with mock.patch.object(sys, "argv", ["ThesisForge.exe", "--no-browser"]), \
             mock.patch("subprocess.call", return_value=0) as call:
            rc = self.mod._run_with_runtime(Path("runtime/python.exe"), Path("C:/pkg"))
        self.assertEqual(rc, 0)
        cmd = call.call_args[0][0]
        self.assertEqual(cmd, [str(Path("runtime/python.exe")), "-u", "-m", "app.main", "--no-browser"])
        env = call.call_args[1]["env"]
        self.assertEqual(env["THESISFORGE_NO_BROWSER"], "1")

    def test_forwards_browser_flag_without_no_browser_env(self) -> None:
        with mock.patch.object(sys, "argv", ["ThesisForge.exe", "--browser"]), \
             mock.patch("subprocess.call", return_value=0) as call:
            self.mod._run_with_runtime(Path("runtime/python.exe"), Path("C:/pkg"))
        cmd = call.call_args[0][0]
        self.assertEqual(cmd, [str(Path("runtime/python.exe")), "-u", "-m", "app.main", "--browser"])
        env = call.call_args[1]["env"]
        self.assertNotIn("THESISFORGE_NO_BROWSER", env)


class PackagingWorkerDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_entry()

    def test_semantic_worker_is_allowed(self) -> None:
        captured = []

        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch("runpy.run_module", side_effect=lambda *args, **kwargs: captured.append(list(sys.argv))), \
             mock.patch.object(sys, "argv", ["ThesisForge.exe"]):
            rc = self.mod._worker(["train_semantic.py", "--run-dir", "runs/demo"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured, [["train_semantic.py", "--run-dir", "runs/demo"]])

    def test_unknown_worker_is_rejected(self) -> None:
        rc = self.mod._worker(["not_a_worker.py"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
