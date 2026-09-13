# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import os
import sys
import unittest
from unittest import mock

from app import desktop


class ResolveStartupModeTest(unittest.TestCase):
    def test_development_defaults_to_browser(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(sys, "frozen", None, create=True):
            self.assertEqual(desktop.resolve_startup_mode([]), "browser")

    def test_browser_flag_wins(self):
        with mock.patch.dict(os.environ, {"THESISFORGE_MODE": "desktop"}):
            self.assertEqual(desktop.resolve_startup_mode(["--browser"]), "browser")

    def test_no_window_flags_map_to_headless(self):
        for flag in ("--no-window", "--no-browser", "--headless"):
            with self.subTest(flag=flag):
                self.assertEqual(desktop.resolve_startup_mode([flag]), "headless")

    def test_desktop_env_and_frozen_enter_desktop_mode(self):
        with mock.patch.dict(os.environ, {"THESISFORGE_MODE": "desktop"}):
            self.assertEqual(desktop.resolve_startup_mode([]), "desktop")
        with mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(desktop.resolve_startup_mode([]), "desktop")


class DesktopFallbackTest(unittest.TestCase):
    def test_missing_webview_falls_back_to_browser(self):
        info = {"server": None, "thread": None, "port": 8765, "url": "http://127.0.0.1:8765/"}
        with mock.patch("app.desktop.webview", None), \
             mock.patch("app.desktop.start_server_thread", return_value=info), \
             mock.patch("app.desktop.run_browser", return_value=0) as rb:
            self.assertEqual(desktop.run_desktop(), 0)
        rb.assert_called_once_with(port=8765, info=info)

    def test_window_start_error_falls_back_to_browser(self):
        info = {"server": None, "thread": None, "port": 8766, "url": "http://127.0.0.1:8766/"}

        class FakeWebview:
            def create_window(self, *a, **k):
                pass

            def start(self, *a, **k):
                raise RuntimeError("WebView2 not found")

            def destroy(self):
                raise RuntimeError("already destroyed")

        with mock.patch("app.desktop.webview", FakeWebview()), \
             mock.patch("app.desktop.start_server_thread", return_value=info), \
             mock.patch("app.desktop.run_browser", return_value=0) as rb:
            self.assertEqual(desktop.run_desktop(), 0)
        rb.assert_called_once_with(port=8766, info=info)

    def test_bridge_opens_system_browser(self):
        with mock.patch("app.desktop.webbrowser.open") as opened:
            bridge = desktop.DesktopBridge("http://127.0.0.1:8765/")
            self.assertEqual(bridge.open_external_browser(), {"ok": True})
            opened.assert_called_once_with("http://127.0.0.1:8765/")


if __name__ == "__main__":
    unittest.main()
