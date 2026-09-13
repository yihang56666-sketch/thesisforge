# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import unittest

from fastapi.testclient import TestClient

from app.main import app


class LocalOriginGuardTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_allows_local_host(self):
        r = self.client.get("/api/health", headers={"Host": "127.0.0.1:8765"})
        self.assertEqual(r.status_code, 200)

    def test_nonlocal_host_blocked(self):
        r = self.client.get("/api/health", headers={"Host": "evil.example.com"})
        self.assertEqual(r.status_code, 403)

    def test_cross_site_post_blocked(self):
        r = self.client.post(
            "/api/config", json={"llm_model": "x"},
            headers={"Host": "127.0.0.1:8765", "Origin": "https://evil.example.com"},
        )
        self.assertEqual(r.status_code, 403)

    def test_same_origin_post_allowed(self):
        r = self.client.post(
            "/api/config", json={"llm_model": "x"},
            headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
        )
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
