# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import json
import time
import unittest
import uuid
from unittest import mock

from fastapi.testclient import TestClient

from app import datasets_hub, runner
from app.main import app


def _headers():
    return {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}


def _make_fake_dataset():
    ds_id = "apitest-ds"
    d = datasets_hub.dataset_dir(ds_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({
        "id": ds_id, "name": "API测试数据", "type": "tabular",
        "target": "target", "columns": ["x", "target"],
        "task": "tabular_classification", "n_rows": 20,
    }, ensure_ascii=False), encoding="utf-8")
    return ds_id


def _make_run(rid, *, name="", group="baseline", note="", state="done",
              metrics=None, primary=None, model="random_forest",
              model_label="随机森林"):
    d = runner.run_dir_of(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({
        "dataset_id": "apitest-ds", "dataset_name": "API测试数据",
        "task": "tabular_classification", "model": model,
        "model_label": model_label, "params": {"n_estimators": 100},
        "name": name, "group": group, "note": note,
        "created_at": "2026-09-13 10:00:00",
    }, ensure_ascii=False), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    if metrics is not None:
        summary = {"metrics": metrics}
        if primary is not None:
            summary["primary_metric"] = primary
        (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


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


class ExperimentApiTest(unittest.TestCase):
    def setUp(self):
        _make_fake_dataset()
        self.client = TestClient(app)
        self.headers = _headers()

    def test_create_run_persists_name_group_note(self):
        captured = {}

        def fake_create(config, script):
            rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            captured["config"] = config
            captured["script"] = script
            d = runner.run_dir_of(rid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False),
                                           encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}),
                                           encoding="utf-8")
            return rid

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = self.client.post("/api/runs", json={
                "dataset_id": "apitest-ds", "task": "tabular_classification",
                "model": "random_forest", "params": {"n_estimators": 100},
                "name": "基线实验", "group": "baseline", "note": "第一组",
            }, headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(captured["config"]["name"], "基线实验")
        self.assertEqual(captured["config"]["group"], "baseline")
        self.assertEqual(captured["config"]["note"], "第一组")

    def test_create_run_dispatches_by_engine(self):
        scripts = []

        def fake_create(config, script):
            scripts.append(script)
            rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            d = runner.run_dir_of(rid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False),
                                           encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}),
                                           encoding="utf-8")
            return rid

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            for model in ("logistic_regression", "mlp"):
                r = self.client.post("/api/runs", json={
                    "dataset_id": "apitest-ds", "task": "tabular_classification",
                    "model": model, "params": {},
                }, headers=self.headers)
                self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(scripts, ["train_sklearn.py", "train_torch.py"])

    def test_create_time_series_run_on_tabular(self):
        captured = {}

        def fake_create(config, script):
            captured["config"] = config
            captured["script"] = script
            rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            d = runner.run_dir_of(rid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False),
                                           encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}),
                                           encoding="utf-8")
            return rid

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = self.client.post("/api/runs", json={
                "dataset_id": "apitest-ds", "task": "time_series_forecasting",
                "model": "lstm", "params": {"lookback": 4, "horizon": 1},
                "name": "时序基线", "group": "baseline",
            }, headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(captured["config"]["task"], "time_series_forecasting")
        self.assertEqual(captured["script"], "train_torch.py")

    def test_patch_meta_updates_config(self):
        rid = "20260913-101500-ca1a1a"
        _make_run(rid, name="旧名字", group="baseline", note="旧备注")
        r = self.client.patch(f"/api/runs/{rid}/meta", json={
            "name": "新名字", "group": "improved", "note": "新备注",
        }, headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["meta"],
                         {"name": "新名字", "group": "improved", "note": "新备注",
                          "batch_id": "", "batch_kind": "", "repeat_index": None})
        cfg = json.loads((runner.run_dir_of(rid) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["group"], "improved")

    def test_patch_meta_rejects_unknown_group(self):
        rid = "20260913-101500-ca1a1b"
        _make_run(rid, group="baseline")
        r = self.client.patch(f"/api/runs/{rid}/meta", json={"group": "weird"},
                              headers=self.headers)
        self.assertEqual(r.status_code, 400)

    def test_compare_returns_sorted_metric_table(self):
        for rid, name, group, acc in [
            ("20260913-101500-ca1a1c", "基线", "baseline", 0.85),
            ("20260913-101500-ca1a1d", "改进", "improved", 0.91),
            ("20260913-101500-ca1a1e", "消融", "ablation", 0.87),
        ]:
            _make_run(rid, name=name, group=group,
                      metrics={"accuracy": acc, "f1": round(acc - 0.02, 2)},
                      primary={"name": "accuracy", "value": acc})
        r = self.client.post("/api/experiments/compare", json={
            "run_ids": ["20260913-101500-ca1a1e", "20260913-101500-ca1a1c",
                        "20260913-101500-ca1a1d"],
        }, headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["count"], 3)
        self.assertEqual([c["group"] for c in body["columns"]],
                         ["baseline", "improved", "ablation"])
        self.assertTrue(body["metric_rows"][0]["is_primary"])
        self.assertEqual(body["metric_rows"][0]["values"], [0.85, 0.91, 0.87])
        self.assertIn("基线", body["csv"])

    def test_batch_repeats_endpoint_forks_source(self):
        rid = "20260913-101500-ca1a01"
        _make_run(rid, name="基线-MLP", group="baseline", model="mlp",
                  model_label="MLP", metrics={"accuracy": 0.9},
                  primary={"name": "accuracy", "value": 0.9})
        cfg_path = runner.run_dir_of(rid) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["params"] = {"seed": 42}
        cfg["random_state"] = 42
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        made = []

        def fake_create(config, script):
            new = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            d = runner.run_dir_of(new)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False),
                                           encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}),
                                           encoding="utf-8")
            made.append((config, script))
            return new

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = self.client.post("/api/experiments/repeats",
                                 json={"run_id": rid, "count": 2}, headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(made), 2)
        self.assertEqual(made[0][1], "train_torch.py")
        self.assertEqual(made[0][0]["batch_kind"], "repeats")
        self.assertEqual(made[0][0]["params"]["seed"], 42)
        self.assertEqual(made[1][0]["params"]["seed"], 43)

    def test_batch_ablation_endpoint_forks_source(self):
        rid = "20260913-101500-ca1a02"
        _make_run(rid, name="基线-MLP", group="baseline", model="mlp",
                  model_label="MLP", metrics={"accuracy": 0.9},
                  primary={"name": "accuracy", "value": 0.9})
        cfg_path = runner.run_dir_of(rid) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["params"] = {"dropout": 0.2}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        made = []

        def fake_create(config, script):
            new = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            d = runner.run_dir_of(new)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(config, ensure_ascii=False),
                                           encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "running"}),
                                           encoding="utf-8")
            made.append((config, script))
            return new

        with mock.patch.object(runner, "create_run", side_effect=fake_create):
            r = self.client.post("/api/experiments/ablation",
                                 json={"run_id": rid, "overrides": {"dropout": [0.5]}},
                                 headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(made[0][0]["batch_kind"], "ablation")
        self.assertEqual(made[0][0]["params"]["dropout"], 0.5)
        self.assertEqual(made[0][0]["group"], "ablation")


if __name__ == "__main__":
    unittest.main()
