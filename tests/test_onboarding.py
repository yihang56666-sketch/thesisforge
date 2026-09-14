# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import json

from app import onboarding


def _tmp_project(tmp_path, monkeypatch):
    p = tmp_path / "project.json"
    monkeypatch.setattr(onboarding, "PROJECT_FILE", p)
    return p


def test_default_progress(tmp_path, monkeypatch):
    _tmp_project(tmp_path, monkeypatch)
    data = onboarding.load_progress()
    assert data["version"] == 1
    assert data["active_step"] == 1
    assert data["steps"]["1"]["state"] == "todo"
    assert set(data["steps"]) == {str(i) for i in range(1, 11)}


def test_save_and_reload(tmp_path, monkeypatch):
    p = _tmp_project(tmp_path, monkeypatch)
    data = onboarding.load_progress()
    data["project"]["title"] = "神经网络优化研究"
    data["steps"]["4"]["state"] = "done"
    data["steps"]["4"]["completed"] = True
    data["active_step"] = 5
    onboarding.save_progress(data)
    reloaded = onboarding.load_progress()
    assert reloaded["project"]["title"] == "神经网络优化研究"
    assert reloaded["steps"]["4"]["state"] == "done"
    assert reloaded["active_step"] == 5


def test_validate_known_steps(tmp_path, monkeypatch):
    _tmp_project(tmp_path, monkeypatch)
    ok = onboarding.update_step(2, "done")
    assert ok is True
    bad = onboarding.update_step("99", "done")
    assert bad is False
    bad_state = onboarding.update_step(2, "weird")
    assert bad_state is False


def test_project_completion_detection(tmp_path, monkeypatch):
    p = _tmp_project(tmp_path, monkeypatch)
    p.write_text(json.dumps({
        "version": 1,
        "project": {"title": "t", "direction": "image"},
        "steps": {str(i): {"state": "todo", "completed": False, "saved_at": None} for i in range(1, 11)},
        "active_step": 1,
    }), encoding="utf-8")
    assert onboarding.looks_completed_manually(onboarding.load_progress()) is False
    data = onboarding.load_progress()
    for i in range(1, 11):
        data["steps"][str(i)]["state"] = "done"
        data["steps"][str(i)]["completed"] = True
    onboarding.save_progress(data)
    assert onboarding.looks_completed_manually(onboarding.load_progress()) is True


def test_old_project_gets_recommended_templates(tmp_path, monkeypatch):
    p = _tmp_project(tmp_path, monkeypatch)
    p.write_text(json.dumps({
        "version": 1,
        "project": {"title": "旧项目"},
        "steps": {str(i): {"state": "todo", "completed": False, "saved_at": None} for i in range(1, 11)},
        "active_step": 3,
    }), encoding="utf-8")
    data = onboarding.load_progress()
    assert data["steps"]["3"]["template"]["missing"] == "impute"
    assert data["steps"]["4"]["template"]["task"] == "tabular_classification"
    assert data["steps"]["5"]["template"]["optimizer"] == "adam"
    assert data["steps"]["6"]["template"] is None


def test_template_persists_and_cleans_unknown_fields(tmp_path, monkeypatch):
    p = _tmp_project(tmp_path, monkeypatch)
    data = onboarding.load_progress()
    data["steps"]["3"]["template"] = {
        "missing": "drop", "impute": "median", "scale": "robust",
        "encode": "label", "augment": "none", "split_first": False,
        "test_size": 0.25, "val_split": 0.15, "seed": 7,
        "hack": "ignored",
    }
    data["steps"]["4"]["template"] = {"task": "image_classification", "arch": "cnn", "params": {"depth": 2}, "hack": "x"}
    onboarding.save_progress(data)
    reloaded = onboarding.load_progress()
    step3 = reloaded["steps"]["3"]["template"]
    assert step3["missing"] == "drop"
    assert step3["scale"] == "robust"
    assert step3["test_size"] == 0.25
    assert "hack" not in step3
    step4 = reloaded["steps"]["4"]["template"]
    assert step4["task"] == "image_classification"
    assert step4["arch"] == "cnn"
    assert step4["params"]["depth"] == 2
    assert "hack" not in step4


def test_wizard_api_roundtrip(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setattr(onboarding, "PROJECT_FILE", tmp_path / "project.json")
    headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
    client = TestClient(main.app)
    r = client.get("/api/wizard", headers=headers)
    assert r.status_code == 200
    assert r.json()["active_step"] == 1
    body = {
        "project": {"title": "测试题目"},
        "steps": {"1": {"state": "done", "completed": True, "saved_at": None}},
        "active_step": 2,
    }
    r2 = client.put("/api/wizard", json=body, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["steps"]["1"]["state"] == "done"
