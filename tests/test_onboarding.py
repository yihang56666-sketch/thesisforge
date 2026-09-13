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
