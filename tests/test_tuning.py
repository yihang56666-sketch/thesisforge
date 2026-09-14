from fastapi.testclient import TestClient

from app import tuning


def test_build_search_space_supported():
    spec = tuning.build_search_space(
        "tabular_classification", "logistic_regression",
        {"n_trials": 8, "metric": "accuracy", "search_mode": "tpe"},
    )
    assert spec["n_trials"] == 8
    assert spec["metric"] == "accuracy"
    assert spec["space"]["C"]["type"] == "float"
    assert spec["space"]["C"]["min"] == 0.001


def test_objective_reports_metric():
    summary = {"primary_metric": {"name": "accuracy", "value": 0.9}, "metrics": {"accuracy": 0.91}}
    assert tuning.objective_value(summary, "accuracy") == 0.91
    assert tuning.objective_value(summary, "primary_metric") == 0.9


def test_run_search_invokes_runner(tmp_path, monkeypatch):
    calls = []

    def fake_run_trial(cfg, out_dir, script, timeout_seconds=3600):
        calls.append(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"primary_metric": {"name": "accuracy", "value": 0.8}, "metrics": {"accuracy": 0.8}}

    monkeypatch.setattr(tuning, "_run_trial", fake_run_trial)
    result = tuning.run_search(
        task="tabular_classification",
        model="logistic_regression",
        base_config={"dataset_dir": str(tmp_path)},
        n_trials=2,
        metric="accuracy",
        search_mode="tpe",
        seed=7,
    )
    assert len(calls) == 2
    assert result["best_value"] >= 0


def test_api_search(monkeypatch, tmp_path):
    from app.main import app

    def fake_run_search(*args, **kwargs):
        return {"ok": True, "best_value": 0.9, "best_params": {"C": 1.0}, "history": []}

    monkeypatch.setattr(tuning, "run_search", fake_run_search)
    from app import datasets_hub
    monkeypatch.setattr(datasets_hub, "load_meta", lambda ds_id: {"name": "Demo", "type": "tabular", "target": "y", "columns": ["x", "y"]})
    client = TestClient(app, base_url="http://127.0.0.1")
    response = client.post("/api/tuning/search", json={
        "task": "tabular_classification",
        "model": "logistic_regression",
        "dataset_id": "demo",
        "dataset_dir": str(tmp_path),
    })
    assert response.status_code == 200
    assert response.json()["best_value"] == 0.9
