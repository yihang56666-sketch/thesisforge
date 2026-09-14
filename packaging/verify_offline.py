# -*- coding: utf-8 -*-
"""Release smoke test for the offline package without requiring PyTorch.

Start the packaged app in headless mode, then run:
    THESISFORGE_BASE=http://127.0.0.1:8765 python packaging/verify_offline.py
"""
from __future__ import annotations

import csv
import io
import os
import random
import time
from typing import Any

import httpx

BASE = os.environ.get("THESISFORGE_BASE", "http://127.0.0.1:8765").rstrip("/")
client = httpx.Client(timeout=120)


def json_response(r: httpx.Response) -> dict[str, Any]:
    r.raise_for_status()
    return r.json()


def wait_run(run_id: str, max_seconds: int = 180) -> dict[str, Any]:
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        detail = json_response(client.get(f"{BASE}/api/runs/{run_id}"))
        if detail["state"] != "running":
            return detail
        time.sleep(1)
    raise TimeoutError(f"run {run_id} did not finish in {max_seconds}s")


def main() -> None:
    health = json_response(client.get(f"{BASE}/api/health"))
    assert health.get("ok") is True
    print(f"health: ok version={health.get('version')}")

    dataset = json_response(client.post(f"{BASE}/api/datasets/builtin", json={"name": "iris"}))["dataset"]
    iris_run = json_response(client.post(f"{BASE}/api/runs", json={
        "dataset_id": dataset["id"], "task": "tabular_classification", "model": "random_forest",
        "params": {"n_estimators": 30}, "test_size": 0.2, "random_state": 42,
        "name": "离线包-基线", "group": "baseline", "note": "release smoke test",
    }))["run_id"]
    detail = wait_run(iris_run)
    assert detail["state"] == "done", detail.get("error") or detail
    print(f"tabular train: done primary={detail.get('summary', {}).get('primary_metric')}")

    random.seed(7)
    positives = ["剧情精彩 值得推荐", "画面出色 配乐很好", "表演生动 节奏紧凑"]
    negatives = ["剧情混乱 节奏拖沓", "演技生硬 不推荐", "亮点很少 浪费时间"]
    rows = [("赞 " + random.choice(positives), "positive") if i % 2 == 0
            else ("差 " + random.choice(negatives), "negative") for i in range(80)]
    random.shuffle(rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["text", "label"])
    writer.writerows(rows)
    text_dataset = json_response(client.post(f"{BASE}/api/datasets/import", files={
        "file": ("offline-smoke-text.csv", buffer.getvalue().encode("utf-8"), "text/csv"),
    }))["dataset"]["id"]
    text_run = json_response(client.post(f"{BASE}/api/runs", json={
        "dataset_id": text_dataset, "task": "text_classification", "model": "tfidf_logreg",
        "params": {"max_features": 1000}, "target": "label", "text_column": "text",
        "name": "离线包-文本", "group": "improved", "note": "release smoke test",
    }))["run_id"]
    detail = wait_run(text_run)
    assert detail["state"] == "done", detail.get("error") or detail
    print(f"text train: done primary={detail.get('summary', {}).get('primary_metric')}")

    compare = json_response(client.post(f"{BASE}/api/experiments/compare", json={
        "run_ids": [iris_run, text_run],
    }))
    assert compare["count"] == 2
    assert "csv" in compare and "markdown" in compare
    print(f"compare: {compare['count']} runs")

    report = json_response(client.post(f"{BASE}/api/report/generate", json={
        "title": "离线包发布验证", "run_ids": [iris_run, text_run],
        "dataset_id": dataset["id"], "author": {"name": "Release Bot"},
        "ai_draft": False,
    }))
    download = client.get(f"{BASE}/api/report/download", params={"filename": report["filename"]})
    assert download.status_code == 200 and len(download.content) > 50_000
    print(f"report: {report['filename']} ({len(download.content)} bytes)")
    print("OFFLINE SMOKE PASSED")


if __name__ == "__main__":
    main()
