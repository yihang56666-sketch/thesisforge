# -*- coding: utf-8 -*-
"""端到端回归测试：需要服务器已在 127.0.0.1:8765 运行。

覆盖三条训练路径（表格 / 文本 / 图像）、实验元数据与对比导出、报告生成。
运行：python test_e2e.py
"""
import csv
import io
import json
import random
import time
import zipfile

import httpx

BASE = "http://127.0.0.1:8765"
c = httpx.Client(timeout=180)


def show(title, r):
    print(f"--- {title} [{r.status_code}]")
    try:
        print(json.dumps(r.json(), ensure_ascii=False)[:300])
        return r.json()
    except Exception:
        print(r.text[:300])
        return {}


def wait_run(rid, max_s=420):
    t0 = time.time()
    while time.time() - t0 < max_s:
        d = c.get(f"{BASE}/api/runs/{rid}").json()
        if d["state"] != "running":
            return d
        time.sleep(2)
    return {"state": "timeout"}


def main():
    health = c.get(f"{BASE}/api/health").json()
    print("服务器:", health)

    # 1. 表格路径：载入鸢尾花并训练随机森林
    r = c.post(f"{BASE}/api/datasets/builtin", json={"name": "iris"})
    ds = r.json()["dataset"]
    print("数据集:", ds["id"], ds["n_rows"], "行")
    r = c.post(f"{BASE}/api/runs", json={
        "dataset_id": ds["id"], "task": "tabular_classification", "model": "random_forest",
        "params": {"n_estimators": 100}, "test_size": 0.2, "random_state": 42})
    iris_run = r.json()["run_id"]
    d = wait_run(iris_run)
    s = d.get("summary") or {}
    print(f"表格训练 [{d['state']}]:", s.get("primary_metric"))
    assert d["state"] == "done", c.get(f"{BASE}/api/runs/{iris_run}/log").json()["log"][-800:]

    # 2. 文本路径：合成电影评论 CSV → TF-IDF + 逻辑回归
    random.seed(7)
    pos = ["这部电影太好看了 剧情精彩", "非常棒的作品 值得推荐", "画面精美 配乐动人", "剧情紧凑 十分精彩", "演员表现出彩 故事感人"]
    neg = ["太难看了 浪费时间", "演技尴尬 无亮点", "剧情混乱 不推荐", "烂片一部 后悔", "节奏拖沓 乏味"]
    rows = []
    for i in range(120):
        rows.append((random.choice(pos) + " 赞", "positive") if i % 2 == 0 else (random.choice(neg) + " 差", "negative"))
    random.shuffle(rows)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["text", "label"])
    w.writerows(rows)
    r = c.post(f"{BASE}/api/datasets/import", files={"file": ("e2e文本测试.csv", out.getvalue().encode("utf-8"), "text/csv")})
    text_ds = r.json()["dataset"]["id"]
    r = c.post(f"{BASE}/api/runs", json={
        "dataset_id": text_ds, "task": "text_classification", "model": "tfidf_logreg",
        "params": {"max_features": 5000}, "target": "label", "text_column": "text"})
    text_run = r.json()["run_id"]
    d = wait_run(text_run)
    s = d.get("summary") or {}
    print(f"文本训练 [{d['state']}]:", s.get("primary_metric"))
    assert d["state"] == "done", c.get(f"{BASE}/api/runs/{text_run}/log").json()["log"][-800:]

    # 3. 图像路径：合成 3 类几何图形 zip → CNN
    from PIL import Image, ImageDraw

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls, color in [("circle", (200, 60, 60)), ("square", (60, 160, 90)), ("triangle", (70, 90, 220))]:
            for i in range(40):
                img = Image.new("RGB", (48, 48), (245, 245, 245))
                dr = ImageDraw.Draw(img)
                if cls == "circle":
                    dr.ellipse([8, 8, 40, 40], fill=color)
                elif cls == "square":
                    dr.rectangle([9, 9, 39, 39], fill=color)
                else:
                    dr.polygon([(24, 7), (41, 40), (7, 40)], fill=color)
                b = io.BytesIO()
                img.save(b, "PNG")
                zf.writestr(f"{cls}/img_{i:03d}.png", b.getvalue())
    r = c.post(f"{BASE}/api/datasets/import", files={"file": ("e2e图像测试.zip", zip_buf.getvalue(), "application/zip")})
    img_ds = r.json()["dataset"]["id"]
    r = c.post(f"{BASE}/api/runs", json={
        "dataset_id": img_ds, "task": "image_classification", "model": "cnn",
        "params": {"epochs": 2, "batch_size": 16, "image_size": 48}})
    img_run = r.json()["run_id"]
    d = wait_run(img_run)
    s = d.get("summary") or {}
    print(f"图像训练 [{d['state']}]:", s.get("primary_metric"), "device:", s.get("device"))
    assert d["state"] == "done", c.get(f"{BASE}/api/runs/{img_run}/log").json()["log"][-1000:]

    # 4. 实验元数据与对比导出
    r = c.patch(f"{BASE}/api/runs/{iris_run}/meta", json={
        "name": "基线-随机森林", "group": "baseline", "note": "E2E 基线"})
    meta = r.json()["meta"]
    print("实验元数据:", meta)
    assert meta["group"] == "baseline" and meta["name"] == "基线-随机森林"
    c.patch(f"{BASE}/api/runs/{text_run}/meta", json={"group": "improved"})
    c.patch(f"{BASE}/api/runs/{img_run}/meta", json={"group": "ablation"})
    cmp = c.post(f"{BASE}/api/experiments/compare", json={
        "run_ids": [iris_run, text_run, img_run]}).json()
    print("实验对比:", cmp.get("count"), "组",
          [col["group"] for col in cmp.get("columns", [])])
    assert cmp["count"] == 3
    assert [col["group"] for col in cmp["columns"]] == ["baseline", "improved", "ablation"]
    assert cmp["metric_rows"][0]["is_primary"]
    assert "基线-随机森林" in cmp["csv"]
    assert cmp["markdown"].startswith("| 指标 |")
    print("对比 CSV 首行:", cmp["csv"].splitlines()[0])

    # 5. AI 分析（未配置 Key 时走内置规则分析器）
    r = c.post(f"{BASE}/api/runs/{iris_run}/analyze").json()
    print("AI 分析来源:", r["source"], "| 字数:", len(r["text"]))

    # 6. AIGC 自检与改写
    sample = "值得注意的是，随着人工智能的不断发展，本文极大地提升了性能。综上所述，效果非常好。"
    chk = c.post(f"{BASE}/api/humanize/check", json={"text": sample}).json()
    rw = c.post(f"{BASE}/api/humanize/rewrite", json={"text": sample}).json()
    print(f"AIGC 自检: {chk['score']}({chk['level']}) -> 改写后 {rw['score_after']['score']}({rw['score_after']['level']})")

    # 7. 报告生成与下载
    r = c.post(f"{BASE}/api/report/generate", json={
        "title": "端到端测试报告", "run_ids": [iris_run], "dataset_id": ds["id"],
        "author": {"school": "测试大学"}, "ai_draft": False})
    rep = r.json()
    print("报告生成:", rep.get("filename"))
    dl = c.get(f"{BASE}/api/report/download", params={"filename": rep["filename"]})
    assert dl.status_code == 200 and len(dl.content) > 50000
    print("报告下载:", len(dl.content), "bytes")

    # 8. 清理测试数据集
    for did in (text_ds, img_ds):
        c.delete(f"{BASE}/api/datasets/{did}")
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
