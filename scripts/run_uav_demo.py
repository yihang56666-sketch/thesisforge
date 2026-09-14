# -*- coding: utf-8 -*-
"""无人机航拍图像分类演示：通过本地服务 API 跑通深度学习实验闭环。

流程：合成无人机航拍风格图像 zip -> CNN 基线 -> CNN 改进 -> 两项消融 ->
生成 CSV/Markdown 对比表与 Word 报告草稿。

重要说明：图像数据是脚本即时生成的合成航拍图，不是真实无人机采集数据；
结果只用于验证软件流程与深度学习训练代码，不能作为真实工程结论。

用法（服务需已启动，默认 127.0.0.1:8765）：
    python scripts/run_uav_demo.py
"""
from __future__ import annotations

import io
import json
import random
import time
import zipfile
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
STATE_PATH = SCRIPTS / ".uav_demo_state.json"
DOCS = ROOT / "docs"
BASE_URL = "http://127.0.0.1:8765"
POLL_SEC = 2.0
MAX_WAIT_SEC = 60 * 30
IMAGE_ZIP_NAME = "synthetic-uav-aerial-demo.zip"


def api(method: str, path: str, **kwargs) -> dict:
    """请求本地服务，出错时抛出带响应正文的错误。"""
    r = httpx.request(method, BASE_URL + path, timeout=120, **kwargs)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text[:500])
        except Exception:
            detail = r.text[:500]
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {detail}")
    return r.json()


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_exists(ds_id: str) -> bool:
    try:
        return bool(api("GET", f"/api/datasets/{ds_id}").get("id"))
    except Exception:
        return False


def run_detail(run_id: str) -> dict:
    return api("GET", f"/api/runs/{run_id}")


def _png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _ground_palette(cls: str, rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if cls == "road":
        return (138, 134, 128), (78, 76, 74)
    if cls == "building":
        return (214, 206, 188), (146, 132, 116)
    if cls == "vegetation":
        return (160, 206, 124), (74, 132, 72)
    if cls == "water":
        return (120, 180, 220), (28, 82, 130)
    raise ValueError(f"未知类别: {cls}")


def draw_aerial(img: Image.Image, cls: str, rng: random.Random) -> None:
    """画一张简单无人机航拍风格图：地物色块 + 少量阴影/噪声。"""
    size = img.size[0]
    d = ImageDraw.Draw(img, "RGBA")
    top, bottom = _ground_palette(cls, rng)
    for y in range(size):
        t = y / max(size - 1, 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        d.line([(0, y), (size - 1, y)], fill=color)

    if cls == "road":
        # 一条横向道路 + 少量车道线
        y0 = rng.randint(16, size - 22)
        d.rectangle([0, y0, size - 1, y0 + 8], fill=(82, 80, 78, 220))
        for x in range(4, size - 4, 10):
            d.line([(x, y0 + 4), (min(size - 1, x + 4), y0 + 4)],
                   fill=(236, 234, 220, 170), width=1)
    elif cls == "building":
        # 两三个矩形屋顶 + 投影
        for _ in range(rng.randint(2, 3)):
            x0, y0 = rng.randint(4, size - 26), rng.randint(4, size - 26)
            w, h = rng.randint(14, 24), rng.randint(14, 22)
            d.rectangle([x0 + 2, y0 + 2, min(size - 1, x0 + w + 2),
                         min(size - 1, y0 + h + 2)], fill=(58, 52, 48, 90))
            d.rectangle([x0, y0, min(size - 1, x0 + w),
                         min(size - 1, y0 + h)], fill=(222, 216, 202, 235))
    elif cls == "vegetation":
        # 树冠/灌木斑块
        for _ in range(rng.randint(7, 12)):
            x0, y0 = rng.randint(2, size - 12), rng.randint(2, size - 12)
            r = rng.randint(4, 9)
            d.ellipse([x0, y0, x0 + r, y0 + r],
                      fill=(72, 142, 68, rng.randint(160, 240)))
    else:
        # 水面/反光区域
        y0 = rng.randint(14, size - 20)
        d.rectangle([0, y0, size - 1, y0 + rng.randint(8, 14)],
                    fill=(44, 108, 162, 220))
        for _ in range(rng.randint(4, 8)):
            x, y = rng.randint(2, size - 10), rng.randint(y0, y0 + 10)
            d.line([(x, y), (x + rng.randint(2, 6), y)],
                   fill=(214, 236, 246, rng.randint(90, 190)), width=1)


def make_synthetic_zip(seed: int = 20260914, per_class: int = 90) -> bytes:
    """生成 3 类 x per_class 张 64x64 合成图（道路/建筑/植被）。"""
    rng = random.Random(seed)
    size = 64
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for cls in ("road", "building", "vegetation"):
            for i in range(per_class):
                img = Image.new("RGB", (size, size), (18, 22, 26))
                draw_aerial(img, cls, rng)
                pix = img.load()
                for _ in range(int(size * size * 0.28)):
                    x, y = rng.randrange(size), rng.randrange(size)
                    r0, g0, b0 = pix[x, y]
                    delta = rng.randint(-26, 26)
                    pix[x, y] = (
                        min(255, max(0, r0 + delta)),
                        min(255, max(0, g0 + delta)),
                        min(255, max(0, b0 + delta)),
                    )
                zf.writestr(f"{cls}/{i:03d}.png", _png_bytes(img))
    return buf.getvalue()


def ensure_image_dataset(state: dict) -> str:
    ds_id = state.get("image_dataset_id")
    if ds_id and dataset_exists(ds_id):
        return ds_id
    data = make_synthetic_zip()
    meta = api("POST", "/api/datasets/import",
               files={"file": (IMAGE_ZIP_NAME, data, "application/zip")})
    ds_id = meta["dataset"]["id"]
    state["image_dataset_id"] = ds_id
    save_state(state)
    print(f"[uav] 无人机航拍合成图像数据集已导入: {ds_id} "
          f"({meta['dataset'].get('n_images', '?')} 张, 3 类)")
    return ds_id


def create_run(dataset_id: str, model: str, params: dict, *, group: str,
               name: str, note: str) -> str:
    body = {
        "dataset_id": dataset_id,
        "task": "image_classification",
        "model": model,
        "params": params,
        "name": name,
        "group": group,
        "note": note,
        "test_size": 0.2,
        "val_split": 0.2,
        "random_state": 42,
    }
    return api("POST", "/api/runs", json=body)["run_id"]


def wait_runs(run_ids: list[str], timeout: int = MAX_WAIT_SEC) -> None:
    deadline = time.time() + timeout
    pending = set(run_ids)
    while pending and time.time() < deadline:
        time.sleep(POLL_SEC)
        for rid in list(pending):
            d = run_detail(rid)
            st = d["state"]
            if st == "done":
                pending.discard(rid)
                continue
            if st in ("failed", "cancelled"):
                err = d.get("error") or "未提供错误信息"
                tail = (d.get("log_tail") or "")[-1200:]
                raise RuntimeError(f"实验 {rid} 状态 {st}: {err}\n{tail}")
    if pending:
        raise TimeoutError(f"等待超时，仍有实验未完成: {sorted(pending)}")


def ensure_run(state: dict, key: str, creator) -> str:
    run_id = (state.get("run_ids") or {}).get(key)
    if run_id:
        try:
            d = run_detail(run_id)
        except Exception:
            d = {}
        if d.get("state") == "done":
            return run_id
        if d.get("state") == "running":
            print(f"[uav] {key} 已在后台运行，等待完成 ...")
            wait_runs([run_id])
            return run_id
    run_id = creator()
    state.setdefault("run_ids", {})[key] = run_id
    save_state(state)
    print(f"[uav] 创建 {key}: {run_id}")
    wait_runs([run_id])
    return run_id


def image_params(optimizer: str = "adamw", lr: float = 0.001, epochs: int = 10,
                 scheduler: str = "cosine", weight_decay: float = 0.0,
                 dropout: float = 0.3, activation: str = "gelu") -> dict:
    return {
        "optimizer": optimizer,
        "lr": lr,
        "batch_size": 32,
        "epochs": epochs,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "early_stop_patience": 3 if epochs > 6 else 0,
        "grad_clip": 1.0,
        "seed": 42,
        "conv_channels": "16,32,64",
        "activation": activation,
        "dropout": dropout,
        "image_size": 64,
    }


def main() -> None:
    state = load_state()
    DOCS.mkdir(exist_ok=True)

    print("[uav] 1/4 准备无人机航拍合成数据集 ...")
    image_ds = ensure_image_dataset(state)

    print("[uav] 2/4 训练 CNN 基线与改进模型 ...")
    baseline = ensure_run(
        state, "baseline",
        lambda: create_run(
            image_ds, "cnn",
            image_params(optimizer="sgd", lr=0.01, epochs=6, scheduler="none",
                         weight_decay=0.0, dropout=0.2, activation="relu"),
            group="baseline", name="无人机航拍基线-CNN-SGD",
            note="道路/建筑/植被三分类；CNN 基线：SGD/lr0.01/6轮/无调度"))
    improved = ensure_run(
        state, "improved",
        lambda: create_run(
            image_ds, "cnn",
            image_params(optimizer="adamw", lr=0.001, epochs=10, scheduler="cosine",
                         weight_decay=1e-4, dropout=0.3, activation="gelu"),
            group="improved", name="无人机航拍改进-CNN-AdamW余弦",
            note="道路/建筑/植被三分类；改进：AdamW/余弦调度/权重衰减/早停/梯度裁剪/GELU+Dropout"))

    print("[uav] 3/4 生成两项消融实验 ...")
    ablation_ids = ensure_run(
        state, "ablation",
        lambda: api("POST", "/api/experiments/ablation",
                    json={"run_id": improved,
                          "overrides": {
                              "dropout": [0.0],
                              "weight_decay": [0.0],
                          }})["run_ids"][0])

    print("[uav] 4/4 生成实验对比表和 Word 报告草稿 ...")
    all_ids = [baseline, improved, ablation_ids]
    cmp = api("POST", "/api/experiments/compare", json={"run_ids": all_ids})
    csv_path = DOCS / "实验对比_无人机航拍.csv"
    md_path = DOCS / "实验对比_无人机航拍.md"
    csv_path.write_text(cmp["csv"], encoding="utf-8")
    header = ("# 实验对比表（无人机航拍合成演示数据，非真实场景结论）\n\n"
              "> 数据：程序生成的合成无人机航拍图像 3 类 + CNN 基线/改进/消融。\n"
              "> 复现：`python scripts/run_uav_demo.py`"
              "（需先启动 `python -m app.main --no-browser`）。\n\n")
    md_path.write_text(header + cmp["markdown"] or "（无已完成实验）", encoding="utf-8")
    print(f"[uav] 对比表已写入 {csv_path.name} / {md_path.name} "
          f"（{cmp['count']} 个实验）")

    doc = api("POST", "/api/report/generate", json={
        "title": "基于深度学习的无人机航拍图像分类研究（合成演示数据）",
        "run_ids": all_ids,
        "dataset_id": image_ds,
        "author": {"name": "毕设工坊演示作者", "school": "演示学校",
                   "major": "人工智能", "student_id": "2020XXXXXX"},
        "ai_draft": False,
    })
    src = Path(doc["path"])
    if src.exists():
        dst = DOCS / ("报告草稿_" + src.name)
        dst.write_bytes(src.read_bytes())
        print(f"[uav] Word 报告已生成并复制到 {dst}")
    else:
        print(f"[uav] 报告接口返回路径但文件不存在: {doc['path']}")

    print("\n[ uav ] 全部完成")
    print("实验总数: %d" % len(all_ids))
    for rid in all_ids:
        d = run_detail(rid)
        summary = d.get("summary") or {}
        print(f"- {d.get('name')}: {summary}")


if __name__ == "__main__":
    main()
