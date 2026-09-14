# -*- coding: utf-8 -*-
"""水面检测简化毕设演示：通过本地服务 API 跑通深度学习二分类闭环。

流程：合成水面/非水面图像 zip -> CNN 基线 -> CNN 改进 -> 实验对比 ->
生成 CSV/Markdown 对比表与 Word 报告草稿。

重要说明：图像数据是脚本即时生成的合成演示图，不是真实水面监控画面；
结果只用于验证软件流程与深度学习训练代码，不能作为真实工程结论。

用法（服务需已启动，默认 127.0.0.1:8765）：
    python scripts/run_water_surface_demo.py
"""
from __future__ import annotations

import io
import json
import math
import random
import time
import zipfile
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
STATE_PATH = SCRIPTS / ".water_demo_state.json"
DOCS = ROOT / "docs"
BASE_URL = "http://127.0.0.1:8765"
POLL_SEC = 2.0
MAX_WAIT_SEC = 60 * 20
IMAGE_ZIP_NAME = "synthetic-water-surface-demo.zip"
IMAGE_DS_NAME = "synthetic-water-surface-demo"


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


def _gradient_row(top: tuple[int, int, int], bottom: tuple[int, int, int], y: int, size: int) -> tuple[int, int, int]:
    t = y / max(size - 1, 1)
    return tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))


def draw_water(img: Image.Image, rng: random.Random) -> None:
    """画一张水面近景：蓝绿色渐变 + 随机波纹 + 阳光反光点。"""
    size = img.size[0]
    d = ImageDraw.Draw(img, "RGBA")
    top = (135, 185, 225)
    bottom = (18, 72, 118)
    for y in range(size):
        d.line([(0, y), (size - 1, y)], fill=_gradient_row(top, bottom, y, size))
    for _ in range(rng.randint(9, 14)):
        y0 = rng.randint(4, size - 4)
        amp = rng.randint(1, 3)
        phase = rng.randint(0, 6)
        wave_len = rng.randint(4, 8)
        shade = (
            60 + rng.randint(-20, 25),
            130 + rng.randint(-25, 30),
            190 + rng.randint(-25, 35),
            rng.randint(120, 220),
        )
        pts = []
        for x in range(0, size, 2):
            yy = y0 + int(math.sin((x + phase) / wave_len) * amp)
            pts.append((x, max(0, min(size - 1, yy))))
        d.line(pts, fill=shade, width=1)
    for _ in range(rng.randint(4, 9)):
        x, y = rng.randint(0, size - 8), rng.randint(0, size - 1)
        w = rng.randint(1, 6)
        d.line([(x, y), (x + w, y)], fill=(245, 250, 230, rng.randint(90, 190)), width=1)


def draw_nonwater(img: Image.Image, rng: random.Random) -> None:
    """画一张非水面场景：草地/沙地/道路/房屋等，与水面纹理明显不同。"""
    size = img.size[0]
    d = ImageDraw.Draw(img, "RGBA")
    palettes = [
        ((185, 215, 145), (62, 118, 58), (210, 225, 170)),      # 草地
        ((235, 205, 155), (168, 124, 66), (205, 155, 95)),      # 沙地
        ((205, 198, 190), (112, 112, 118), (240, 235, 228)),    # 城镇
        ((150, 190, 95), (90, 140, 80), (180, 220, 140)),       # 农田
    ]
    top, bottom, accent = palettes[rng.randrange(len(palettes))]
    for y in range(size):
        d.line([(0, y), (size - 1, y)], fill=_gradient_row(top, bottom, y, size))
    for _ in range(rng.randint(8, 14)):
        color = (accent if rng.random() < 0.4 else bottom)
        x0, y0 = rng.randint(0, size - 8), rng.randint(0, size - 4)
        w, h = rng.randint(4, 12), rng.randint(2, 5)
        d.rectangle([x0, y0, min(size - 1, x0 + w), min(size - 1, y0 + h)],
                    fill=(color[0], color[1], color[2], rng.randint(80, 180)))
    if rng.random() < 0.7:
        y = rng.randint(size // 3, 2 * size // 3)
        color = (90, 92, 96, rng.randint(120, 200))
        d.rectangle([0, y, size - 1, min(size - 1, y + rng.randint(2, 4))], fill=color)
    if rng.random() < 0.5:
        # 简单房屋轮廓
        x0 = rng.randint(4, size - 24)
        y0 = rng.randint(4, size - 20)
        w, h = rng.randint(10, 18), rng.randint(8, 14)
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=(225, 222, 210, 200))
        d.rectangle([x0, y0 + h // 2, x0 + w, y0 + h], fill=(128, 92, 58, 200))
        d.polygon([(x0 - 2, y0), (x0 + w // 2, y0 - 5), (x0 + w + 2, y0)],
                  fill=(170, 60, 50, 220))


def make_synthetic_zip(seed: int = 20260913, per_class: int = 150) -> bytes:
    """生成 2 类 x per_class 张 64x64 合成图（水面/非水面，非真实数据）。"""
    rng = random.Random(seed)
    size = 64
    buf = io.BytesIO()

    def draw(img: Image.Image, cls: str, rng: random.Random) -> None:
        if cls == "water":
            draw_water(img, rng)
        else:
            draw_nonwater(img, rng)

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for cls in ("water", "non_water"):
            for i in range(per_class):
                img = Image.new("RGB", (size, size), (18, 22, 26))
                draw(img, cls, rng)
                pix = img.load()
                for _ in range(int(size * size * 0.30)):
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
    print(f"[water] 水面合成图像数据集已导入: {ds_id} "
          f"({meta['dataset'].get('n_images', '?')} 张, 2 类)")
    return ds_id


def create_run(dataset_id: str, model: str, params: dict, *, group: str, name: str, note: str) -> str:
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
            print(f"[water] {key} 已在后台运行，等待完成 ...")
            wait_runs([run_id])
            return run_id
    run_id = creator()
    state.setdefault("run_ids", {})[key] = run_id
    save_state(state)
    print(f"[water] 创建 {key}: {run_id}")
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

    print("[water] 1/4 准备水面/非水面合成数据集 ...")
    image_ds = ensure_image_dataset(state)

    print("[water] 2/4 训练 CNN 基线与改进模型 ...")
    baseline = ensure_run(
        state, "baseline",
        lambda: create_run(
            image_ds, "cnn",
            image_params(optimizer="sgd", lr=0.01, epochs=6, scheduler="none",
                         weight_decay=0.0, dropout=0.2, activation="relu"),
            group="baseline", name="水面检测基线-CNN-SGD",
            note="水面/非水面合成二分类；CNN 基线：SGD/lr0.01/6轮/无调度"))
    improved = ensure_run(
        state, "improved",
        lambda: create_run(
            image_ds, "cnn",
            image_params(optimizer="adamw", lr=0.001, epochs=10, scheduler="cosine",
                         weight_decay=1e-4, dropout=0.3, activation="gelu"),
            group="improved", name="水面检测改进-CNN-AdamW余弦",
            note="水面/非水面合成二分类；改进：AdamW/余弦调度/权重衰减/早停/梯度裁剪/GELU+Dropout"))

    print("[water] 3/4 生成实验对比表 ...")
    all_ids = [baseline, improved]
    cmp = api("POST", "/api/experiments/compare", json={"run_ids": all_ids})
    csv_path = DOCS / "实验对比_水面检测.csv"
    md_path = DOCS / "实验对比_水面检测.md"
    csv_path.write_text(cmp["csv"], encoding="utf-8")
    header = ("# 实验对比表（水面检测合成演示数据，非真实场景结论）\n\n"
              "> 数据：程序生成的合成水面/非水面图像 2 类 + CNN 基线与改进。\n"
              "> 复现：`python scripts/run_water_surface_demo.py`"
              "（需先启动 `python -m app.main --no-browser`）。\n\n")
    md_path.write_text(header + cmp["markdown"] or "（无已完成实验）", encoding="utf-8")
    print(f"[water] 对比表已写入 {csv_path.name} / {md_path.name} "
          f"（{cmp['count']} 个实验）")

    print("[water] 4/4 生成 Word 报告草稿 ...")
    doc = api("POST", "/api/report/generate", json={
        "title": "基于深度学习的水面检测研究（合成演示数据）",
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
        print(f"[water] Word 报告已生成并复制到 {dst}")
    else:
        print(f"[water] 报告接口返回路径但文件不存在: {doc['path']}")

    print("\n[ water ] 全部完成")
    print("实验总数: %d" % len(all_ids))


if __name__ == "__main__":
    main()
