# -*- coding: utf-8 -*-
"""正式毕设演示实验：通过本地服务 API 跑通完整实验闭环。

流程：合成医学风格图像数据集 -> 图像 CNN 基线/改进 -> 改进实验重复 3 次 ->
一键消融（dropout/weight_decay/optimizer/scheduler）-> 表格乳腺癌基线+MLP ->
生成对比 CSV/Markdown 与 Word 报告草稿。

重要说明：图像数据是脚本即时生成的合成演示图，不是真实医学病例；
结果只用于验证软件流程与训练代码，不能作为真实医学结论。

用法（服务需已启动，默认 127.0.0.1:8765）：
    python scripts/run_thesis_demo.py

脚本会复用已完成的实验，断网/中断后可重跑续接。
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
STATE_PATH = SCRIPTS / ".thesis_demo_state.json"
DOCS = ROOT / "docs"
BASE_URL = "http://127.0.0.1:8765"
POLL_SEC = 2.0
MAX_WAIT_SEC = 60 * 45
IMAGE_ZIP_NAME = "synthetic-medical-demo.zip"
IMAGE_DS_NAME = "synthetic-medical-demo"


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


def make_synthetic_zip(seed: int = 20260913, per_class: int = 90) -> bytes:
    """生成 4 类 x per_class 张 64x64 合成图（医学影像风格纹理，非真实数据）。"""
    rng = random.Random(seed)
    size = 64
    buf = io.BytesIO()

    def draw_class(img: Image.Image, cls: int, rng: random.Random) -> None:
        d = ImageDraw.Draw(img, "RGBA")
        cx, cy = size / 2, size / 2
        colors = [
            (80, 170, 240, 255),   # class_0 亮蓝病灶
            (235, 150, 60, 255),   # class_1 橙黄纹理
            (150, 240, 130, 255),  # class_2 浅绿双团
            (235, 90, 120, 255),   # class_3 粉红棋盘
        ]
        color = colors[cls % len(colors)]
        if cls == 0:
            r = rng.randint(8, 13)
            ox, oy = rng.randint(-6, 6), rng.randint(-6, 6)
            d.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r], fill=color)
            d.ellipse([cx + ox - r * 2.5, cy + oy - r * 2.5,
                       cx + ox + r * 2.5, cy + oy + r * 2.5],
                      outline=(255, 255, 255, 90), width=2)
        elif cls == 1:
            for i in range(-size, size, 5):
                off = rng.randint(-1, 1)
                d.line([(i, 0), (i + off, size)], fill=color, width=rng.randint(1, 2))
        elif cls == 2:
            for dx in (-9, 8):
                r = rng.randint(5, 8)
                x = cx + dx + rng.randint(-3, 3)
                y = cy + rng.randint(-8, 8)
                d.ellipse([x - r, y - r, x + r, y + r], fill=color)
            d.line([(cx - 14, cy), (cx + 14, cy)], fill=(255, 255, 255, 110), width=2)
        else:
            cell = rng.randint(10, 13)
            for i in range(0, size, cell):
                for j in range(0, size, cell):
                    if (i // cell + j // cell) % 2 == 0:
                        d.rectangle([i, j, i + cell - 1, j + cell - 1], fill=color)

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for cls in range(4):
            folder = f"class_{cls}"
            for i in range(per_class):
                img = Image.new("RGB", (size, size), (18, 22, 26))
                draw_class(img, cls, rng)
                # 均匀噪声 + 随机亮度/对比度扰动，逼近真实成像噪声
                pix = img.load()
                for _ in range(int(size * size * 0.35)):
                    x, y = rng.randrange(size), rng.randrange(size)
                    r0, g0, b0 = pix[x, y]
                    delta = rng.randint(-28, 28)
                    pix[x, y] = (min(255, max(0, r0 + delta)),
                                 min(255, max(0, g0 + delta)),
                                 min(255, max(0, b0 + delta)))
                zf.writestr(f"{folder}/{i:03d}.png", _png_bytes(img))
    return buf.getvalue()


def _png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


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
    print(f"[demo] 合成图像数据集已导入: {ds_id} "
          f"({meta['dataset']['n_images']} 张, 4 类)")
    return ds_id


def ensure_builtin(state: dict, key: str) -> str:
    ds_id = state.get("builtin_ds_id")
    if ds_id and dataset_exists(ds_id):
        return ds_id
    meta = api("POST", "/api/datasets/builtin", json={"name": key})
    ds_id = meta["dataset"]["id"]
    state["builtin_ds_id"] = ds_id
    save_state(state)
    print(f"[demo] 内置数据集已载入: {ds_id} "
          f"({meta['dataset']['n_rows']} 行)")
    return ds_id


def create_run(task: str, dataset_id: str, model: str, params: dict, *,
               group: str, name: str, note: str, text_column: str | None = None) -> str:
    body = {
        "dataset_id": dataset_id,
        "task": task,
        "model": model,
        "params": params,
        "name": name,
        "group": group,
        "note": note,
        "test_size": 0.2,
        "val_split": 0.2,
        "random_state": 42,
    }
    if text_column:
        body["text_column"] = text_column
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


def ensure_single_run(state: dict, key: str, creator) -> str:
    run_id = (state.get("run_ids") or {}).get(key)
    if run_id:
        try:
            d = run_detail(run_id)
        except Exception:
            d = {}
        if d.get("state") == "done":
            return run_id
        if d.get("state") == "running":
            print(f"[demo] {key} 已在后台运行，等待完成 ...")
            wait_runs([run_id])
            return run_id
    run_id = creator()
    state.setdefault("run_ids", {})[key] = run_id
    save_state(state)
    print(f"[demo] 创建 {key}: {run_id}")
    wait_runs([run_id])
    return run_id


def ensure_batch(state: dict, key: str, creator) -> list[str]:
    ids = (state.get("run_ids") or {}).get(key) or []
    if ids:
        live = []
        pending = []
        for rid in ids:
            try:
                d = run_detail(rid)
            except Exception:
                continue
            if d.get("state") == "done":
                live.append(rid)
            elif d.get("state") == "running":
                pending.append(rid)
        if pending:
            print(f"[demo] {key} 批次在后台运行，等待完成 ...")
            wait_runs(pending)
        if live:
            return live
    new_ids = creator()
    state.setdefault("run_ids", {})[key] = new_ids
    save_state(state)
    print(f"[demo] 创建 {key} 批次: {len(new_ids)} 个实验")
    wait_runs(new_ids)
    return new_ids


def image_params(optimizer: str = "adamw", lr: float = 0.001, epochs: int = 15,
                 scheduler: str = "cosine", weight_decay: float = 0.0,
                 dropout: float = 0.4, activation: str = "gelu") -> dict:
    return {
        "optimizer": optimizer,
        "lr": lr,
        "batch_size": 32,
        "epochs": epochs,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "early_stop_patience": 4 if epochs > 6 else 0,
        "grad_clip": 1.0,
        "seed": 42,
        "conv_channels": "32,64,128",
        "activation": activation,
        "dropout": dropout,
        "image_size": 64,
    }


def main() -> None:
    state = load_state()
    DOCS.mkdir(exist_ok=True)

    print("[demo] 1/5 准备数据集 ...")
    image_ds = ensure_image_dataset(state)
    table_ds = ensure_builtin(state, "breast_cancer")

    print("[demo] 2/5 图像 CNN：基线 + 改进 ...")
    image_baseline = ensure_single_run(
        state, "image_baseline",
        lambda: create_run(
            "image_classification", image_ds, "cnn",
            image_params(optimizer="sgd", lr=0.01, epochs=6, scheduler="none",
                         weight_decay=0.0, dropout=0.3, activation="relu"),
            group="baseline", name="图像基线-CNN-SGD",
            note="合成演示数据（非真实医学影像）；CNN 基线：SGD/lr0.01/6轮/无调度"))
    image_improved = ensure_single_run(
        state, "image_improved",
        lambda: create_run(
            "image_classification", image_ds, "cnn",
            image_params(optimizer="adamw", lr=0.001, epochs=15, scheduler="cosine",
                         weight_decay=1e-4, dropout=0.4, activation="gelu"),
            group="improved", name="图像改进-CNN-AdamW余弦",
            note="合成演示数据；改进：AdamW/余弦调度/权重衰减/早停/梯度裁剪/GELU+Dropout"))

    print("[demo] 3/5 改进实验：重复 3 次 + 一键消融 ...")
    repeats = ensure_batch(
        state, "image_repeats",
        lambda: api("POST", "/api/experiments/repeats",
                    json={"run_id": image_improved, "count": 3})["run_ids"])
    ablation_ids = ensure_batch(
        state, "image_ablation",
        lambda: api("POST", "/api/experiments/ablation",
                    json={"run_id": image_improved,
                          "overrides": {
                              "dropout": [0.0],
                              "weight_decay": [0.0],
                              "optimizer": ["sgd"],
                              "scheduler": ["none"],
                          }})["run_ids"])

    print("[demo] 4/5 表格乳腺癌：逻辑回归基线 + MLP 改进 ...")
    table_baseline = ensure_single_run(
        state, "table_baseline",
        lambda: create_run(
            "tabular_classification", table_ds, "logistic_regression",
            {"C": 1.0, "max_iter": 1000},
            group="baseline", name="表格基线-逻辑回归",
            note="乳腺癌威斯康星公开数据集（表格分类基线）"))
    table_mlp = ensure_single_run(
        state, "table_mlp",
        lambda: create_run(
            "tabular_classification", table_ds, "mlp",
            {"hidden_sizes": "128,64", "activation": "gelu", "dropout": 0.2,
             "optimizer": "adamw", "lr": 0.001, "batch_size": 32, "epochs": 12,
             "scheduler": "cosine", "weight_decay": 1e-4,
             "early_stop_patience": 4, "grad_clip": 1.0, "seed": 42},
            group="improved", name="表格改进-MLP",
            note="乳腺癌威斯康星公开数据集（MLP 神经网络对照）"))

    all_ids = [image_baseline, image_improved] + repeats + ablation_ids + \
        [table_baseline, table_mlp]
    report_ids = [image_baseline, image_improved, repeats[0], ablation_ids[0],
                  table_baseline, table_mlp]

    print("[demo] 5/5 生成对比表与 Word 报告 ...")
    cmp = api("POST", "/api/experiments/compare", json={"run_ids": all_ids})
    csv_path = DOCS / "实验对比_演示数据.csv"
    md_path = DOCS / "实验对比_演示数据.md"
    csv_path.write_text(cmp["csv"], encoding="utf-8")
    header = ("# 实验对比表（合成演示数据，非真实医学结论）\n\n"
              "> 数据：程序生成的合成医学风格图像 4 类 + 乳腺癌威斯康星公开数据集。\n"
              "> 复现：`python scripts/run_thesis_demo.py`（需先启动 `python -m app.main --no-browser`）。\n\n")
    md_path.write_text(header + cmp["markdown"] or "（无已完成实验）", encoding="utf-8")
    print(f"[demo] 对比表已写入 {csv_path.name} / {md_path.name} "
          f"（{cmp['count']} 个实验）")

    doc = api("POST", "/api/report/generate", json={
        "title": "基于神经网络的医学影像分类研究（合成演示数据）",
        "run_ids": report_ids,
        "dataset_id": image_ds,
        "author": {"name": "毕设工坊演示作者", "school": "演示学校",
                   "major": "人工智能", "student_id": "2020XXXXXX"},
        "ai_draft": False,
    })
    src = Path(doc["path"])
    if src.exists():
        dst = DOCS / ("报告草稿_" + src.name)
        dst.write_bytes(src.read_bytes())
        print(f"[demo] Word 报告已生成并复制到 {dst}")
    else:
        print(f"[demo] 报告接口返回路径但文件不存在: {doc['path']}")

    print("\n[ demo ] 全部完成")
    print("实验总数: %d" % len(all_ids))
    print("复现文档: docs/实验报告_演示数据.md（下一步生成）")


if __name__ == "__main__":
    main()
