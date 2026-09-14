# -*- coding: utf-8 -*-
"""YOLO 目标检测训练脚本（独立子进程运行，依赖 ultralytics）。

用法: python app/train_detection.py --run-dir data/runs/<run_id>
从 run_dir/config.json 读取配置；数据集为 YOLO 格式（images/ + labels/ 或 data.yaml）。
产出 metrics.jsonl / summary.json / YOLO 训练产物 / status.json。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from app.model_scanner import find_model_path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_WEIGHT_NAMES = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolov8n-seg": "yolov8n-seg.pt",
    "yolov8s-seg": "yolov8s-seg.pt",
    "yolov8m-seg": "yolov8m-seg.pt",
    "yolov8l-seg": "yolov8l-seg.pt",
    "yolov8x-seg": "yolov8x-seg.pt",
    "rtdetr-l": "rtdetr-l.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26n-seg": "yolo26n-seg.pt",
    "yolo26s-seg": "yolo26s-seg.pt",
}


def yolo_weights(model_key: str) -> str:
    if model_key in _WEIGHT_NAMES:
        return _WEIGHT_NAMES[model_key]
    raise ValueError(f"不支持的检测/分割模型: {model_key}")


def find_model_weights(model_key: str, extra_dirs=None) -> str:
    """优先使用用户本机已拷贝的权重，否则交给 Ultralytics 下载。"""
    local = find_model_path(model_key, extra_dirs)
    if local is not None:
        return str(local)
    return yolo_weights(model_key)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe(base: Path, name: str) -> Path:
    base = base.resolve()
    t = base / name
    if ".." in t.parts or not t.resolve().is_relative_to(base):
        raise ValueError("非法路径")
    return t.resolve()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_data_yaml(dataset_dir: Path) -> Path:
    for name in ("data.yaml", "data.yml"):
        p = dataset_dir / name
        if p.exists():
            return p
    for p in sorted(dataset_dir.rglob("data.y*ml")):
        return p
    raise FileNotFoundError(f"YOLO 数据集缺少 data.yaml: {dataset_dir}")


def materialize_data_yaml(dataset_dir: Path, data_yaml: Path, run_dir: Path) -> Path:
    """把 data.yaml 的 path 重写为数据集绝对路径。

    Ultralytics 把相对 path 解析到它自己的全局 datasets_dir 设置，
    中文路径还可能被转码；这里落到 run_dir 里用绝对路径，不动用户原始文件。
    """
    import yaml

    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    spec["path"] = str(dataset_dir.resolve())
    out = run_dir / "data.yaml"
    out.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()

    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)
        write_json(safe(run_dir, "metrics.jsonl"), events)

    t0 = time.time()
    try:
        cfg = json.loads(safe(run_dir, "config.json").read_text(encoding="utf-8"))
        task = cfg["task"]
        model_key = cfg["model"]
        params = cfg.get("params") or {}
        dataset_dir = Path(cfg["dataset_dir"]).resolve()
        log(f"任务: {task} | 模型: {model_key} | 参数: {params}")

        data_yaml = materialize_data_yaml(dataset_dir, find_data_yaml(dataset_dir), run_dir)
        log(f"数据配置: {data_yaml}（path 已重写为数据集绝对路径）")

        weights_cfg = cfg.get("weights_path")
        weights = str(weights_cfg) if weights_cfg and Path(weights_cfg).is_file() else find_model_weights(model_key)

        if model_key.startswith("rtdetr"):
            from ultralytics import RTDETR
            model = RTDETR(weights)
        else:
            from ultralytics import YOLO
            model = YOLO(weights)
        device = params.get("device", "auto")
        if device == "auto":
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = 0
        log(f"设备: {device}")

        emit({"type": "stage", "name": "train_start"})
        model.train(
            data=str(data_yaml),
            epochs=int(params.get("epochs", 50)),
            imgsz=int(params.get("imgsz", 640)),
            batch=int(params.get("batch_size", 16)),
            lr0=float(params.get("lr0", 0.01)),
            patience=int(params.get("patience", 20)),
            seed=int(params.get("seed", 42)),
            device=device,
            project=str(run_dir),
            name="yolo",
            exist_ok=True,
            verbose=False,
        )
        save_dir = run_dir / "yolo"
        log(f"训练完成，产物目录: {save_dir}")

        emit({"type": "stage", "name": "val_start"})
        metrics_obj = model.val(data=str(data_yaml), device=device, verbose=False)
        box = metrics_obj.box
        metrics = {
            "map50": round(float(box.map50), 5),
            "map50_95": round(float(box.map), 5),
            "precision": round(float(box.mp), 5),
            "recall": round(float(box.mr), 5),
        }
        emit({"type": "test", **metrics})
        log(f"测试指标: mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}")

        artifacts: list[str] = []
        for src_name, out_name in (("results.png", "results.png"), ("results.csv", "results.csv"),
                                   ("confusion_matrix.png", "confusion_matrix.png"),
                                   ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png")):
            src = save_dir / src_name
            if src.exists():
                shutil.copy2(src, safe(run_dir, out_name))
                artifacts.append(out_name)
        # 验证批预测示例，答辩可直接展示
        for src in sorted(save_dir.rglob("val_batch*_pred.jpg"))[:2]:
            out = safe(run_dir, src.name)
            shutil.copy2(src, out)
            artifacts.append(src.name)

        summary: dict = {
            "task": task, "model": model_key, "model_label": cfg.get("model_label", model_key),
            "dataset_name": cfg.get("dataset_name"), "params": params, "engine": "ultralytics",
            "metrics": metrics,
            "primary_metric": {"name": "map50", "value": metrics["map50"]},
            "split_scheme": "ultralytics train/val (data.yaml)",
            "epochs_run": int(params.get("epochs", 50)),
            "device": str(device),
            "artifacts": artifacts,
            "train_time_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        weights_path = save_dir / "weights" / "best.pt"
        if weights_path.exists():
            shutil.copy2(weights_path, safe(run_dir, "best.pt"))
            artifacts.append("best.pt")
            try:
                from app.export_predict import write_predict_script

                write_predict_script(run_dir, "ultralytics", cfg)
                artifacts.append("predict.py")
            except Exception:
                log("predict.py 导出失败，训练结果不受影响")
            summary["artifacts"] = artifacts
            log("最佳模型已保存: best.pt")
        write_json(safe(run_dir, "summary.json"), summary)
        emit({"type": "summary", "primary_metric": summary["primary_metric"], "metrics": metrics})
        log(f"全部完成，用时 {summary['train_time_sec']}s")
        write_json(safe(run_dir, "status.json"),
                   {"state": "done", "error": None, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 0
    except Exception:
        err = traceback.format_exc()
        log("训练失败:\n" + err)
        low = err.lower()
        if "out of memory" in low or "cuda oom" in low:
            kind, hint = "oom", "显存/内存不足。建议减小 batch_size 或 imgsz，或改用 CPU 训练。"
        elif "no space left" in low or ("disk" in low and "space" in low):
            kind, hint = "disk", "磁盘空间不足。请清理实验输出目录所在磁盘后重试。"
        else:
            kind, hint = "error", "训练异常，具体原因见下方错误信息与运行日志（data/logs/launch.log）。"
        write_json(safe(run_dir, "status.json"),
                   {"state": "failed", "error": err[-1500:], "failure_kind": kind,
                    "failure_hint": hint, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 1


if __name__ == "__main__":
    sys.exit(main())
