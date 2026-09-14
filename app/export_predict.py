"""为已完成的实验生成可直接运行的推理脚本 predict.py。

脚本与权重放在同一个实验目录,运行时只需 `python predict.py ...`。
它不是训练流程的一部分,失败也不应影响实验结论,因此调用方统一吞掉异常。
"""
from __future__ import annotations

from pathlib import Path


def _literal(value) -> str:
    return repr(value)


def _torch_template(cfg: dict) -> str:
    task = cfg.get("task")
    model = cfg.get("model")
    params = cfg.get("params") or {}
    text_column = cfg.get("text_column")
    target = cfg.get("target")
    return f'''# -*- coding: utf-8 -*-
"""自动生成的推理脚本:在测试样本上复现你的模型预测结果。

用法:
  表格任务: python predict.py --input 一行样本.csv
  文本任务: python predict.py --text "一段新的文本"
  图像任务: python predict.py --image 图片.png

模型权重同目录的 best.pt,请与 predict.py 保持在同一文件夹。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path


def _find_root(start: Path):
    candidates = []
    env = os.environ.get("THESISFORGE_ROOT")
    if env:
        candidates.append(Path(env))
    executable = Path(sys.executable).resolve()
    candidates += [
        start, *start.parents,
        executable.parent, *executable.parents,
        Path.cwd(), *Path.cwd().parents,
    ]
    for candidate in candidates:
        if (candidate / "app" / "networks.py").exists():
            return candidate
    return None


HERE = Path(__file__).resolve().parent
_ROOT = _find_root(HERE)
if _ROOT is not None:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch

from app.networks import build_model

TASK = {_literal(task)}
MODEL = {_literal(model)}
PARAMS = {_literal(params)}
TEXT_COLUMN = {_literal(text_column)}
TARGET = {_literal(target)}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\\u4e00-\\u9fff]")


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(str(text).lower()) if t]


def load_model():
    ckpt = torch.load(HERE / "best.pt", map_location="cpu", weights_only=False)
    dims = {{}}
    if ckpt.get("num_features") is not None:
        dims["num_features"] = int(ckpt["num_features"])
    if ckpt.get("num_classes") is not None:
        dims["num_classes"] = int(ckpt["num_classes"])
    if ckpt.get("num_tokens") is not None:
        dims["num_tokens"] = int(ckpt["num_tokens"])
        dims["max_seq_len"] = int(ckpt.get("max_seq_len") or 128)
    net = build_model(TASK, MODEL, PARAMS, **dims)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return ckpt, net


def predict_tabular(ckpt, net, csv_path: str):
    row = pd.read_csv(csv_path).head(1)
    pre = ckpt.get("preprocessor")
    if pre is None:
        raise RuntimeError("该模型没有保存预处理流程,请重新训练后再导出")
    x = pre.transform(row)
    x = np.asarray(x.toarray() if hasattr(x, "toarray") else x).astype(np.float32)
    with torch.no_grad():
        logits = net(torch.tensor(x))
    return _decode(ckpt, logits)


def predict_text(ckpt, net, text: str):
    vocab = ckpt.get("vocab")
    if not vocab:
        raise RuntimeError("该模型没有保存词表,请重新训练后再导出")
    seq_len = int(ckpt.get("max_seq_len") or 128)
    ids = [vocab.get(t, 1) for t in tokenize(text)][:seq_len]
    row = np.zeros((1, seq_len), dtype=np.int64)
    row[0, : len(ids)] = ids
    with torch.no_grad():
        logits = net(torch.tensor(row))
    return _decode(ckpt, logits)


def predict_image(ckpt, net, image_path: str):
    from PIL import Image
    from torchvision import transforms

    size = int(ckpt.get("image_size") or 64)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = net(img)
    return _decode(ckpt, logits)


def _decode(ckpt, logits):
    if TASK == "tabular_regression":
        return {{"predicted_value": round(float(logits.reshape(-1)[0]), 4)}}
    probs = torch.softmax(logits[0], dim=0)
    classes = ckpt.get("classes") or []
    idx = int(probs.argmax(dim=0))
    return {{
        "predicted_label": classes[idx] if idx < len(classes) else str(idx),
        "probabilities": [
            {{"label": classes[i] if i < len(classes) else str(i), "prob": round(float(p), 4)}}
            for i, p in enumerate(probs)
        ],
    }}


def main():
    parser = argparse.ArgumentParser(description="用导出的 best.pt 做单样本预测")
    parser.add_argument("--input", help="表格任务:一行样本的 csv 文件")
    parser.add_argument("--text", help="文本任务:待预测文本")
    parser.add_argument("--image", help="图像任务:待预测图片路径")
    args = parser.parse_args()

    ckpt, net = load_model()
    if TASK == "image_classification":
        if not args.image:
            parser.error("图像任务需要 --image")
        result = predict_image(ckpt, net, args.image)
    elif TASK == "text_classification":
        if not args.text:
            parser.error("文本任务需要 --text")
        result = predict_text(ckpt, net, args.text)
    else:
        if not args.input:
            parser.error("表格任务需要 --input")
        result = predict_tabular(ckpt, net, args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def _sklearn_template(cfg: dict) -> str:
    task = cfg.get("task")
    text_column = cfg.get("text_column")
    return f'''# -*- coding: utf-8 -*-
"""自动生成的推理脚本:用同目录 model.pkl 做单样本预测。

用法:
  表格任务: python predict.py --input 一行样本.csv
  文本任务: python predict.py --text "一段新的文本"

模型文件为 model.pkl,请与 predict.py 保持在同一文件夹。
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
TASK = {_literal(task)}
TEXT_COLUMN = {_literal(text_column)}


def predict(input_csv=None, text=None):
    pipe = joblib.load(HERE / "model.pkl")
    if TASK == "text_classification":
        if not text:
            raise SystemExit("文本任务需要 --text")
        x = pd.DataFrame({{TEXT_COLUMN or "text": [text]}})
    else:
        if not input_csv:
            raise SystemExit("表格任务需要 --input")
        x = pd.read_csv(input_csv).head(1)
    pred = pipe.predict(x)
    out = {{"prediction": str(pred[0])}}
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(x)[0]
            out["probabilities"] = [round(float(p), 4) for p in proba]
        except Exception:
            pass
    return out


def main():
    parser = argparse.ArgumentParser(description="用导出的 model.pkl 做单样本预测")
    parser.add_argument("--input", help="表格任务:一行样本的 csv 文件")
    parser.add_argument("--text", help="文本任务:待预测文本")
    args = parser.parse_args()
    print(json.dumps(predict(args.input, args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def _ultralytics_template(cfg: dict) -> str:
    task = cfg.get("task")
    model = cfg.get("model")
    params = cfg.get("params") or {}
    return f'''# -*- coding: utf-8 -*-
"""自动生成的推理脚本:用同目录 best.pt 对一张图片做检测/分割预测。

用法:
  python predict.py --image 图片.png

输出:
  终端打印 JSON,包含检测框、类别、置信度;分割任务还包含掩码数量。
  可视化结果保存到同目录 predict_output/ 下。
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = {_literal(task)}
MODEL = {_literal(model)}
PARAMS = {_literal(params)}


def load_model():
    weights = HERE / "best.pt"
    if not weights.exists():
        raise SystemExit("找不到 best.pt,请把它和 predict.py 放在同一目录")
    if str(MODEL).startswith("rtdetr"):
        from ultralytics import RTDETR
        return RTDETR(str(weights))
    from ultralytics import YOLO
    return YOLO(str(weights))


def predict(image_path: str, conf: float):
    model = load_model()
    out_dir = HERE / "predict_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        project=str(out_dir),
        name="predict",
        exist_ok=True,
        conf=conf,
        device=PARAMS.get("device", "auto"),
    )
    if not results:
        return {{"image": image_path, "boxes": [], "classes": [], "confidence": [], "masks": 0}}

    result = results[0]
    boxes = result.boxes
    class_names = getattr(model, "names", {{}}) or {{}}
    payload = {{
        "image": image_path,
        "output_dir": str(out_dir / "predict"),
        "boxes": [box.tolist() for box in boxes.xywh.tolist()],
        "classes": [
            class_names.get(int(index), str(int(index)))
            for index in boxes.cls.tolist()
        ],
        "confidence": [round(float(value), 4) for value in boxes.conf.tolist()],
        "masks": 0,
    }}
    if TASK == "semantic_segmentation" and result.masks is not None:
        payload["masks"] = len(result.masks)
    return payload


def main():
    parser = argparse.ArgumentParser(description="用导出的 best.pt 做单张图片预测")
    parser.add_argument("--image", required=True, help="待预测图片路径")
    parser.add_argument("--conf", type=float, default=float(PARAMS.get("conf", 0.25)),
                        help="置信度阈值")
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.conf), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def write_predict_script(run_dir: Path, engine: str, cfg: dict) -> Path:
    """按训练引擎生成 predict.py,返回脚本路径。"""
    run_dir = Path(run_dir).resolve()
    if engine == "ultralytics":
        text = _ultralytics_template(cfg)
    elif engine == "torch":
        text = _torch_template(cfg)
    else:
        text = _sklearn_template(cfg)
    path = run_dir / "predict.py"
    path.write_text(text, encoding="utf-8")
    return path
