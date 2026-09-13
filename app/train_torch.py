"""PyTorch 神经网络训练脚本（独立子进程运行）。

用法: python app/train_torch.py --run-dir data/runs/<run_id>
从 run_dir/config.json 读取配置，由 app.trainer 统一执行表格 MLP、图像 CNN/ResNet18、
文本 LSTM/GRU/TextCNN/Transformer。产出 metrics.jsonl / summary.json / best.pt /
图表 / status.json；失败时写 failed 状态并返回非 0 退出码。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.trainer import train_from_run_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    return train_from_run_dir(Path(args.run_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
