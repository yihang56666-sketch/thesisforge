"""全局路径与运行时配置（LLM 接口等）。

凭据安全约定：
- 源码中绝不写入任何真实 API Key；
- 运行时优先从环境变量 LLM_API_KEY 读取；
- 用户在控制面板填写的 Key 保存在 data/runtime_config.json（用户本地数据，
  已被 .gitignore 排除，不属于源码）。
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

def _bundle_root() -> Path:
    """只读资源根目录（web/ 等）。PyInstaller 打包后是解包临时目录。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def _app_home() -> Path:
    """可写根目录：打包后是 EXE 所在目录，开发时是仓库根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


FROZEN = bool(getattr(sys, "frozen", False))
BUNDLE_DIR = _bundle_root()
ROOT = _app_home()
WEB_DIR = BUNDLE_DIR / "web"

_data_override = os.environ.get("THESISFORGE_DATA_DIR", "").strip()
DATA_DIR = Path(_data_override).resolve() if _data_override else ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
EXPORTS_DIR = DATA_DIR / "exports"
UPLOADS_DIR = DATA_DIR / "uploads"
CONFIG_FILE = DATA_DIR / "runtime_config.json"

APP_VERSION = "0.2.1"

DEFAULT_RUNTIME_CONFIG: dict = {
    # OpenAI 兼容接口地址，如 https://api.deepseek.com/v1
    "llm_base_url": "",
    "llm_model": "",
    "llm_api_key": "",          # 由用户在面板中填写，或留空走环境变量
    "llm_api_key_env": "LLM_API_KEY",
    "llm_max_tokens": 2000,
    "llm_temperature": 0.4,
}

_lock = threading.Lock()


def ensure_dirs() -> None:
    for d in (DATA_DIR, DATASETS_DIR, RUNS_DIR, EXPORTS_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_runtime_config() -> dict:
    cfg = dict(DEFAULT_RUNTIME_CONFIG)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_runtime_config(update: dict) -> dict:
    with _lock:
        cfg = load_runtime_config()
        for k in DEFAULT_RUNTIME_CONFIG:
            if k in update and update[k] is not None:
                cfg[k] = update[k]
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return cfg


def resolve_api_key(cfg: dict | None = None) -> str:
    """环境变量优先，其次面板中保存的 Key。"""
    cfg = cfg or load_runtime_config()
    env_name = cfg.get("llm_api_key_env") or "LLM_API_KEY"
    key = (os.environ.get(env_name) or "").strip()
    if key:
        return key
    return (cfg.get("llm_api_key") or "").strip()
