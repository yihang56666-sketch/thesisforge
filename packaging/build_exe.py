# -*- coding: utf-8 -*-
"""构建 ThesisForge 的 Windows EXE（两种方式，共用同一个入口脚本）。

1) 启动器 EXE（小）    dist/launcher/ThesisForge.exe
   放进离线整合包根目录，与 runtime\\python.exe 配对；双击即可，不用碰 .bat。

2) 独立单文件 EXE（大） dist/ThesisForge-vX.Y.Z-win-x64.exe
   自带 Python 与全部科学计算依赖，默认打开桌面窗口（WebView2 缺失时自动
   降级为浏览器），双击就能跑表格/文本任务。
   刻意排除 torch：CUDA 版体积超过 GitHub 单文件 2GB 限制，
   图像训练请用离线整合包（内嵌 Python + 一键安装脚本）。

用法：
    python packaging/build_exe.py            # 只构建独立 EXE
    python packaging/build_exe.py --both     # 启动器 + 独立都构建

依赖 PyInstaller，仅开发机需要（见 requirements-dev.txt），终端用户无需安装。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import APP_VERSION  # noqa: E402

DIST = ROOT / "dist"
BUILD = ROOT / "build" / "pyinstaller"
ENTRY = ROOT / "packaging" / "ThesisForge.py"
ICON = ROOT / "packaging" / "app.ico"

# 打包机装了 Anaconda 的 cuDNN torch，hooks 会顺带拖进 3GB DLL，必须显式排除
EXCLUDES = [
    "torch", "torchvision", "torchaudio", "torchgen",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "jedi",
    "pytest", "setuptools", "pip", "wheel",
    "notebook", "nbformat", "jupyter_client", "jupyter", "IPython.kernel",
    "tkinter", "test", "setuptools._vendor", "cython",
    "sklearn.tests", "numpy.tests", "scipy.tests", "pandas.tests",
    "matplotlib.tests", "matplotlib_backports.tests",
]


def step(msg: str) -> None:
    print(f"[exe] {msg}", flush=True)


def _common(name: str, outdir: Path, collect_app: bool = False) -> list[str]:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile", "--console",
        "--name", name,
        "--distpath", str(DIST / outdir),
        "--workpath", str(BUILD / name),
        "--specpath", str(BUILD / name),
        "--paths", str(ROOT),                       # 让 app/ 成为可发现的包
        "--add-data", f"{ROOT / 'web'}{os.pathsep}web",
    ]
    if collect_app:
        # worker 靠模块名动态加载，PyInstaller 需要显式收进 app 全套子模块
        cmd += ["--collect-submodules", "app"]
    if ICON.exists():
        cmd += ["--icon", str(ICON)]
    return cmd


def build_standalone() -> Path:
    step(f"构建独立单文件 EXE v{APP_VERSION}（数分钟）...")
    cmd = _common("ThesisForge", Path(""), collect_app=True)
    # 不加 --collect-all：PyInstaller 自带 sklearn/pandas/numpy/matplotlib 钩子，
    # collect-all 会把 *.tests 全部拖进来，体积和时间都翻倍。
    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    cmd.append(str(ENTRY))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    produced = DIST / "ThesisForge.exe"        # onefile 直接落在 distpath 根下
    if not produced.exists():
        raise SystemExit(f"PyInstaller 未产出 {produced}")
    DIST.mkdir(exist_ok=True)
    target = DIST / f"ThesisForge-v{APP_VERSION}-win-x64.exe"
    if target.exists():
        target.unlink()
    shutil.move(str(produced), str(target))
    step(f"完成: {target}  ({target.stat().st_size / 1048576:.0f} MB)")
    return target


def build_launcher() -> Path:
    step("构建启动器 EXE（小，供离线整合包使用）...")
    cmd = _common("ThesisForge-launcher", Path("launcher"))
    cmd += [
        "--exclude-module", "app", "--exclude-module", "torch",
        "--exclude-module", "torchvision", "--exclude-module", "pandas",
        "--exclude-module", "numpy", "--exclude-module", "sklearn",
        "--exclude-module", "scipy", "--exclude-module", "matplotlib",
        "--exclude-module", "fastapi", "--exclude-module", "uvicorn",
        "--exclude-module", "docx", "--exclude-module", "httpx",
        "--exclude-module", "PIL", "--exclude-module", "webview",
    ]
    cmd.append(str(ENTRY))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    produced = DIST / "launcher" / "ThesisForge-launcher.exe"
    if not produced.exists():
        raise SystemExit(f"PyInstaller 未产出 {produced}")
    step(f"完成: {produced}  ({produced.stat().st_size / 1048576:.1f} MB)")
    return produced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true", help="同时构建启动器 EXE")
    ap.add_argument("--launcher-only", action="store_true", help="只构建启动器 EXE")
    args = ap.parse_args()

    if args.launcher_only:
        build_launcher()
    elif args.both:
        build_launcher()
        build_standalone()
    else:
        build_standalone()


if __name__ == "__main__":
    main()
