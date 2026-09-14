# -*- coding: utf-8 -*-
"""构建 Windows 离线整合包：内嵌 Python 3.13 + 预装依赖 + 源码，解压即用。

产物: dist/ThesisForge-v{APP_VERSION}-win64-offline.zip
安全约定：
- 下载地址为硬编码的 python.org 官方 https 直链，经 validate_public_http_url
  域名白名单校验后，由 app.datasets_hub._fetch 执行（内含 SSRF 逐跳校验：
  仅 http/https、拒绝内网/环回/保留地址、重定向逐跳复检）；
- zip 解压逐成员校验（拒绝 ..、盘符/绝对路径成员），限制在目标目录内。
不含 PyTorch（体积原因），包内提供 CPU/CUDA 一键安装脚本。
"""
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import APP_VERSION  # noqa: E402
from app.datasets_hub import _fetch  # noqa: E402
from app.security import validate_public_http_url  # noqa: E402

DIST = ROOT / "dist"
PKG = DIST / f"ThesisForge-v{APP_VERSION}-win64"
RT = PKG / "runtime"
PY_EMBED_URL = "https://www.python.org/ftp/python/3.13.9/python-3.13.9-embed-amd64.zip"
PY_EMBED_CACHE = ROOT / "build" / "cache"
ALLOWED_HOST = "www.python.org"
DEPS = ["fastapi", "uvicorn", "scikit-learn", "pandas", "numpy", "matplotlib",
        "joblib", "httpx", "python-docx", "python-multipart", "pywebview"]


def step(msg):
    print(f"[build] {msg}", flush=True)


def safe_extract_zip(zf_source, dest: Path) -> None:
    """防 zip-slip：逐成员规范化校验，拒绝 ..、盘符/绝对路径，限制在 dest 内。"""
    dest = dest.resolve()
    with zipfile.ZipFile(zf_source) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            normalized = member.filename.replace("\\", "/")
            if ":" in normalized or normalized.startswith("/"):
                continue
            target = (dest / member.filename).resolve()
            if ".." in target.parts or not target.is_relative_to(dest):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def download_python_embed() -> bytes:
    """下载官方内嵌 Python；支持缓存和环境变量提供的离线包。"""
    cached = os.environ.get("THESISFORGE_PY_EMBED_ZIP", "").strip()
    if cached:
        data = Path(cached).read_bytes()
        PY_EMBED_CACHE.mkdir(parents=True, exist_ok=True)
        (PY_EMBED_CACHE / Path(PY_EMBED_URL).name).write_bytes(data)
        return data
    cache_file = PY_EMBED_CACHE / Path(PY_EMBED_URL).name
    if cache_file.exists():
        return cache_file.read_bytes()
    validate_public_http_url(PY_EMBED_URL)
    if urlparse(PY_EMBED_URL).hostname != ALLOWED_HOST:
        raise ValueError(f"仅允许从 {ALLOWED_HOST} 下载")
    data = _fetch(PY_EMBED_URL)
    PY_EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(data)
    return data


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)
    (RT / "site-packages").mkdir(parents=True)

    # 1. 内嵌 Python
    step("下载内嵌 Python 3.13.9 ...")
    content = download_python_embed()
    safe_extract_zip(io.BytesIO(content), RT)
    step("配置 python313._pth ...")
    pth = RT / "python313._pth"
    pth.write_text("python313.zip\n.\n..\nsite-packages\nimport site\n", encoding="ascii")

    # 2. 预装依赖（用系统 pip 安装到包内 site-packages）
    site_packages_src = os.environ.get("THESISFORGE_SITE_PACKAGES", "").strip()
    if site_packages_src:
        step(f"复用本地依赖目录 {site_packages_src} ...")
        shutil.copytree(site_packages_src, RT / "site-packages", dirs_exist_ok=True)
    else:
        step("安装核心依赖（较大，请耐心）...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(RT / "site-packages"),
                        "--no-warn-script-location", *DEPS], check=True)
    if not (RT / "site-packages" / "pip").exists():
        step("安装 pip（供包内一键安装 PyTorch 用）...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(RT / "site-packages"),
                        "--no-warn-script-location", "pip"], check=True)
    else:
        step("复用本地 site-packages 中的 pip ...")

    # 3. 复制源码
    step("复制源码与文档 ...")
    shutil.copytree(ROOT / "app", PKG / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "web", PKG / "web")
    docs_dir = PKG / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc in ["新手使用手册.md", "目标检测指南.md", "模型解释性指南.md",
                "时间序列预测指南.md", "自动调参指南.md", "ISSUES.md", "ROADMAP.md"]:
        shutil.copy2(ROOT / "docs" / doc, docs_dir / doc)
    for f in ["README.md", "LICENSE", "requirements.txt"]:
        shutil.copy2(ROOT / f, PKG / f)

    # 4. 启动器与安装脚本
    step("复制启动器 EXE ...")
    launcher = DIST / "launcher" / "ThesisForge-launcher.exe"
    if not launcher.exists():
        raise SystemExit("缺少启动器 EXE，请先运行: python packaging/build_exe.py --launcher-only")
    shutil.copy2(launcher, PKG / "ThesisForge.exe")
    step("生成启动脚本 ...")
    (PKG / "启动毕设工坊.bat").write_bytes(
        ("@echo off\r\nchcp 65001 >nul\r\ncd /d %~dp0\r\n"
        "echo ========================================\r\n"
        "echo   毕设工坊 ThesisForge  桌面窗口\r\n"
        "echo   关闭窗口即停止服务\r\n"
        "echo ========================================\r\n"
        "ThesisForge.exe\r\n"
        "pause\r\n").encode("utf-8"))
    (PKG / "安装图像训练-CPU版.bat").write_bytes(
        ("@echo off\r\nchcp 65001 >nul\r\ncd /d %~dp0\r\n"
         "echo 正在安装 CPU 版 PyTorch（约 200MB，需要网络）...\r\n"
         "runtime\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\r\n"
         "pause\r\n").encode("utf-8"))
    (PKG / "安装图像训练-GPU版.bat").write_bytes(
        ("@echo off\r\nchcp 65001 >nul\r\ncd /d %~dp0\r\n"
         "echo 正在安装 CUDA 12.6 版 PyTorch（约 2.5GB，需要网络与 NVIDIA 显卡）...\r\n"
         "runtime\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126\r\n"
         "pause\r\n").encode("utf-8"))
    (PKG / "使用说明.txt").write_text(
        f"""毕设工坊 ThesisForge v{APP_VERSION} — Windows 离线整合包
================================================

【快速开始】
1. 双击「ThesisForge.exe」，默认打开 Windows 桌面窗口；
   第一次启动需等待几秒解包，不会弹黑色终端；
   若系统未装 WebView2，会自动改用浏览器打开
2. 或双击「启动毕设工坊.bat」（同样是桌面窗口，关闭窗口即停止）
3. 首次使用会弹出新手引导，照着总览页的清单做即可
4. 想直接使用浏览器而非桌面窗口，可运行：ThesisForge.exe --browser
5. 想让服务在后台无界面运行，可加参数：ThesisForge.exe --no-browser

【启动出问题怎么办】
双击后若窗口没出现，查看本目录 data/logs/launch.log；
致命错误会弹出系统提示框，并写明日志位置。

【图像分类训练】
本包为控制体积未预装 PyTorch。需要图像训练时：
- NVIDIA 显卡：双击「安装图像训练-GPU版.bat」
- 无独立显卡：双击「安装图像训练-CPU版.bat」
表格/文本任务无需安装，开箱即用。

【数据与隐私】
所有数据、实验记录、AI Key 均保存在本目录 data/ 下，不上传任何服务器。
如需备份或迁移，拷贝整个文件夹即可。

【AI 分析配置】
在「AI 设置」页填入 OpenAI 兼容接口（智谱 glm-4-flash 有免费额度）。
不配置也能用：内置规则分析器会生成基础分析。

【新手文档】
docs 目录内含新手使用手册、目标检测、时间序列、自动调参和模型解释性指南。

【注意】
- 请勿使用 360 等软件"清理"本目录的 runtime 文件夹
- 端口 8765 被占用时，程序会自动改用 8766-8785 的可用端口，实际地址以窗口显示为准
""", encoding="utf-8")

    # 5. 压缩
    step("压缩为 zip ...")
    zpath = DIST / f"ThesisForge-v{APP_VERSION}-win64-offline.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for base, _dirs, files in os.walk(PKG):
            for f in files:
                fp = Path(base) / f
                zf.write(fp, fp.relative_to(DIST))
    size_mb = zpath.stat().st_size / 1024 / 1024
    step(f"完成: {zpath}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
