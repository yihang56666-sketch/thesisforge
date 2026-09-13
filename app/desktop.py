"""桌面/浏览器/无头三种启动形态。

独立 EXE 与离线整合包默认进入 pywebview 桌面窗口；WebView2 缺失或
pywebview 启动失败时自动降级为外部浏览器。本地服务跑在线程中，
窗口关闭后置 should_exit 并 join，保证进程能干净退出。
"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser

from .config import APP_VERSION, DATA_DIR, WEB_DIR
from .main import app

try:
    import webview  # pyproject: pywebview；未安装时桌面模式自动降级
except Exception:  # pragma: no cover - 依赖缺失分支
    webview = None

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


def _banner(url: str, mode: str) -> None:
    print(f"毕设工坊 ThesisForge v{APP_VERSION}  ->  {url}  [{mode}]", flush=True)
    print(f"数据目录: {DATA_DIR}", flush=True)
    if not WEB_DIR.exists():
        print("警告：找不到 web/ 静态资源目录，界面将无法显示。", flush=True)


def start_server_thread(host: str = "127.0.0.1", port: int | None = None) -> dict:
    """在线程中启动 uvicorn，等待健康检查就绪后返回服务句柄。"""
    import uvicorn

    from .main import find_free_port

    port = int(port or os.environ.get("TF_PORT") or 8765)
    port = find_free_port(port)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="tf-uvicorn", daemon=True)
    thread.start()

    deadline = time.time() + 20
    while time.time() < deadline and not server.started and not server.should_exit:
        time.sleep(0.1)
    if server.should_exit and not server.started:
        thread.join(timeout=5)
        raise RuntimeError("本地服务启动失败，请检查 8765-8785 端口与运行环境")
    if httpx is not None:
        url = f"http://{host}:{port}/api/health"
        for _ in range(40):
            try:
                if httpx.get(url, timeout=0.5).status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
    return {"server": server, "thread": thread, "port": port, "url": f"http://{host}:{port}/"}


def stop_server(info: dict) -> None:
    server = info["server"]
    thread = info["thread"]
    server.should_exit = True
    thread.join(timeout=10)


class DesktopBridge:
    """暴露给 pywebview 窗口的少量只读能力，不接触任何用户凭据。"""

    def __init__(self, url: str):
        self.url = url

    def version(self) -> str:
        return APP_VERSION

    def open_external_browser(self) -> dict:
        webbrowser.open(self.url)
        return {"ok": True}


def run_desktop(host: str = "127.0.0.1", port: int | None = None) -> int:
    info = start_server_thread(host, port)
    url = info["url"]
    _banner(url, "desktop")
    if webview is None:
        print("未安装 pywebview，改用浏览器打开。", flush=True)
        return run_browser(port=info["port"], info=info)
    try:
        webview.create_window(
            "毕设工坊 ThesisForge",
            url,
            width=1280,
            height=820,
            min_size=(960, 640),
            js_api=DesktopBridge(url),
        )
        webview.start(private_mode=False)
    except Exception as e:  # WebView2 缺失等场景
        print(f"桌面窗口启动失败（{e}），改用浏览器打开。", flush=True)
        try:
            if callable(getattr(webview, "destroy", None)):
                webview.destroy()
        except Exception:
            pass
        return run_browser(port=info["port"], info=info)
    stop_server(info)
    return 0


def run_browser(host: str = "127.0.0.1", port: int | None = None, info: dict | None = None) -> int:
    info = info or start_server_thread(host, port)
    url = info["url"]
    _banner(url, "browser")
    if os.environ.get("THESISFORGE_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        info["thread"].join()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server(info)
    return 0


def run_headless(host: str = "127.0.0.1", port: int | None = None) -> int:
    info = start_server_thread(host, port)
    _banner(info["url"], "headless")
    try:
        info["thread"].join()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server(info)
    return 0


def resolve_startup_mode(argv: list[str] | None = None) -> str:
    """解析启动形态：--browser / --no-window / THESISFORGE_MODE / frozen。"""
    args = {str(a).lower() for a in (argv if argv is not None else sys.argv[1:])}
    if "--browser" in args:
        return "browser"
    if "--no-window" in args or "--no-browser" in args or "--headless" in args:
        return "headless"
    env_mode = os.environ.get("THESISFORGE_MODE", "").strip().lower()
    if env_mode == "desktop":
        return "desktop"
    if getattr(sys, "frozen", False):
        return "desktop"
    return "browser"


if __name__ == "__main__":
    mode = resolve_startup_mode()
    if mode == "headless":
        sys.exit(run_headless())
    if mode == "browser":
        sys.exit(run_browser())
    sys.exit(run_desktop())
