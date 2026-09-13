# -*- coding: utf-8 -*-
"""毕设工坊 ThesisForge 打包入口（PyInstaller 用），一份源码支持三种形态。

1) 训练子进程：ThesisForge.exe --tf-worker train_torch.py --run-dir <目录>
   PyInstaller 下 sys.executable 指向 EXE 本身，直接再拉起会启动第二个 Web 服务。
   runner.worker_cmd 因此在 frozen 时改用 --tf-worker 让同一个 EXE 跑训练脚本。
2) 启动器模式：EXE 与 runtime\\python.exe 同包（离线整合包）。
   交给内嵌 Python 跑源码（THESISFORGE_MODE=desktop），功能完整
   （可自行安装 PyTorch 与 WebView2 做图像训练/桌面窗口）。
3) 独立模式：单文件 EXE，默认进入 pywebview 桌面窗口，WebView2
   缺失时自动降级为浏览器。为控制体积不打包 PyTorch，图像分类会
   返回明确提示，其余任务开箱即用。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from app.config import log_message, notify_fatal
except Exception:  # 启动器形态排除了 app 包，这里提供等效的独立实现
    def log_message(message: str) -> None:
        try:
            log_path = Path(sys.executable).resolve().parent / "data" / "logs" / "launch.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception:
            pass

    def notify_fatal(title: str, message: str) -> None:
        log_message(f"FATAL {title}: {message}")
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
        except Exception:
            pass


def _worker(argv: list[str]) -> int:
    """在同一 EXE 进程内以 __main__ 方式运行 app.train_* 训练脚本。"""
    import runpy

    allowed = {"train_sklearn.py", "train_torch.py"}
    name = argv[0] if argv else ""
    if name not in allowed:
        print(f"[ThesisForge] 非法的训练模块参数：{name!r}", file=sys.stderr)
        return 2
    module = "app." + name[:-3]
    try:
        import importlib.util

        if importlib.util.find_spec(module) is None:
            raise ImportError(module)
    except ImportError:
        print(f"[ThesisForge] 本程序未包含 {module}，无法执行该训练任务。", file=sys.stderr)
        return 3
    # 让训练脚本的 argparse 看到与原 `python app/train_x.py --run-dir ...` 一致的参数
    sys.argv = [name] + argv[1:]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    except BaseException as e:  # noqa: BLE001 子进程异常必须留下可读日志
        import traceback

        traceback.print_exc()
        print(f"[ThesisForge] 训练进程异常：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def _embedded_runtime(exe_dir: Path):
    """离线整合包形态：同级 runtime/python.exe（或 python）存在时返回其路径。"""
    for cand in ("python.exe", "python"):
        p = exe_dir / "runtime" / cand
        if p.exists():
            return p
    return None


def _run_with_runtime(runtime: Path, exe_dir: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(exe_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["THESISFORGE_MODE"] = "desktop"
    if "--no-browser" in sys.argv[1:]:
        env["THESISFORGE_NO_BROWSER"] = "1"
    cmd = [str(runtime), "-u", "-m", "app.main"] + sys.argv[1:]
    log_message(f"launcher start runtime={runtime} args={' '.join(sys.argv[1:])}")
    try:
        return subprocess.call(cmd, cwd=str(exe_dir), env=env)
    except OSError as e:
        log_message(f"launcher fatal: cannot start embedded python: {e}")
        notify_fatal("毕设工坊启动失败", f"无法启动内嵌 Python：{e}")
        return 1


def _banner() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> int:
    _banner()
    argv = sys.argv[1:]
    if argv and argv[0] == "--tf-worker":
        return _worker(argv[1:])

    exe_dir = Path(sys.executable).resolve().parent
    runtime = _embedded_runtime(exe_dir) if getattr(sys, "frozen", False) else None
    if runtime is not None:
        return _run_with_runtime(runtime, exe_dir)

    try:
        from app.main import main
    except ImportError as e:
        log_message(f"launcher fatal: runtime modules incomplete: {e}")
        notify_fatal("毕设工坊启动失败", f"运行库不完整，无法启动：{e}")
        if not getattr(sys, "frozen", False):
            print("开发环境请使用：python -m app.main", file=sys.stderr)
        return 1
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
