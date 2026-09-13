"""毕设工坊后端：FastAPI 路由 + 静态页面服务。

安全约定：
- 出站下载请求必须经过 app.security.validate_public_http_url（拒绝内网/保留地址）；
- 所有凭据仅来自环境变量或用户运行时输入，接口返回时对 Key 做脱敏；
- 实验/数据集 ID 拼接路径前一律校验。
"""
from __future__ import annotations

import json
import importlib.util
import os
import re
import shutil
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ai, catalog, datasets_hub, experiments, humanize, report, runner
from .config import (
    APP_VERSION, DATA_DIR, DATASETS_DIR, EXPORTS_DIR, RUNS_DIR, WEB_DIR,
    ensure_dirs, load_runtime_config, resolve_api_key, save_runtime_config,
)
from .config import UPLOADS_DIR
from .security import SafeURLError

ensure_dirs()

app = FastAPI(title="ThesisForge 毕设工坊", version=APP_VERSION)

_DS_ID_RE = re.compile(r"^[0-9a-zA-Z\u4e00-\u9fff\-]{1,80}$")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", ""}


def safe_report_filename(title: str) -> str:
    """去掉不能用于 Windows 文件名的字符并截断，空标题回退为「报告」。"""
    return (re.sub(r"[^\w\u4e00-\u9fff-]+", "", (title or "").strip())[:40] or "报告")


def _host_of(value: str) -> str:
    """从 Host / Origin 中取出纯主机名（去端口，保留 IPv6 方括号）。"""
    v = (value or "").strip().lower()
    if not v:
        return ""
    if v.startswith("["):                      # IPv6 字面量，如 [::1]:8765
        end = v.find("]")
        return v[: end + 1] if end != -1 else v
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


@app.middleware("http")
async def local_origin_guard(request: Request, call_next):
    """本机服务防护：Host 必须是回环地址（挡 DNS rebinding），
    写操作的 Origin 必须与本机同源（挡浏览器发起的跨站 POST/DELETE）。"""
    host = _host_of(request.headers.get("host") or "")
    if host not in _LOCAL_HOSTS:
        return JSONResponse(status_code=403, content={"detail": "拒绝访问：仅允许本机访问该服务"})
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        origin = request.headers.get("origin") or ""
        if origin:
            from urllib.parse import urlparse

            oh = _host_of(urlparse(origin).netloc)
            if oh not in _LOCAL_HOSTS:
                return JSONResponse(status_code=403, content={"detail": "拒绝访问：来源不是本机"})
    return await call_next(request)


def ds_dir_of(ds_id: str) -> Path:
    if not _DS_ID_RE.match(str(ds_id or "")):
        raise HTTPException(400, "非法的数据集 ID")
    p = (DATASETS_DIR / ds_id).resolve()
    if not p.is_relative_to(DATASETS_DIR.resolve()) or p == DATASETS_DIR.resolve():
        raise HTTPException(400, "非法的数据集路径")
    return p


def json_err(e: Exception, code: int = 400):
    return JSONResponse(status_code=code, content={"detail": str(e)})


# ================================================================ 基础
@app.get("/api/health")
def health():
    info = {"ok": True, "version": APP_VERSION, "cuda": None}
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        info["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import sklearn

        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return info


# ================================================================ AI 设置
class LLMConfig(BaseModel):
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_max_tokens: int | None = None
    llm_temperature: float | None = None


@app.get("/api/config")
def get_config():
    cfg = load_runtime_config()
    key = resolve_api_key(cfg)
    return {
        "llm_base_url": cfg.get("llm_base_url", ""),
        "llm_model": cfg.get("llm_model", ""),
        "llm_api_key_env": cfg.get("llm_api_key_env", "LLM_API_KEY"),
        "env_key_present": bool((__import__("os").environ.get(cfg.get("llm_api_key_env") or "LLM_API_KEY") or "").strip()),
        "stored_key_present": bool((cfg.get("llm_api_key") or "").strip()),
        "key_masked": (key[:4] + "****" + key[-4:]) if len(key) > 8 else ("已配置" if key else ""),
        "configured": ai.is_configured(),
        "llm_max_tokens": cfg.get("llm_max_tokens"),
        "llm_temperature": cfg.get("llm_temperature"),
    }


@app.post("/api/config")
def set_config(body: LLMConfig):
    update = body.model_dump()
    if update.get("llm_api_key") == "":
        update["llm_api_key"] = ""  # 用户显式清空
    cfg = save_runtime_config(update)
    return {"ok": True, "configured": ai.is_configured(), "key_masked": ("已配置" if resolve_api_key(cfg) else "")}


@app.post("/api/config/test")
async def test_llm():
    try:
        r = await ai.test_connection()
        return {"ok": True, "reply": r["reply"]}
    except ai.AIError as e:
        return JSONResponse(status_code=400, content={"ok": False, "detail": str(e)})


# ================================================================ 数据集
@app.get("/api/datasets")
def list_datasets():
    return {"datasets": datasets_hub.list_datasets(), "builtin": [
        {"key": k, "name": v["name"], "task": v["task"], "desc": v["desc"]} for k, v in datasets_hub.BUILTIN_CATALOG.items()
    ]}


class BuiltinReq(BaseModel):
    name: str


@app.post("/api/datasets/builtin")
def load_builtin(req: BuiltinReq):
    try:
        meta = datasets_hub.load_builtin(req.name)
        return {"ok": True, "dataset": meta}
    except Exception as e:
        raise HTTPException(400, f"载入失败: {e}")


@app.post("/api/datasets/import")
async def import_dataset(file: UploadFile = File(...)):
    # 分块落盘并计数，避免超大文件被整体读进内存
    suffix = Path(file.filename or "").suffix.lower()
    tmp = UPLOADS_DIR / f"upload-{int(time.time() * 1000)}-{os.getpid()}{suffix}"
    total = 0
    try:
        with tmp.open("wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > datasets_hub.MAX_DOWNLOAD_BYTES:
                    raise HTTPException(400, "文件超过 1GB 限制")
                out.write(chunk)
        if total == 0:
            raise HTTPException(400, "上传文件为空")
        meta = datasets_hub.import_path(file.filename or "uploaded.csv", tmp)
        return {"ok": True, "dataset": meta}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)


class DownloadReq(BaseModel):
    url: str
    name: str | None = None


@app.post("/api/datasets/download")
def download_dataset(req: DownloadReq):
    try:
        meta = datasets_hub.download_url(req.url, req.name)
        return {"ok": True, "dataset": meta}
    except SafeURLError as e:
        raise HTTPException(400, f"链接被安全策略拒绝: {e}")
    except Exception as e:
        raise HTTPException(400, f"下载失败: {e}")


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str):
    try:
        datasets_hub.delete_dataset(ds_id)
        return {"ok": True}
    except FileNotFoundError:
        raise HTTPException(404, "数据集不存在")


@app.get("/api/datasets/{ds_id}")
def dataset_detail(ds_id: str):
    try:
        return datasets_hub.load_meta(ds_id)
    except FileNotFoundError:
        raise HTTPException(404, "数据集不存在")


@app.get("/api/datasets/{ds_id}/preview")
def dataset_preview(ds_id: str, rows: int = 20):
    try:
        return datasets_hub.preview(ds_id, min(rows, 100))
    except FileNotFoundError:
        raise HTTPException(404, "数据集不存在")


@app.get("/api/datasets/{ds_id}/eda/{fname}")
def dataset_eda_file(ds_id: str, fname: str):
    p = ds_dir_of(ds_id) / "eda" / fname
    p = p.resolve()
    if not p.is_relative_to(ds_dir_of(ds_id).resolve()) or p.suffix != ".png" or not p.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(p, media_type="image/png")


class AnalyzeReq(BaseModel):
    pass


@app.post("/api/datasets/{ds_id}/analyze")
async def analyze_dataset(ds_id: str):
    try:
        meta = datasets_hub.load_meta(ds_id)
    except FileNotFoundError:
        raise HTTPException(404, "数据集不存在")
    r = await ai.analyze_dataset(meta)
    # 保存分析记录
    try:
        (ds_dir_of(ds_id) / "ai_analysis.md").write_text(r["text"], encoding="utf-8")
    except Exception:
        pass
    return r


# ================================================================ 模型目录
@app.get("/api/models")
def get_models():
    return {"catalog": catalog.CATALOG}


# ================================================================ 实验
class CreateRunReq(BaseModel):
    dataset_id: str
    task: str
    model: str
    params: dict = {}
    target: str | None = None
    text_column: str | None = None
    test_size: float = 0.2
    random_state: int = 42
    val_split: float = 0.2
    name: str | None = None
    group: str | None = None
    note: str | None = None


@app.post("/api/runs")
def create_run(req: CreateRunReq):
    if req.task not in catalog.CATALOG:
        raise HTTPException(400, "未知任务类型")
    spec = catalog.get_model_spec(req.task, req.model)
    if not spec:
        raise HTTPException(400, "未知模型")
    try:
        ds_meta = datasets_hub.load_meta(req.dataset_id)
    except FileNotFoundError:
        raise HTTPException(404, "数据集不存在")

    task = req.task
    if task == "tabular_classification" and ds_meta["type"] != "tabular":
        raise HTTPException(400, "该数据集不是表格数据")
    if task == "image_classification" and ds_meta["type"] != "image":
        raise HTTPException(400, "该任务需要图像数据集（zip，类别文件夹结构）")
    if task == "text_classification":
        if ds_meta["type"] != "tabular":
            raise HTTPException(400, "文本分类需要包含文本列与标签列的表格数据(CSV)")
        if not req.text_column or req.text_column == req.target:
            raise HTTPException(400, "请在高级选项中选择文本列与标签列")

    script = "train_torch.py" if task == "image_classification" else "train_sklearn.py"
    if script == "train_torch.py":
        # 图像训练依赖 PyTorch；离线整合包与精简 EXE 可能未安装，提前给出可操作的提示
        try:
            has_torch = importlib.util.find_spec("torch") is not None
        except (ImportError, ValueError):
            has_torch = False
        if not has_torch:
            raise HTTPException(400, "未检测到 PyTorch，无法进行图像分类训练。请运行「安装图像训练-CPU版.bat」"
                                    "（或 GPU 版），或改用表格/文本任务。")
    # 标签列兜底：未指定时使用最后一列
    target = req.target if task != "image_classification" else None
    if task != "image_classification" and not target:
        if ds_meta.get("target"):
            target = ds_meta["target"]
        elif ds_meta.get("columns"):
            target = ds_meta["columns"][-1]
        else:
            raise HTTPException(400, "数据集缺少标签列，请先重新导入")
        if ds_meta.get("columns") and target not in ds_meta["columns"]:
            raise HTTPException(400, f"标签列 {target} 不在数据集中")
    # 参数裁剪：只保留目录中定义的键，并做类型/范围约束
    # 白名单 + 类型 + 有限性 + min/max 夹取，全部收在 catalog.sanitize_params 里，
    # 这样直接打 API 也拿不到 epochs=99999 这类能把机器跑挂的参数。
    clean_params = catalog.sanitize_params(spec, req.params)

    ds_dir = ds_dir_of(req.dataset_id)
    config = {
        "name": (req.name or "").strip()[:80],
        "group": req.group if req.group in runner.GROUPS else runner.GROUP_BASELINE,
        "note": (req.note or "").strip()[:500],
        "dataset_id": req.dataset_id,
        "dataset_name": ds_meta.get("name"),
        "dataset_dir": str(ds_dir),
        "task": task,
        "model": req.model,
        "model_label": spec["label"],
        "params": clean_params,
        "target": target,
        "text_column": req.text_column,
        "test_size": min(max(float(req.test_size), 0.05), 0.5),
        "val_split": min(max(float(req.val_split), 0.05), 0.5),
        "random_state": int(req.random_state),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    run_id = runner.create_run(config, script)
    return {"ok": True, "run_id": run_id}


@app.get("/api/runs")
def list_runs():
    return {"runs": runner.list_runs()}


class RunMetaReq(BaseModel):
    name: str | None = None
    group: str | None = None
    note: str | None = None


@app.patch("/api/runs/{run_id}/meta")
def update_run_meta(run_id: str, req: RunMetaReq):
    try:
        meta = runner.update_meta(run_id, req.model_dump(exclude_none=True))
        return {"ok": True, "meta": meta}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "实验不存在")


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    try:
        return runner.run_detail(run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "实验不存在")


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str):
    try:
        return {"log": runner.tail_log(run_id, 60000)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    try:
        ok = runner.cancel_run(run_id)
        return {"ok": ok}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    try:
        d = runner.run_dir_of(run_id)
        runner.cancel_run(run_id)
        time.sleep(0.3)
        shutil.rmtree(d, ignore_errors=True)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/runs/{run_id}/artifacts/{fname}")
def run_artifact(run_id: str, fname: str):
    try:
        d = runner.run_dir_of(run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    p = (d / fname).resolve()
    if not p.is_relative_to(d.resolve()) or p.suffix not in (".png", ".pt", ".pkl", ".json") or not p.exists():
        raise HTTPException(404, "文件不存在")
    media = "image/png" if p.suffix == ".png" else "application/octet-stream"
    return FileResponse(p, media_type=media, filename=p.name)


@app.post("/api/runs/{run_id}/analyze")
async def analyze_run(run_id: str):
    try:
        detail = runner.run_detail(run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "实验不存在")
    meta = {
        "dataset_name": detail["config"].get("dataset_name"),
        "task": detail["config"].get("task"),
        "model_label": detail["config"].get("model_label"),
        "params": detail["config"].get("params"),
        "test_size": detail["config"].get("test_size"),
        "n_train": (detail["summary"] or {}).get("n_train"),
        "n_val": (detail["summary"] or {}).get("n_val"),
        "n_test": (detail["summary"] or {}).get("n_test"),
        "split_scheme": (detail["summary"] or {}).get("split_scheme"),
        "metrics": (detail["summary"] or {}).get("metrics"),
        "cv_scores": (detail["summary"] or {}).get("cv_scores"),
        "epochs": (detail["summary"] or {}).get("epochs"),
        "primary_metric": (detail["summary"] or {}).get("primary_metric"),
    }
    r = await ai.analyze_run(meta)
    try:
        (runner.run_dir_of(run_id) / "ai_analysis.md").write_text(r["text"], encoding="utf-8")
    except Exception:
        pass
    return r


# ================================================================ 实验对比
class CompareReq(BaseModel):
    run_ids: list[str] = []


@app.post("/api/experiments/compare")
def compare_experiments(req: CompareReq):
    try:
        return experiments.build_comparison(req.run_ids)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ================================================================ AIGC 自检与降 AI 味
class HumanizeCheckReq(BaseModel):
    text: str


class HumanizeRewriteReq(BaseModel):
    text: str
    use_llm: bool = False


@app.post("/api/humanize/check")
def humanize_check(req: HumanizeCheckReq):
    if len(req.text) > 30000:
        raise HTTPException(400, "文本过长（限 3 万字符）")
    return humanize.score_ai_flavor(req.text)


@app.post("/api/humanize/rewrite")
async def humanize_rewrite(req: HumanizeRewriteReq):
    if len(req.text) > 30000:
        raise HTTPException(400, "文本过长（限 3 万字符）")
    try:
        result = humanize.humanize_text(req.text)
    except Exception as e:
        raise HTTPException(500, f"处理失败: {e}")
    llm_note = None
    if req.use_llm:
        if not ai.is_configured():
            raise HTTPException(400, "深度改写需要先配置 AI 接口")
        try:
            rewritten = await humanize.llm_rewrite(result["text"])
            result["text"] = rewritten
            result["changes"].append("LLM 深度改写（保持事实与数字不变）")
            result["score_after"] = humanize.score_ai_flavor(rewritten)
            llm_note = "ok"
        except ai.AIError as e:
            llm_note = f"LLM 改写失败，仅保留规则改写结果：{e}"
    return result | {"llm_note": llm_note}


# ================================================================ 报告
class ReportReq(BaseModel):
    title: str = "基于机器学习的毕业设计研究"
    run_ids: list[str] = []
    dataset_id: str | None = None
    author: dict = {}
    ai_draft: bool = False


@app.post("/api/report/generate")
async def generate_report(req: ReportReq):
    runs = []
    for rid in req.run_ids[:6]:
        try:
            d = runner.run_detail(rid)
        except (ValueError, FileNotFoundError):
            continue
        if d["state"] != "done" or not d["summary"]:
            continue
        runs.append({"run_id": rid, "config": d["config"], "summary": d["summary"],
                     "run_dir": str(RUNS_DIR / rid),
                     "name": d["config"].get("name"),
                     "group": d["config"].get("group") or runner.GROUP_BASELINE})

    dataset_meta, dataset_dir = None, None
    if req.dataset_id:
        try:
            dataset_meta = datasets_hub.load_meta(req.dataset_id)
            dataset_dir = ds_dir_of(req.dataset_id)
        except (FileNotFoundError, HTTPException):
            pass

    drafts: dict[str, str] = {}
    if req.ai_draft and ai.is_configured():
        ctx = {
            "title": req.title,
            "dataset": {k: dataset_meta.get(k) for k in ("name", "task", "n_rows", "desc")} if dataset_meta else None,
            "runs": [{"model": (r["summary"] or {}).get("model_label"), "metrics": (r["summary"] or {}).get("metrics"),
                      "params": r["config"].get("params"),
                      "primary_metric": (r["summary"] or {}).get("primary_metric"),
                      "split_scheme": (r["summary"] or {}).get("split_scheme")} for r in runs],
            "写作要求": "实验结果段落中必须区分验证集与测试集：test_* 指标称『测试集』，val_* 指标称『验证集』，不得混用。",
        }
        sections = [
            ("abstract", "摘要"), ("background", "第一章 研究背景与意义"),
            ("content", "第一章 研究内容"), ("related", "第二章 相关技术基础"),
            ("preprocess", "第三章 数据预处理"), ("conclusion", "第五章 总结与展望"),
        ]
        for r in runs:
            sections.append((f"analysis:{r['run_id']}", f"第四章 { (r['summary'] or {}).get('model_label', '') }实验结果分析"))
        import asyncio

        async def one(key, t):
            return key, await ai.draft_section(key, t, ctx)

        results = await asyncio.gather(*[one(k, t) for k, t in sections])
        for k, text in results:
            if text:
                # AI 起草的正文自动做一次规则去 AI 味（改写措辞与节奏，不动事实）
                try:
                    text = humanize.humanize_text(text)["text"]
                except Exception:
                    pass
                drafts[k] = text

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = EXPORTS_DIR / f"{safe_report_filename(req.title)}-{ts}.docx"
    try:
        path = report.build_report(out, req.title, req.author or {}, dataset_meta, dataset_dir, runs, drafts)
    except Exception as e:
        raise HTTPException(500, f"报告生成失败: {e}")
    return {"ok": True, "path": str(path), "filename": path.name, "ai_sections": sorted(drafts.keys())}


@app.get("/api/report/list")
def list_reports():
    files = sorted(
        ({"filename": p.name, "size": p.stat().st_size, "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))}
         for p in EXPORTS_DIR.glob("*.docx")),
        key=lambda x: x["created_at"], reverse=True,
    )
    return {"reports": files}


@app.get("/api/report/download")
def download_report(filename: str):
    p = (EXPORTS_DIR / filename).resolve()
    if not p.is_relative_to(EXPORTS_DIR.resolve()) or p.suffix != ".docx" or not p.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(p, filename=p.name)


# ================================================================ 静态页面
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# ================================================================ 直接运行入口
def find_free_port(preferred: int = 8765) -> int:
    import socket

    for port in [preferred] + [preferred + i for i in range(1, 21)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("8765-8785 端口都被占用，请释放一个后重试")


def run_server(host: str = "127.0.0.1", port: int | None = None, open_browser: bool = True) -> None:
    """启动本地服务。打包成 EXE 后这就是双击后的行为。"""
    import threading
    import webbrowser

    import uvicorn

    port = int(port or os.environ.get("TF_PORT") or 8765)
    port = find_free_port(port)
    url = f"http://{host}:{port}/"
    print(f"毕设工坊 ThesisForge v{APP_VERSION}  ->  {url}", flush=True)
    print(f"数据目录: {DATA_DIR}", flush=True)
    if not WEB_DIR.exists():
        print("警告：找不到 web/ 静态资源目录，界面将无法显示。", flush=True)
    # 只做 finder 查询，不真的 import torch（那是好几秒的 DLL 加载）
    if importlib.util.find_spec("torch") is not None:
        print("图像分类：已检测到 PyTorch，可用。", flush=True)
    else:
        print("图像分类：未安装 PyTorch，仅支持表格/文本任务（图像训练请用离线整合包）。", flush=True)
    if open_browser and os.environ.get("THESISFORGE_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n已退出。", flush=True)


def main(argv: list[str] | None = None) -> int:
    """统一启动入口：解析 --browser/--no-window/THESISFORGE_MODE 后启动对应形态。"""
    from .desktop import resolve_startup_mode, run_browser, run_desktop, run_headless

    mode = resolve_startup_mode(argv if argv is not None else sys.argv[1:])
    if mode == "headless":
        run_headless()
    elif mode == "browser":
        run_browser()
    else:
        run_desktop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
