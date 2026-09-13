"""AI 分析模块。

- 通过 httpx 调用 OpenAI 兼容接口（DeepSeek / 智谱 GLM / 通义千问 / OpenAI 等）；
- API Key 从环境变量或面板配置读取，源码不写入任何凭据；
- 未配置 Key 时自动降级为内置规则分析器，仍然产出可用的中文分析报告。
"""
from __future__ import annotations

import json

import httpx

from .config import load_runtime_config, resolve_api_key


class AIError(RuntimeError):
    pass


def _endpoint(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise AIError("未配置 AI 接口地址（base_url）")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def is_configured() -> bool:
    cfg = load_runtime_config()
    return bool(cfg.get("llm_base_url") and cfg.get("llm_model") and resolve_api_key(cfg))


async def chat(messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
    cfg = load_runtime_config()
    key = resolve_api_key(cfg)
    if not key:
        raise AIError("未配置 API Key：请在「AI 设置」页填写，或设置环境变量 LLM_API_KEY")
    payload = {
        "model": cfg["llm_model"],
        "messages": messages,
        "max_tokens": int(max_tokens or cfg.get("llm_max_tokens") or 2000),
        "temperature": float(temperature if temperature is not None else cfg.get("llm_temperature", 0.4)),
    }
    url = _endpoint(cfg.get("llm_base_url", ""))
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.HTTPError as e:
        raise AIError(f"连接 AI 接口失败: {e.__class__.__name__}: {e}")
    if resp.status_code == 401:
        raise AIError("AI 接口返回 401：API Key 无效")
    if resp.status_code == 404:
        raise AIError("AI 接口返回 404：请检查 base_url 是否正确（通常需要以 /v1 结尾）")
    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise AIError(f"AI 接口错误 {resp.status_code}: {detail}")
    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise AIError(f"AI 接口返回格式异常: {resp.text[:300]}")


async def test_connection() -> dict:
    cfg = load_runtime_config()
    if not cfg.get("llm_base_url") or not cfg.get("llm_model"):
        raise AIError("请先填写 base_url 与模型名")
    reply = await chat([{"role": "user", "content": "请只回复两个字：正常"}], max_tokens=16, temperature=0.0)
    return {"reply": reply.strip()[:50]}


# ================================================================ 提示词构建
def _dataset_prompt(meta: dict) -> list[dict]:
    stats = meta.get("stats") or {}
    compact = {
        "名称": meta.get("name"), "类型": meta.get("type"), "任务": meta.get("task"),
        "行数": meta.get("n_rows"), "类别数": meta.get("n_classes"),
        "类别分布": stats.get("class_counts"), "缺失值": stats.get("missing"),
        "列类型": stats.get("column_types"),
    }
    return [
        {"role": "system", "content": "你是一位人工智能专业毕业设计指导老师，擅长数据分析与建模指导。请用中文、条理清晰地回答，控制在 500 字以内，使用 Markdown 小标题与列表。"},
        {"role": "user", "content": (
            "这是我毕设使用的数据集概况（JSON）：\n" + json.dumps(compact, ensure_ascii=False) +
            "\n\n请从以下角度分析：1) 数据规模与类别分布是否适合建模，是否存在类别不平衡；"
            "2) 缺失值与特征类型需要的预处理；3) 适合选用的模型与评价方法；4) 做毕业设计时这一步要注意什么（如数据划分防泄漏）。"
        )},
    ]


def _run_prompt(meta: dict) -> list[dict]:
    compact = {
        "数据集": meta.get("dataset_name"), "任务": meta.get("task"), "模型": meta.get("model_label"),
        "超参数": meta.get("params"), "测试集占比": meta.get("test_size"),
        "指标": meta.get("metrics"), "交叉验证": meta.get("cv_scores"),
        "训练轮曲线末尾": (meta.get("epochs") or [])[-5:],
    }
    return [
        {"role": "system", "content": "你是一位人工智能专业毕业设计指导老师。请用中文回答，控制在 600 字以内，使用 Markdown 小标题与列表，语气务实、直指问题。"},
        {"role": "user", "content": (
            "这是我一次模型训练实验的结果（JSON）：\n" + json.dumps(compact, ensure_ascii=False) +
            "\n\n请分析：1) 主要指标说明模型表现如何；2) 是否存在过拟合/欠拟合或类别不平衡导致的虚假高分；"
            "3) 给出 3-5 条具体可执行的改进建议（算法、超参、数据、评估四个层面）；4) 这些结果如何写进论文实验章节。"
        )},
    ]


# ================================================================ 规则降级分析
def _fmt_pct(x) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


def fallback_dataset_analysis(meta: dict) -> str:
    stats = meta.get("stats") or {}
    lines = ["> 当前未配置 AI 接口，以下为内置规则分析器生成的分析。在「AI 设置」页配置后可获得更深入的解读。\n"]
    lines.append("## 数据概况")
    lines.append(f"- 数据集：{meta.get('name')}，类型：{meta.get('type')}，任务：{meta.get('task')}")
    if meta.get("n_rows"):
        lines.append(f"- 样本量：{meta['n_rows']} 行 × {len(meta.get('columns') or [])} 列")
    counts = stats.get("class_counts") or {}
    if counts:
        vals = list(counts.values())
        ratio = max(vals) / max(min(vals), 1)
        lines.append(f"- 类别数：{len(counts)}，最多/最少类样本比 ≈ {ratio:.1f}:1")
        if ratio >= 3:
            lines.append(f"- 存在类别不平衡（{ratio:.1f}:1）：建议使用 class_weight、过采样(如 SMOTE) 或按类别宏平均 F1 评估，不要只看准确率。")
    missing = stats.get("missing") or {}
    miss_cols = {k: v for k, v in missing.items() if v}
    if miss_cols:
        lines.append(f"- 存在缺失值：{miss_cols}，建议中位数/众数填充或删除高缺失列。")
    else:
        lines.append("- 未检出缺失值，可跳过缺失处理步骤。")
    lines.append("\n## 建模建议")
    task = meta.get("task", "")
    if "regression" in task:
        lines.append("- 回归任务：先跑线性回归做基线，再对比随机森林/GBDT；评价指标用 R²、RMSE、MAE 组合呈现。")
    else:
        lines.append("- 分类任务：先跑逻辑回归/随机森林建立基线，再逐步尝试 GBDT、SVM、MLP；表格数据上 GBDT 常最优。")
    lines.append("- 数据划分：先切分再标准化（工具已按此实现），固定 random_state 保证实验可复现；论文中报告测试集与交叉验证两组结果。")
    lines.append("- 下一步：进入「模型训练」页选择模型，先跑一版基线，再在「AI 分析」中对照改进。")
    return "\n".join(lines)


def fallback_run_analysis(meta: dict) -> str:
    metrics = meta.get("metrics") or {}
    lines = ["> 当前未配置 AI 接口，以下为内置规则分析器生成的分析。配置后可获得更深入的解读。\n"]
    lines.append("## 结果解读")
    shown = False
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            shown = True
    if shown:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                pretty = _fmt_pct(v) if k in ("accuracy", "val_accuracy", "precision", "recall", "f1", "f1_macro", "f1_weighted", "r2") else f"{v:.4f}"
                lines.append(f"- {k} = {pretty}")
    cv = meta.get("cv_scores") or []
    if cv:
        import numpy as np

        mean, std = float(np.mean(cv)), float(np.std(cv))
        spread = (max(cv) - min(cv)) if cv else 0
        lines.append(f"- 交叉验证：均值 {mean:.4f} ± {std:.4f}，折间极差 {spread:.4f}")
        if spread > 0.05:
            lines.append("- 注意：各折得分波动较大，模型稳定性一般：可尝试更多数据、交叉验证调参或更强的正则化。")
    epochs = meta.get("epochs") or []
    if len(epochs) >= 3:
        tr_last, va_last = epochs[-1].get("train_acc"), epochs[-1].get("val_acc")
        if tr_last is not None and va_last is not None and tr_last - va_last > 0.08:
            lines.append(f"- 注意：训练准确率({_fmt_pct(tr_last)})明显高于验证({_fmt_pct(va_last)})，存在过拟合：建议数据增强、Dropout、权重衰减或更小模型。")
        if len(epochs) >= 5:
            tail = [e.get("val_acc") or 0 for e in epochs[-4:]]
            if max(tail) - min(tail) < 0.002:
                lines.append("- 验证准确率已收敛，继续增加轮数收益有限；可改试学习率调度或更大模型。")
    lines.append("\n## 改进建议（毕业设计通用路线）")
    lines.append("1. **基线对比**：至少再跑 1-2 个对照模型（逻辑回归/随机森林/GBDT 或 CNN vs ResNet），论文中需要对比表格。")
    lines.append("2. **超参数搜索**：围绕学习率、树数量/深度、正则化系数各取 3 个值做网格/随机搜索，记录每组结果。")
    lines.append("3. **数据层面**：检查类别不平衡（用宏平均 F1）、尝试特征选择或数据增强。")
    lines.append("4. **消融实验**：对你提出的改进点逐个开/关，证明每个模块的有效性——这是毕设答辩的核心证据。")
    lines.append("5. **可视化**：把本页的损失/准确率曲线、混淆矩阵图直接插入论文实验章节，并配文字解读。")
    return "\n".join(lines)


async def analyze_dataset(meta: dict) -> dict:
    if is_configured():
        try:
            text = await chat(_dataset_prompt(meta))
            return {"text": text, "source": "llm"}
        except AIError as e:
            return {"text": fallback_dataset_analysis(meta) + f"\n\n（AI 接口调用失败：{e}）", "source": "fallback+error", "error": str(e)}
    return {"text": fallback_dataset_analysis(meta), "source": "fallback"}


async def analyze_run(meta: dict) -> dict:
    if is_configured():
        try:
            text = await chat(_run_prompt(meta))
            return {"text": text, "source": "llm"}
        except AIError as e:
            return {"text": fallback_run_analysis(meta) + f"\n\n（AI 接口调用失败：{e}）", "source": "fallback+error", "error": str(e)}
    return {"text": fallback_run_analysis(meta), "source": "fallback"}


async def draft_section(section_key: str, section_title: str, context: dict) -> str:
    """为报告章节起草正文；失败时返回 None（报告使用模板文字）。"""
    if not is_configured():
        return None
    prompt = [
        {"role": "system", "content": "你是一位人工智能专业毕业论文写作者。请用中文学术语言撰写，避免口语，可使用『本文』作为第一人称，输出纯正文（Markdown 列表可用），不要输出标题编号。"},
        {"role": "user", "content": (
            f"请为毕业论文章节《{section_title}》撰写正文（300-500 字），可参考以下实验素材(JSON)：\n"
            + json.dumps(context, ensure_ascii=False)
        )},
    ]
    try:
        return await chat(prompt, max_tokens=1200)
    except AIError:
        return None
