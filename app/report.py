"""报告工坊：将数据集、实验结果与 AI 撰写的章节文本组装为 .docx 论文初稿。

排版规格（常见本科毕设要求的最大公约数，可按学校模板微调）：
- A4；页边距 上/下 2.6cm、左 3.0cm、右 2.5cm；
- 正文：宋体（西文 Times New Roman）小四 12pt，1.5 倍行距，首行缩进 2 字符；
- 一级标题：黑体三号 16pt 居中；二级：黑体四号 14pt；三级：黑体小四 13pt；
- 图注/表注：宋体五号 10.5pt 居中，图注在下、表注在上；表格用三线表；
- 页眉：校名/论文标识 宋体小五；页脚：页码域（宋体五号居中），封面不显示；
- 目录：Word TOC 域，打开后按 F9 / 右键「更新域」生成；
- 参考文献：GB/T 7714 风格；含英文摘要（Abstract）与致谢占位。
"""
from __future__ import annotations

import datetime
import re
import statistics
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from .catalog import get_model_spec
from .experiments import GROUP_LABELS, group_rank

INK = RGBColor(0, 0, 0)
GRAY = RGBColor(0x66, 0x66, 0x66)


# ---------------------------------------------------------------- 基础工具
def _font(run, east: str = "宋体", size: float = 12, bold: bool = False, color=INK, latin: str | None = None) -> None:
    """统一设置中西文字体。east 传『宋体/黑体』时西文默认 Times New Roman；
    east 传『Times New Roman』（英文段落）时中西文都用 TNR。"""
    if latin is None:
        latin = east if east == "Times New Roman" else "Times New Roman"
    run.font.name = latin
    run._element.get_or_add_rPr()
    rfonts = run._element.rPr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def _first_line_chars(p, chars: int = 200) -> None:
    """按『字符数』设置首行缩进（Word 对中文按字符缩进最标准）。"""
    pPr = p._p.get_or_add_pPr()
    ind = pPr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        pPr.append(ind)
    ind.set(qn("w:firstLineChars"), str(chars))
    ind.set(qn("w:firstLine"), str(int(chars / 100 * 240)))  # 回退值：2字符≈24pt=480twips的一半？480=24pt


def _heading(doc: Document, text: str, level: int) -> None:
    """使用 Word 内置 Heading 样式（保证目录域可识别大纲级别），并覆写中文字体。"""
    h = doc.add_heading("", level=level)
    run = h.add_run(text)
    cfg = {1: ("黑体", 16, WD_ALIGN_PARAGRAPH.CENTER),
           2: ("黑体", 14, WD_ALIGN_PARAGRAPH.LEFT),
           3: ("黑体", 13, WD_ALIGN_PARAGRAPH.LEFT)}[level]
    _font(run, east=cfg[0], size=cfg[1], bold=(level >= 2))
    h.alignment = cfg[2]
    h.paragraph_format.space_before = Pt({1: 17, 2: 13, 3: 12}[level])
    h.paragraph_format.space_after = Pt({1: 16.5, 2: 13, 3: 12}[level])
    h.paragraph_format.line_spacing = 1.5
    # 覆写样式自带的颜色（Word 默认标题为蓝色）
    for r in h.runs:
        r.font.color.rgb = INK


def _para(doc: Document, text: str, size: float = 12, bold: bool = False,
          align=None, indent: bool = True, east: str = "宋体", color=INK) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.line_spacing = 1.5
    if indent:
        _first_line_chars(p, 200)
    run = p.add_run(text)
    _font(run, east=east, size=size, bold=bold, color=color)
    return p


def _md_to_paras(doc: Document, text: str) -> None:
    """AI 输出的简易 Markdown → docx 段落。列表转为连贯句，弱化列表体。"""
    lines = (text or "").splitlines()
    buf: list[str] = []
    def flush_list():
        if not buf:
            return
        items = [re.sub(r"^\d+[.、)]\s*", "", b).strip().rstrip("。；;") for b in buf]
        marks = ["一是", "二是", "三是", "四是", "五是", "六是"][: len(items)]
        _para(doc, "；".join(f"{m}{it}" for m, it in zip(marks, items)) + "。")
        buf.clear()
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        clean = line.replace("**", "").lstrip("# ")
        if re.match(r"^\s*([-*]|\d+[.、)])\s+", line):
            buf.append(clean)
        elif line.startswith(("### ", "## ", "# ")):
            flush_list()
            _para(doc, clean, bold=True, indent=False)
        elif line.startswith("> "):
            flush_list()
            _para(doc, line[2:].replace("**", ""), size=10.5, east="楷体", color=GRAY, indent=False)
        else:
            flush_list()
            _para(doc, clean)
    flush_list()


def _caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.2
    run = p.add_run(text)
    _font(run, east="宋体", size=10.5, bold=False)


def _figure(doc: Document, img_path: Path, caption: str, width_cm: float = 12.0) -> bool:
    if not img_path.exists():
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(img_path), width=Cm(width_cm))
    _caption(doc, caption)
    return True


def _three_line_table(doc: Document, headers: list[str], rows: list[list],
                      caption: str | None = None) -> None:
    """三线表：顶线/底线 1.5 磅，栏目线 0.75 磅，无竖线（学位论文标准）。"""
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(6)
        run = cap.add_run(caption)
        _font(run, east="宋体", size=10.5, bold=False)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htext in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(htext))
        _font(run, east="宋体", size=10.5, bold=True)
        # 栏目线：表头行下边框 0.75 磅
        tcPr = cell._tc.get_or_add_tcPr()
        tcB = OxmlElement("w:tcBorders")
        top = OxmlElement("w:top")
        top.set(qn("w:val"), "single")
        top.set(qn("w:sz"), "12")
        tcB.append(top)
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        tcB.append(bottom)
        tcPr.append(tcB)
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run("" if v is None else str(v))
            _font(run, east="宋体", size=10.5)
            if i == len(rows):
                tcPr = cell._tc.get_or_add_tcPr()
                tcB = OxmlElement("w:tcBorders")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single")
                bottom.set(qn("w:sz"), "12")
                tcB.append(bottom)
                tcPr.append(tcB)


def _field(p, instr: str, hint: str) -> None:
    """插入 Word 域（页码 / 目录）。"""
    r1 = p.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    r1._r.append(fld)
    r2 = p.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    r2._r.append(it)
    r3 = p.add_run()
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    r3._r.append(sep)
    rh = p.add_run(hint)
    _font(rh, east="宋体", size=10.5, color=GRAY)
    r4 = p.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    r4._r.append(end)


# ---------------------------------------------------------------- 研究解读
def _primary_metric(run: dict) -> tuple[str, float]:
    pm = (run.get("summary") or {}).get("primary_metric") or {}
    value = pm.get("value")
    try:
        return str(pm.get("name", "primary_metric")), float(value)
    except (TypeError, ValueError):
        return str(pm.get("name", "primary_metric")), 0.0


def _run_by_group(runs: list[dict], group: str) -> dict | None:
    for run in runs:
        if (run.get("config") or {}).get("group") == group:
            return run
    return None


def _run_eval_source(run: dict) -> str:
    return {"test": "独立测试集", "val": "验证集"}.get(
        (run.get("summary") or {}).get("eval_source"), "评估集"
    )


def _abstract_text(title: str, dataset_meta: dict | None, runs: list[dict]) -> str:
    """无 AI 草稿时，基于真实实验结果生成更具体的中文摘要。"""
    if not runs:
        return ""
    dataset_name = (dataset_meta or {}).get("name")
    lead = f"本文以{dataset_name}为实验对象，围绕{title}展开研究。" if dataset_name \
        else f"本文围绕{title}展开研究。"

    base = _run_by_group(runs, "baseline")
    improved = _run_by_group(runs, "improved")
    result = ""
    if base and improved:
        base_name, base_value = _primary_metric(base)
        imp_name, imp_value = _primary_metric(improved)
        source = _run_eval_source(improved)
        base_label = (base.get("summary") or {}).get("model_label", "基线模型")
        imp_label = (improved.get("summary") or {}).get("model_label", "改进模型")
        if base_name == imp_name:
            relative = (imp_value - base_value) / base_value * 100 if base_value else 0.0
            result = (f"在{source}上，{base_name}由基线模型{base_label}的 {base_value:.4f} "
                      f"提升至改进模型{imp_label}的 {imp_value:.4f}，相对提升 {relative:.1f}%。")
        else:
            result = (f"基线模型{base_label}的{base_name}为 {base_value:.4f}，"
                      f"改进模型{imp_label}的{imp_name}为 {imp_value:.4f}。")
    else:
        run = base or improved or runs[0]
        name, value = _primary_metric(run)
        source = _run_eval_source(run)
        label = (run.get("summary") or {}).get("model_label", "所选模型")
        result = f"{label}模型在{source}上的{name}为 {value:.4f}。"

    return lead + result + "实验过程覆盖数据预处理、模型训练、结果评估与可视化分析。"


def _conclusion_text(dataset_meta: dict | None, runs: list[dict]) -> str:
    """无 AI 草稿时，基于真实实验结果生成更具体的结论。"""
    if not runs:
        return ""
    dataset_name = (dataset_meta or {}).get("name")
    lead = f"本文在{dataset_name}上完成了数据处理、模型训练与结果分析。" if dataset_name \
        else "本文完成了数据处理、模型训练与结果分析。"

    base = _run_by_group(runs, "baseline")
    improved = _run_by_group(runs, "improved")
    ablation = _run_by_group(runs, "ablation")
    result = ""
    if base and improved:
        base_name, base_value = _primary_metric(base)
        imp_name, imp_value = _primary_metric(improved)
        base_label = (base.get("summary") or {}).get("model_label", "基线模型")
        imp_label = (improved.get("summary") or {}).get("model_label", "改进模型")
        if base_name == imp_name:
            result = (f"改进模型{imp_label}将{base_name}从基线模型{base_label}的 "
                      f"{base_value:.4f} 提升至 {imp_value:.4f}。")
        else:
            result = (f"基线模型{base_label}的{base_name}为 {base_value:.4f}，"
                      f"改进模型{imp_label}的{imp_name}为 {imp_value:.4f}。")
    elif base:
        name, value = _primary_metric(base)
        source = _run_eval_source(base)
        label = (base.get("summary") or {}).get("model_label", "基线模型")
        result = f"{label}模型在{source}上的{name}为 {value:.4f}。"
    elif improved:
        name, value = _primary_metric(improved)
        source = _run_eval_source(improved)
        label = (improved.get("summary") or {}).get("model_label", "改进模型")
        result = f"{label}模型在{source}上的{name}为 {value:.4f}。"

    if ablation:
        imp_name, imp_value = _primary_metric(improved) if improved else ("主指标", 0.0)
        abl_name, abl_value = _primary_metric(ablation)
        if imp_name == abl_name:
            imp_label = (improved.get("summary") or {}).get("model_label", "完整方案")
            abl_label = (ablation.get("summary") or {}).get("model_label", "消融方案")
            result += (f"消融实验中，{abl_label}的{abl_name}为 {abl_value:.4f}，"
                       f"与完整方案{imp_label}相差 {abl_value - imp_value:+.4f}。")
    return lead + result + "未来可在更大规模数据、更细粒度标注与模型解释方面继续完善。"


def _english_abstract_text(title: str, dataset_meta: dict | None, runs: list[dict]) -> str:
    """Generate a concrete English abstract from real run metadata."""
    if not runs:
        return ""
    dataset_name = (dataset_meta or {}).get("name")
    lead = (f"This thesis investigates {title} on the {dataset_name} dataset." if dataset_name
            else f"This thesis investigates {title}.")
    results: list[str] = []
    for run in runs:
        name, value = _primary_metric(run)
        source = {"test": "independent test set", "val": "validation set"}.get(
            (run.get("summary") or {}).get("eval_source"), "evaluation set"
        )
        label = (run.get("summary") or {}).get("model_label", "selected")
        results.append(
            f"The {label} model achieves a primary metric {name} of {value:.4f} on the {source}."
        )
    return (lead + " The experiments cover data preprocessing, model training, quantitative "
            "evaluation, and visualization analysis. " + " ".join(results))


def _task_methodology_notes(runs: list[dict]) -> list[str]:
    """按任务家族补充实验设计说明，避免报告只写通用流程。"""
    tasks: list[str] = []
    for run in runs:
        task = (run.get("config") or {}).get("task")
        if task and task not in tasks:
            tasks.append(task)

    notes: list[str] = []
    task_notes = {
        "object_detection": (
            "目标检测实验采用 YOLO 格式标注，其中 bbox 标注用于界定目标区域；"
            "评价统一报告 mAP50 与 mAP50-95。验证集用于模型与超参数选择，"
            "测试集仅在最终评估时使用一次，避免结果选择偏置。"
        ),
        "semantic_segmentation": (
            "像素级语义分割实验要求图像与掩码逐像素对齐，评价以 IoU、Dice 和 pixel accuracy 为主；"
            "验证集用于选择最优权重，测试集用于报告最终边界与区域重叠性能。"
        ),
        "image_classification": (
            "图像分类实验采用训练、验证、测试三段划分；训练阶段使用随机翻转、颜色扰动等增强，"
            "评价报告准确率与宏平均 F1，并辅以混淆矩阵定位主要误分类类别。"
        ),
        "text_classification": (
            "文本分类实验先完成清洗、分词与词表构建，再使用嵌入层表示文本；"
            "评价以准确率和宏平均 F1 为主，必要时按类别检查精确率与召回率。"
        ),
        "tabular_classification": (
            "表格分类实验对数值特征执行缺失填充与标准化，对类别特征执行编码；"
            "所有拟合操作只在训练集或训练折内完成，评价使用准确率、宏平均 F1 与 ROC-AUC，"
            "并通过交叉验证检查稳定性。"
        ),
        "time_series_forecasting": (
            "时间序列预测实验按时间顺序划分训练、验证与测试集，避免随机划分导致未来信息泄漏；"
            "评价以 MAE、RMSE 和 MAPE 为主，并结合预测曲线检查滞后与趋势误差。"
        ),
    }
    for task in tasks:
        if task in task_notes:
            notes.append(task_notes[task])
    return notes


def _run_analysis_text(run: dict, dataset_meta: dict | None, runs: list[dict]) -> str:
    """为单个实验生成可复现的结果解读，替代简单的指标占位文案。"""
    s = run.get("summary") or {}
    metrics = s.get("metrics") or {}
    pm = s.get("primary_metric") or {}
    primary_name = str(pm.get("name") or (next(iter(metrics), "primary_metric")))
    primary_value = pm.get("value")
    if primary_value is None and metrics:
        primary_value = metrics.get(primary_name, next(iter(metrics.values())))
    try:
        primary_value = float(primary_value)
    except (TypeError, ValueError):
        primary_value = 0.0
    label = s.get("model_label", run.get("config", {}).get("model_label", "该模型"))
    source = _run_eval_source(run)

    parts = [f"在{source}上，{label}的主指标 {primary_name} = {primary_value:.4f}。"]
    other_metrics = {k: v for k, v in metrics.items() if k != primary_name}
    if other_metrics:
        other_text = "；".join(
            f"{k} = {v:.4f}" if isinstance(v, (int, float)) else f"{k} = {v}"
            for k, v in other_metrics.items()
        )
        parts.append(f"其余指标为：{other_text}。")

    base = _run_by_group(runs, "baseline")
    if base and base is not run:
        base_name, base_value = _primary_metric(base)
        if base_name == primary_name:
            delta = primary_value - base_value
            parts.append(f"相比基线，{primary_name}变化 {delta:+.4f}，可用于判断改进是否真正有效。")

    epochs = s.get("epochs") or []
    if epochs:
        last = epochs[-1]
        train_acc = last.get("train_acc")
        val_acc = last.get("val_acc")
        if isinstance(train_acc, (int, float)) and isinstance(val_acc, (int, float)):
            gap = train_acc - val_acc
            if gap > 0.08:
                parts.append(
                    f"训练准确率 {train_acc:.4f} 与验证准确率 {val_acc:.4f} 相差 {gap:.4f}，"
                    "存在过拟合迹象；可加强数据增强、提高权重衰减、增加 Dropout 或减小模型容量。"
                )
            elif gap < 0.02 and primary_value < 0.80:
                parts.append(
                    "训练与验证表现接近但主指标偏低，存在欠拟合迹象；可增加训练轮次、"
                    "放宽正则化或扩大模型容量。"
                )

    if dataset_meta:
        counts = (dataset_meta.get("stats") or {}).get("class_counts") or {}
        if counts:
            values = list(counts.values())
            ratio = max(values) / max(min(values), 1)
            if ratio >= 3:
                parts.append(
                    f"数据集存在类别不平衡（最多/最少类样本比约 {ratio:.1f}:1），"
                    "不能只看准确率，应同时检查宏平均 F1、按类别的精确率与召回率。"
                )
            if "accuracy" in metrics and "macro_f1" in metrics:
                try:
                    acc = float(metrics["accuracy"])
                    f1 = float(metrics["macro_f1"])
                    if acc - f1 > 0.05:
                        parts.append(
                            f"准确率 {acc:.4f} 高于宏平均 F1 {f1:.4f}，说明部分类别表现偏弱；"
                            "应结合混淆矩阵定位误分类来源，必要时使用类别权重或重采样。"
                        )
                except (TypeError, ValueError):
                    pass

    task = run.get("config", {}).get("task")
    if task == "object_detection" and {"mAP50", "mAP50-95"} & set(metrics):
        parts.append("mAP50 反映低阈值下能否检出目标，mAP50-95 更看重定位质量；"
                     "两者差距大时，应检查边界框回归和密集小目标问题。")
    elif task == "semantic_segmentation" and {"iou", "dice"} & set(metrics):
        parts.append("IoU 与 Dice 更能反映目标区域重叠程度；两者偏低时应重点检查边界模糊、"
                     "小目标和类别边界预测。")
    elif task == "time_series_forecasting" and {"rmse", "mae"} <= set(metrics):
        parts.append("RMSE 大于 MAE 时说明存在较大误差点；应结合预测曲线检查异常波动、"
                     "滞后预测和极端值区间。")

    return "".join(parts)


def _repeat_notes(runs: list[dict]) -> list[str]:
    """汇总同一 batch 的重复实验，帮助判断性能波动是否可接受。"""
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        cfg = run.get("config") or {}
        if cfg.get("batch_kind") == "repeats" and cfg.get("batch_id"):
            grouped.setdefault(str(cfg["batch_id"]), []).append(run)

    notes: list[str] = []
    for batch_runs in grouped.values():
        if len(batch_runs) < 2:
            continue
        metric_name = _primary_metric(batch_runs[0])[0]
        values = [_primary_metric(run)[1] for run in batch_runs]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        label = (batch_runs[0].get("summary") or {}).get("model_label", "该方案")
        stability = "波动很小，结果较稳定" if std <= 0.02 else "波动偏大，建议检查随机性或样本划分"
        notes.append(
            f"重复实验显示，{label} 共运行 {len(batch_runs)} 次，{metric_name}平均值为 {mean:.3f}，"
            f"样本标准差为 {std:.3f}；{stability}。"
        )
    return notes


def _task_background_text(runs: list[dict]) -> str:
    """按实验覆盖的任务类型生成更贴近选题的研究背景，而不是通用套话。"""
    tasks = {(r.get("config") or {}).get("task") for r in runs}
    tasks.discard(None)
    pieces: list[str] = []
    if "image_classification" in tasks:
        pieces.append(
            "在图像识别任务中，卷积神经网络和迁移学习方法已经能在中小规模数据集上取得稳定表现；"
            "但实际系统还要求模型在类别混淆、样本不平衡和有限算力条件下保持可靠性。"
        )
    if "object_detection" in tasks:
        pieces.append(
            "目标检测同时需要判断目标是否存在并给出空间位置，评价口径也比分类更严格；"
            "如何在轻量化网络与定位精度之间取得平衡，是应用部署中的关键问题。"
        )
    if "semantic_segmentation" in tasks:
        pieces.append(
            "语义分割关注每个像素的类别归属，常用于水域、道路、建筑轮廓等边界敏感场景；"
            "这类任务不仅要求类别判断正确，还要求区域重叠指标达到可用水平。"
        )
    if "text_classification" in tasks:
        pieces.append(
            "文本分类需要把非结构化语言转换为可学习的向量表示，并兼顾词序、语义极性和类别均衡；"
            "轻量神经网络和注意力机制为中小规模语料提供了可行的建模路径。"
        )
    if "tabular_classification" in tasks or "tabular_regression" in tasks:
        pieces.append(
            "表格数据具有特征语义明确、样本规模适中但分布差异大的特点；"
            "传统机器学习模型与神经网络、集成模型的比较，能够帮助判断不同归纳偏置的适用范围。"
        )
    if "time_series_forecasting" in tasks:
        pieces.append(
            "时间序列预测依赖时间依赖性和趋势结构，随机划分会引入未来信息泄漏；"
            "因此建模与评估必须保持时间顺序，并同时关注短期误差和长期趋势。"
        )
    if not pieces:
        pieces.append(
            "随着人工智能方法在具体应用场景中的普及，如何根据数据形式、任务目标和算力约束"
            "选择并验证合适的模型，成为工程落地的核心问题。"
        )
    return " ".join(pieces)


def _domain_context(title: str | None, dataset_meta: dict | None) -> set[str]:
    """从题目与数据集信息中识别报告应重点解释的应用场景。"""
    text = " ".join([
        str(title or ""),
        str((dataset_meta or {}).get("name") or ""),
        str((dataset_meta or {}).get("desc") or ""),
    ])
    contexts: set[str] = set()
    if any(k in text for k in ("水面", "水域", "水环境", "河道", "湖泊", "海洋")):
        contexts.add("water_surface")
    if any(k in text for k in ("无人机", "航拍", "低空", "遥感")):
        contexts.add("aerial_detection")
    return contexts


def _domain_context_text(contexts: set[str]) -> str:
    """生成与具体采集环境直接相关的背景说明，而不是通用领域套话。"""
    pieces: list[str] = []
    if "water_surface" in contexts:
        pieces.append(
            "在水面场景中，倒影与反光会改变背景纹理，光照变化会影响目标与水面的对比度，"
            "小目标还容易因边界模糊而降低分割或检测稳定性；因此模型分析和数据增强"
            "需要围绕这些实际干扰条件展开。"
        )
    if "aerial_detection" in contexts:
        pieces.append(
            "在无人机航拍场景中，拍摄视角与高度变化会带来目标尺度差异，"
            "小目标在复杂背景下更容易漏检，光照变化也会影响颜色与边缘特征；"
            "实验设计需要考虑这些条件下的模型稳定性。"
        )
    return " ".join(pieces)


def _domain_limitations_text(contexts: set[str]) -> list[str]:
    """按具体场景补充研究局限，帮助报告从干扰因素回看模型能力。"""
    notes: list[str] = []
    if "water_surface" in contexts:
        notes.append(
            "水面场景的倒影、反光和光照变化会改变局部纹理与边界对比度；"
            "当前实验尚未充分量化这些因素单独引起的性能变化，后续应补充分场景误差分析。"
        )
    if "aerial_detection" in contexts:
        notes.append(
            "航拍场景中视角、高度变化和小目标密度会影响模型尺度适应性；"
            "当前实验对这些条件的组合变化覆盖有限，后续应在更多拍摄距离与视角下验证。"
        )
    return notes


def _literature_review_text(runs: list[dict], dataset_meta: dict | None) -> str:
    """按真实实验覆盖的任务族生成文献综述骨架，避免第二章只写工具栈。"""
    if not runs:
        return ""
    dataset_name = (dataset_meta or {}).get("name") or "目标数据集"
    tasks = {(r.get("config") or {}).get("task") for r in runs}
    tasks.discard(None)
    pieces = [
        f"围绕{dataset_name}任务，相关研究通常沿着“特征表示—模型结构—训练策略—评价口径”四条线索展开。"
        "早期工作更多依赖人工特征与传统分类器，近年研究则把重点转向端到端学习、迁移学习、注意力机制"
        "以及面向具体部署约束的轻量化设计。"
    ]
    for text in _domain_context_text(_domain_context(None, dataset_meta)).split("。"):
        if text:
            pieces.append(text + "。")
    if "tabular_classification" in tasks or "tabular_regression" in tasks:
        pieces.append(
            "在表格数据分析中，逻辑回归、随机森林和梯度提升方法因可解释性与较强的小样本表现被广泛用作基线；"
            "多层感知机等神经结构则用于探索高维特征间的非线性关系。已有实验普遍强调：划分方式、类别不平衡"
            "和超参数选择会显著影响结论的可比性。"
        )
    if "image_classification" in tasks:
        pieces.append(
            "图像分类研究经历了从卷积特征到深度表征的演进，CNN 通过局部感受野降低图像先验建模成本，"
            "ResNet 通过残差连接缓解深层网络的退化问题；迁移学习进一步说明在中小规模数据集上，"
            "预训练表征加任务头部微调往往比从零训练更稳定。"
        )
    if "object_detection" in tasks:
        pieces.append(
            "目标检测研究可分为一阶段与二阶段方法：YOLO 系列强调端到端实时预测，"
            "DETR 及其变体则将检测建模为集合预测问题，并以注意力机制改善全局关系建模。"
            "评价研究通常同时报告 mAP50、mAP50-95、精确率与召回率，以避免只看单一阈值指标。"
        )
    if "semantic_segmentation" in tasks:
        pieces.append(
            "语义分割研究强调像素级对齐与边界保持，U-Net 及其变体通过编码器-解码器结构兼顾上下文信息"
            "与空间细节，广泛用于水域、道路、医学影像等边界敏感场景。相关文献普遍以 IoU、Dice 与像素准确率"
            "共同评价区域重叠和边界质量。"
        )
    if "text_classification" in tasks:
        pieces.append(
            "文本分类研究从词袋与 TF-IDF 特征发展到嵌入表示、TextCNN、循环网络与注意力结构；"
            "轻量模型适合小语料快速验证，注意力机制则有助于捕捉长距离语义依赖。"
            "在小样本与类别不平衡场景中，宏平均 F1 通常比准确率更能反映分类器的真实可用性。"
        )
    if "time_series_forecasting" in tasks:
        pieces.append(
            "时间序列预测研究强调时间依赖性、周期性与外部变量建模，LSTM 和 GRU 通过门控结构缓解长期依赖，"
            "Transformer 类方法则利用注意力捕捉远距离模式。已有工作普遍指出，时间序列必须按时间顺序划分，"
            "否则随机划分会引入未来信息并高估模型性能。"
        )
    pieces.append(
        "综合已有研究可以看出，单一模型的结果难以支撑普适性结论；更可靠的路线是在同一数据划分、同一指标"
        "体系下构造基线与改进方案，并通过消融实验定位每个组件的贡献。本文的实验设计正是围绕这一思路展开。"
    )
    return " ".join(pieces)


def _research_method_text(dataset_meta: dict | None, runs: list[dict]) -> str:
    """生成与真实数据集和实验路线一致的研究方法说明。"""
    dataset_name = (dataset_meta or {}).get("name") or "所选数据集"
    tasks = {(r.get("config") or {}).get("task") for r in runs}
    tasks.discard(None)

    pieces = [
        f"本文以{dataset_name}为对象，按照“数据获取与质量控制—预处理与数据划分—模型构建与训练—"
        "结果评估与误差分析—对比与消融验证”的路线组织研究工作。"
    ]
    for text in _domain_context_text(_domain_context(None, dataset_meta)).split("。"):
        if text:
            pieces.append(text + "。")
    if "tabular_classification" in tasks or "tabular_regression" in tasks:
        pieces.append(
            "在表格数据上，先检查缺失值、重复样本、字段类型和类别分布，再执行缺失填充、"
            "标准化与编码处理；模型选择围绕可解释性和非线性建模能力分别构造基线与改进方案。"
        )
    if "image_classification" in tasks:
        pieces.append(
            "在图像任务上，先检查图像尺寸、类别分布和数据样例，再使用缩放、翻转、颜色扰动等"
            "增强方法扩大有效样本变化；模型设计兼顾特征提取能力、参数规模与训练稳定性。"
        )
    if "object_detection" in tasks:
        pieces.append(
            "在目标检测任务上，采用 YOLO 格式组织图像与标注，统一输入尺寸，"
            "并通过边界框回归、类别置信度和位置重叠指标共同评价检测结果。"
        )
    if "semantic_segmentation" in tasks:
        pieces.append(
            "在语义分割任务上，保持图像与掩码的空间对齐，统一掩码编码和颜色映射，"
            "并通过 IoU、Dice 与像素准确率判断边界和区域重叠质量。"
        )
    if "text_classification" in tasks:
        pieces.append(
            "在文本任务上，先清洗噪声符号并统一分词方式，再构建词表或使用预训练表示；"
            "模型设计兼顾局部词序特征、全局语义特征与类别均衡问题。"
        )
    if "time_series_forecasting" in tasks:
        pieces.append(
            "在时间序列任务上，按时间顺序构造滑动窗口并划分数据，避免随机划分引入未来信息；"
            "模型评估同时关注短期误差、长期趋势和预测滞后。"
        )

    if any((r.get("summary") or {}).get("cv_scores") for r in runs):
        pieces.append(
            "实验同时使用交叉验证检查模型稳定性，避免单次划分带来的偶然波动；"
            "多次重复实验的结果按均值与标准差汇总，作为结论可靠性的依据。"
        )
    else:
        pieces.append(
            "实验采用统一的数据划分与随机种子设置，保证训练、验证和测试过程可复现；"
            "评估结果只使用未参与训练的样本，避免模型选择偏置。"
        )
    pieces.append(
        "最终研究结果通过指标表、训练曲线、混淆矩阵或预测可视化共同核对，"
        "既关注整体性能，也关注误差来源和模型适用条件。"
    )
    return "".join(pieces)


def _evaluation_metrics_text(runs: list[dict]) -> list[str]:
    """按真实实验任务解释评价指标与分析方法。"""
    tasks = {(r.get("config") or {}).get("task") for r in runs}
    tasks.discard(None)
    notes: list[str] = []
    if "tabular_classification" in tasks or "text_classification" in tasks:
        notes.append(
            "分类任务使用准确率反映整体判断正确比例，使用宏平均 F1 反映每个类别"
            "精确率与召回率的综合表现；当类别分布不均衡时，宏平均 F1 比单一准确率更可靠。"
        )
    if "image_classification" in tasks:
        notes.append(
            "图像分类结果除准确率外，还结合混淆矩阵检查主要误分类类别，"
            "并使用训练曲线判断模型是否稳定收敛。"
        )
    if "object_detection" in tasks:
        notes.append(
            "目标检测使用 mAP50 衡量低 IoU 阈值下的整体检测能力，使用 mAP50-95 衡量不同"
            " IoU 阈值下的定位质量；精确率与召回率用于判断漏检和误检之间的权衡。"
        )
    if "semantic_segmentation" in tasks:
        notes.append(
            "语义分割使用 IoU 与 Dice 衡量预测区域和真实区域的重叠程度，"
            "使用像素准确率作为补充；两者共同反映边界质量和类别区域完整性。"
        )
    if "time_series_forecasting" in tasks:
        notes.append(
            "时间序列预测使用 MAE 衡量平均偏差，使用 RMSE 强调较大误差点，"
            "并结合预测曲线检查滞后、相位偏移和趋势误差。"
        )
    if "tabular_regression" in tasks:
        notes.append(
            "回归任务使用 MAE、RMSE 与 R² 共同解释误差大小、异常点敏感性和拟合程度，"
            "避免只看单一指标得出片面结论。"
        )
    notes.append(
        "所有指标均在未参与训练的数据上计算；验证集用于模型与超参数选择，"
        "测试集只用于最终评估，避免反复调参造成结果高估。"
    )
    return notes


def _work_summary_text(dataset_meta: dict | None, runs: list[dict]) -> str:
    """把实验过程组织成论文式的工作总结。"""
    dataset_name = (dataset_meta or {}).get("name") or "所选数据集"
    task_text = {
        "tabular_classification": "表格数据分类",
        "tabular_regression": "表格数据回归",
        "image_classification": "图像分类",
        "object_detection": "目标检测",
        "semantic_segmentation": "语义分割",
        "text_classification": "文本分类",
        "time_series_forecasting": "时间序列预测",
    }
    tasks = list(dict.fromkeys(
        task_text.get((r.get("config") or {}).get("task")) for r in runs
    ))
    tasks = [t for t in tasks if t]
    scope = "、".join(tasks) if tasks else "模型训练与评估"
    return (
        f"本文围绕{scope}任务，基于{dataset_name}完成了数据探索、预处理、模型训练、"
        "性能评估和结果分析。研究过程使用统一的数据划分与随机种子设置，"
        "并将实验配置、日志、指标和图表留档保存，便于后续复现和扩展。"
    )


def _limitations_text(dataset_meta: dict | None, runs: list[dict]) -> list[str]:
    """根据数据规模、实验次数和结果形态生成研究局限。"""
    notes: list[str] = []
    for warning in (dataset_meta or {}).get("quality_warnings") or []:
        if "缺失值" in warning:
            notes.append("数据集中存在缺失值，预处理策略会影响结果；后续应在真实场景中核查缺失机制并做敏感性分析。")
        if "重复样本" in warning:
            notes.append("数据集中存在重复样本，若划分不当可能导致信息泄漏；后续需要更严格的数据来源核查与去重。")
        if "类别分布不均" in warning:
            notes.append("类别分布不均会影响少数类识别，后续应补充少数类样本并评估按类别指标。")
        if "样本规模较小" in warning:
            notes.append("样本规模较小会放大数据划分的偶然性，后续应扩大样本并使用交叉验证或多组随机划分。")
        if "无法读取或损坏" in warning:
            notes.append("数据集中存在损坏图像，可能降低可用训练样本数量；后续应清理坏图并记录清洗后的数据规模。")
        if "内容重复" in warning:
            notes.append("数据集中存在内容重复图像，若划分不当可能导致评估虚高；后续应去重后再补充独立场景样本。")
        if "尺寸异常" in warning:
            notes.append("数据集中存在尺寸异常图像，统一缩放策略可能引入变形；后续应检查采集设置并评估不同缩放方案。")
    n_rows = (dataset_meta or {}).get("n_rows") or ((dataset_meta or {}).get("stats") or {}).get("n_rows")
    try:
        n_rows = int(n_rows)
    except (TypeError, ValueError):
        n_rows = 0
    if n_rows and n_rows < 1000:
        notes.append(
            f"本文使用的样本规模约为 {n_rows} 条，模型学到的规律可能对更复杂的真实场景"
            "泛化有限；后续需要引入更大规模、更多场景的数据继续验证。"
        )

    counts = ((dataset_meta or {}).get("stats") or {}).get("class_counts") or {}
    if counts and len(counts) > 1:
        values = list(counts.values())
        if max(values) / max(min(values), 1) >= 3:
            notes.append(
                "数据集存在类别分布不均，少数类别样本偏少会影响指标稳定性；"
                "后续可采用重采样、类别权重或更多少数类样本加以缓解。"
            )

    if len(runs) < 3:
        notes.append(
            "本文实验数量有限，尚未充分覆盖不同架构、不同优化策略和不同数据划分；"
            "结论应理解为在当前实验设置下的观察结果，而非普适性结论。"
        )

    has_repeats = any(
        (r.get("config") or {}).get("batch_kind") == "repeats" for r in runs
    )
    if not has_repeats:
        notes.append(
            "多数实验只报告单次或少量重复结果，随机性影响尚未充分量化；"
            "后续可通过多次重复实验报告均值和标准差，提高结论稳健性。"
        )

    notes.append(
        "当前研究主要关注模型性能与实验可复现性，对部署延迟、能耗、内存占用和"
        "真实业务约束的评估还不充分。"
    )
    return notes


def _future_work_text(runs: list[dict]) -> str:
    """按实验覆盖范围给出可执行的后续研究方向。"""
    tasks = {(r.get("config") or {}).get("task") for r in runs}
    tasks.discard(None)
    directions: list[str] = ["扩大数据来源和场景覆盖，验证模型在不同分布下的泛化能力"]
    if "tabular_classification" in tasks or "text_classification" in tasks:
        directions.append("比较更多轻量神经网络、集成模型和注意力结构，寻找性能与成本的平衡点")
    if "image_classification" in tasks or "object_detection" in tasks or "semantic_segmentation" in tasks:
        directions.append("引入更强的迁移学习、注意力机制或小目标增强策略，改善复杂场景下的表现")
    if "time_series_forecasting" in tasks:
        directions.append("研究长周期依赖、外部变量和多步预测不确定性对模型性能的影响")
    if any((r.get("config") or {}).get("task") in ("tabular_classification", "image_classification", "text_classification") for r in runs):
        directions.append("增加特征重要性、注意力热力图或类激活图等解释方法，增强模型决策过程透明度")
    directions.append("结合推理速度、内存占用和部署环境评估方法，使研究结论更接近实际应用条件")
    return "；".join(directions) + "。"


def _references_for_runs(runs: list[dict]) -> list[str]:
    """按真实使用到的模型家族生成基础文献；学生仍应补充领域文献。"""
    refs = {
        "sklearn": "PEDREGOSA F, VAROQUAUX G, GRAMFORT A, et al. Scikit-learn: machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.",
        "torch": "PASZKE A, GROSS S, MASSA F, et al. PyTorch: an imperative style, high-performance deep learning library[C]//Advances in Neural Information Processing Systems 32. 2019: 8026-8037.",
        "random_forest": "BREIMAN L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32.",
        "gradient_boosting": "FRIEDMAN J H. Greedy function approximation: a gradient boosting machine[J]. Annals of Statistics, 2001, 29(5): 1189-1232.",
        "resnet": "HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.",
        "lstm": "HOCHREITER S, SCHMIDHUBER J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735-1780.",
        "gru": "CHO K, VAN MERRIENBOER B, GULCEHRE C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation[C]//Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing. 2014: 1724-1734.",
        "transformer": "VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems 30. 2017: 5998-6008.",
        "yolo": "REDMON J, DIVVALA S, GIRSHICK R, et al. You only look once: unified, real-time object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 779-788.",
        "detr": "CARION N, MASSA F, SYNNAEVE G, et al. End-to-end object detection with transformers[C]//European Conference on Computer Vision. 2020: 213-229.",
        "unet": "RONNEBERGER O, FISCHER P, BROX T. U-Net: convolutional networks for biomedical image segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. 2015: 234-241.",
        "attention": "BAHDANAU D, CHO K, BENGIO Y. Neural machine translation by jointly learning to align and translate[C]//International Conference on Learning Representations. 2015.",
    }
    used: set[str] = {"sklearn", "torch"}
    for r in runs:
        cfg = r.get("config") or {}
        model = str(cfg.get("model", "")).lower()
        task = str(cfg.get("task", "")).lower()
        if model in {"random_forest", "rf"}:
            used.add("random_forest")
        if model in {"gbdt", "gradient_boosting", "xgboost", "lightgbm"}:
            used.add("gradient_boosting")
        if model in {"resnet18", "resnet50", "resnet"}:
            used.add("resnet")
        if model == "lstm":
            used.add("lstm")
        if model == "gru":
            used.add("gru")
        if model in {"transformer", "text_transformer", "rtdetr-l", "rt_detr"}:
            used.add("transformer")
        if model.startswith("yolov8") or model.startswith("yolo"):
            used.add("yolo")
        if model == "unet":
            used.add("unet")
        if model == "textcnn":
            used.add("attention")
        if task == "object_detection":
            used.update({"yolo", "detr"})
    preferred_order = ["sklearn", "torch", "random_forest", "gradient_boosting", "resnet",
                       "lstm", "gru", "transformer", "attention", "yolo", "detr", "unet"]
    ordered = [refs[k] for k in preferred_order if k in used]
    ordered.append("（请按 GB/T 7714-2015 著录规则补充与选题直接相关的领域文献，一般不少于 15 篇）")
    return ordered


def _research_notes(runs: list[dict], dataset_meta: dict | None) -> list[str]:
    """从已有实验结果生成研究性解读；不引入外部模型输出，保证可复现。"""
    notes: list[str] = []
    base = _run_by_group(runs, "baseline")
    improved = _run_by_group(runs, "improved")
    repeat_batches = _repeat_notes(runs)
    notes.extend(repeat_batches)
    if base and improved:
        base_name, base_value = _primary_metric(base)
        imp_name, imp_value = _primary_metric(improved)
        if base_name == imp_name:
            base_label = (base.get("summary") or {}).get("model_label", "基线模型")
            imp_label = (improved.get("summary") or {}).get("model_label", "改进模型")
            relative = (imp_value - base_value) / base_value * 100 if base_value else 0.0
            notes.append(
                f"整体对比显示，{imp_label} 的{base_name}由基线 {base_label} 的 {base_value:.4f} "
                f"提升至 {imp_value:.4f}，相对变化 {relative:.1f}%。这个差值是判断改进是否有效的核心依据。"
            )

    if improved:
        ablation = _run_by_group(runs, "ablation")
        if ablation:
            imp_name, imp_value = _primary_metric(improved)
            abl_name, abl_value = _primary_metric(ablation)
            if imp_name == abl_name:
                imp_label = (improved.get("summary") or {}).get("model_label", "完整方案")
                abl_label = (ablation.get("summary") or {}).get("model_label", "消融方案")
                notes.append(
                    f"消融实验中，{abl_label} 的{abl_name}为 {abl_value:.4f}，与完整方案 {imp_label} "
                    f"的 {imp_value:.4f} 相差 {abl_value - imp_value:+.4f}；该差值可用来解释被移除或修改的组件对最终性能的贡献。"
                )

    if dataset_meta:
        counts = ((dataset_meta.get("stats") or {}).get("class_counts") or {})
        if counts:
            values = list(counts.values())
            ratio = max(values) / max(min(values), 1)
            if ratio >= 3:
                notes.append(
                    f"数据集存在类别不平衡（最多/最少类样本比约 {ratio:.1f}:1）。"
                    "此时不能只看准确率，应同时报告宏平均 F1、按类别的精确率与召回率，必要时使用类别权重或重采样。"
                )

    for run in runs:
        epochs = (run.get("summary") or {}).get("epochs") or []
        if epochs:
            last = epochs[-1]
            train_acc = last.get("train_acc")
            val_acc = last.get("val_acc")
            if isinstance(train_acc, (int, float)) and isinstance(val_acc, (int, float)) \
                    and train_acc - val_acc > 0.08:
                label = (run.get("summary") or {}).get("model_label", "该模型")
                notes.append(
                    f"{label} 的训练准确率 {train_acc:.4f} 明显高于验证准确率 {val_acc:.4f}，"
                    "存在过拟合迹象；后续可加强数据增强、提高权重衰减、增加 Dropout 或减小模型容量。"
                )

    if runs:
        eval_source = {"test": "独立测试集", "val": "验证集"}.get(
            runs[0].get("summary", {}).get("eval_source"), "评估集"
        )
        notes.append(
            f"本章结果以{eval_source}为主。模型选择仍应依据验证集或交叉验证完成，"
            "测试集只用于最终评估，避免反复查看同一份测试数据造成选择偏置。"
        )
    return notes


# ---------------------------------------------------------------- 文档组装
def _setup_page(doc: Document, header_text: str | None) -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.6)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.5)
    sec.header_distance, sec.footer_distance = Cm(1.5), Cm(1.75)
    # 页眉
    if header_text:
        hp = sec.header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = hp.add_run(header_text)
        _font(run, east="宋体", size=9, color=GRAY)
    # 页脚页码域
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(fp, "PAGE", "1")
    for r in fp.runs:
        _font(r, east="宋体", size=10.5)
    # 封面页不显示页眉页脚
    sec.different_first_page_header_footer = True


def _setup_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.element.get_or_add_rPr()
    rfonts = normal.element.rPr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    normal.font.color.rgb = INK
    normal.paragraph_format.line_spacing = 1.5


def build_report(
    out_path: Path,
    title: str,
    author_info: dict,
    dataset_meta: dict | None,
    dataset_dir: Path | None,
    runs: list[dict],
    drafts: dict[str, str],
) -> Path:
    """runs: [{run_id, config, summary, run_dir}]；drafts: 章节键 → 正文文本（已去过 AI 味）。"""
    runs = sorted(runs, key=lambda r: (group_rank(r["config"].get("group") or "baseline"),
                                       r.get("name") or "", r.get("run_id") or ""))
    doc = Document()
    _setup_styles(doc)
    school = (author_info or {}).get("school") or ""
    _setup_page(doc, header_text=(f"{school}本科毕业设计（论文）" if school else "本科毕业设计（论文）"))

    # ---------------- 封面
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run(school or "（学校名称）"), "宋体", 22, bold=True)
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run("本科毕业设计（论文）"), "黑体", 22, bold=True)
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run(title or "基于机器学习的毕业设计研究"), "黑体", 18, bold=True)
    for _ in range(3):
        doc.add_paragraph()
    info = author_info or {}
    for label, key in [("学院", "college"), ("专业", "major"), ("姓名", "name"),
                       ("学号", "student_id"), ("指导教师", "advisor")]:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _font(p.add_run(f"{label}：{info.get(key) or '＿＿＿＿＿＿＿＿'}"), "宋体", 14)
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run(datetime.datetime.now().strftime("%Y 年 %m 月")), "宋体", 14)
    doc.add_page_break()

    # ---------------- 目录（自动域）
    _heading(doc, "目  录", 1)
    p = doc.add_paragraph()
    _field(p, 'TOC \\o "1-3" \\h \\z \\u', "（目录为自动域：在 Word 中点这里按 F9，或右键「更新域」生成）")
    doc.add_page_break()

    # ---------------- 中文摘要
    _heading(doc, "摘  要", 1)
    if drafts.get("abstract"):
        _md_to_paras(doc, drafts["abstract"])
    else:
        abstract = _abstract_text(title, dataset_meta, runs)
        if not abstract:
            abstract = (f"本文围绕“{title}”展开研究。研究工作涵盖数据集的获取与预处理、"
                        f"基线模型的构建与训练、算法的改进与对比实验，以及结果的可视化分析。"
                        f"（配置 AI 接口后可自动起草本节，导出后请务必用自己的语言重写。）")
        _para(doc, abstract)
    kw = "、".join(["机器学习", "毕业设计"] + sorted({r["config"].get("model_label", "").split(" (")[0] for r in runs})[:2])
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.5
    _first_line_chars(p, 200)
    _font(p.add_run(f"关键词：{kw}"), "黑体", 12)
    doc.add_page_break()

    # ---------------- 英文摘要
    _heading(doc, "Abstract", 1)
    english_abstract = _english_abstract_text(title, dataset_meta, runs)
    if not english_abstract:
        english_abstract = ("This thesis investigates the application of machine learning methods "
                            "on the selected dataset, covering data collection and preprocessing, "
                            "baseline model construction, comparative experiments, and visualized "
                            "result analysis. Replace this paragraph with your own English abstract "
                            "before submission.")
    _para(doc, english_abstract, east="Times New Roman")
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.5
    _font(p.add_run("Key words: machine learning; graduation design; model evaluation"), "Times New Roman", 12, bold=True)
    doc.add_page_break()

    # ---------------- 第一章 绪论
    _heading(doc, "第一章  绪论", 1)
    _heading(doc, "1.1  研究背景与意义", 2)
    if drafts.get("background"):
        _md_to_paras(doc, drafts["background"])
    else:
        task_background = _task_background_text(runs) if runs else (
            "随着人工智能技术的快速发展，机器学习与深度学习方法在各行各业得到了广泛应用。"
            "如何利用数据驱动的方法解决实际问题，是当前研究的热点之一。"
        )
        domain_context = _domain_context(title, dataset_meta)
        domain_text = _domain_context_text(domain_context)
        _para(doc, task_background + " 本文以此为背景展开研究，"
                   "重点通过可复现实验验证模型选择、改进策略与评价指标之间的关系。"
                   "（导出后请补充 2-3 段与选题直接相关的领域背景。）")
        if domain_text:
            _para(doc, domain_text)
    _heading(doc, "1.2  研究内容", 2)
    if drafts.get("content"):
        _md_to_paras(doc, drafts["content"])
    else:
        items = []
        if dataset_meta:
            items.append(f"基于{dataset_meta.get('name')}数据集完成数据的获取、清洗与探索性分析")
        for r in runs:
            m = r["config"].get("model_label", r["config"].get("model"))
            if m:
                items.append(f"构建并训练{m}模型，完成参数调优与性能评估")
        items.append("设计对比实验，对结果进行可视化与误差分析，并总结改进方向")
        for i, it in enumerate(items):
            _para(doc, f"（{i + 1}）{it}；" if i < len(items) - 1 else f"（{i + 1}）{it}。", indent=False)
    _heading(doc, "1.3  研究方法与技术路线", 2)
    if drafts.get("method"):
        _md_to_paras(doc, drafts["method"])
    else:
        _para(doc, _research_method_text(dataset_meta, runs))
    _heading(doc, "1.4  论文组织结构", 2)
    _para(doc, "本文共分为五章：第一章绪论；第二章介绍相关技术基础；第三章介绍数据集与预处理方法；"
               "第四章给出实验设置、结果与分析；第五章总结全文并展望未来工作。")

    # ---------------- 第二章 相关技术
    _heading(doc, "第二章  相关技术基础", 1)
    _heading(doc, "2.1  文献综述", 2)
    if drafts.get("related"):
        _md_to_paras(doc, drafts["related"])
    else:
        _para(doc, _literature_review_text(runs, dataset_meta))
        _para(doc, "本章介绍研究所涉及的关键技术。实验基于 Python 生态实现：传统机器学习模型采用 "
                   "scikit-learn，深度学习模型采用 PyTorch；实验过程通过可视化控制面板管理，"
                   "保证实验配置可追溯、结果可复现。")
        seen = set()
        for r in runs:
            spec = get_model_spec(r["config"].get("task", ""), r["config"].get("model", ""))
            if spec and spec["model"] not in seen:
                seen.add(spec["model"])
                _para(doc, f"{spec['label']}：{spec['desc']}", indent=False)
        if any(r["config"].get("task") in ("object_detection", "semantic_segmentation") for r in runs):
            _para(doc, "检测与分割实验使用 Ultralytics 训练管线：输入图像按统一尺寸缩放，"
                       "训练阶段应用几何与颜色增强；评价指标统一采用 mAP50 与 mAP50-95，"
                       "实例分割另按 mask 口径计算精确率与召回率。验证集用于模型选择，"
                       "最终结果不在训练过程中反复使用。", indent=False)
        if any(r["config"].get("task") == "semantic_segmentation" for r in runs):
            _para(doc, "像素级语义分割实验采用 images/ + masks/ 数据组织方式，网络输出与输入图像逐像素对齐；"
                       "评价指标以 IoU 与 Dice 为主，同时报告 pixel accuracy。该指标能衡量分割边界与目标区域的"
                       "重叠程度，比单一像素准确率更贴近分割任务的可用性。", indent=False)

    # ---------------- 第三章 数据与预处理
    _heading(doc, "第三章  数据集与预处理", 1)
    if dataset_meta:
        stats = dataset_meta.get("stats") or {}
        _heading(doc, "3.1  数据集介绍", 2)
        _para(doc, f"本文使用{dataset_meta.get('name')}数据集。{dataset_meta.get('desc') or ''}"
                   f"该数据集共 {dataset_meta.get('n_rows') or stats.get('n_rows', '-')} 条样本、"
                   f"{len(dataset_meta.get('columns') or [])} 个字段，标签列为 {dataset_meta.get('target') or '图像类别文件夹'}。")
        if stats.get("class_counts"):
            _heading(doc, "3.2  数据分布", 2)
            total = sum(stats["class_counts"].values())
            rows = [[str(k), v, f"{v / total * 100:.1f}%"] for k, v in stats["class_counts"].items()]
            _three_line_table(doc, ["类别", "样本数", "占比"], rows, caption=f"表 3-1  {dataset_meta.get('name')}类别分布")
        if dataset_dir:
            eda = dataset_dir / "eda"
            figs = [("class_balance.png", "类别分布"), ("histograms.png", "数值特征分布"),
                    ("correlation.png", "特征相关性矩阵"), ("samples.png", "数据样例")]
            fi = 0
            for fname, cap in figs:
                if (eda / fname).exists():
                    fi += 1
                    _figure(doc, eda / fname, f"图 3-{fi}  {cap}")
        quality_warnings = dataset_meta.get("quality_warnings") or []
        if quality_warnings:
            _para(doc, "数据质量检查发现：" + " ".join(quality_warnings)
                  + "上述问题已作为预处理和结果解释的约束条件，避免把数据缺陷误读为模型规律。")
        _heading(doc, "3.3  预处理与数据划分", 2)
        has_vision = any(r["config"].get("task") in ("image_classification", "object_detection", "semantic_segmentation") for r in runs)
        if has_vision:
            _para(doc, "图像任务先划分为训练集、验证集与测试集：权重在训练集上更新，"
                       "模型与超参数依据验证集选择，测试集仅在最终评估时使用一次。"
                       "分类模型按图像尺寸缩放并使用数据增强；检测与分割模型采用 YOLO 格式标注，"
                       "目标检测使用矩形框（bbox），实例分割使用多边形/掩码（mask）。"
                       "所有实验固定随机种子以保证可复现。")
        else:
            _para(doc, "数值特征经中位数填充与标准化处理，类别特征经众数填充与独热编码；"
                   "数据先按分层抽样划分为训练集与测试集，再在训练集内部拟合填充值与标准化参数"
                   "（填充器、标准化器均作为 Pipeline 的一部分只在训练折/训练集上 fit），避免数据泄漏。"
                   "图像任务进一步划分为训练集、验证集与测试集：权重在训练集上更新，"
                   "模型与超参数依据验证集选择，测试集仅在最终评估时使用一次。"
                   "所有实验固定随机种子以保证可复现。")
        if drafts.get("preprocess"):
            _md_to_paras(doc, drafts["preprocess"])
    else:
        _para(doc, "（未选择数据集，可在此补充数据来源与预处理说明。）")

    # ---------------- 第四章 实验与分析
    _heading(doc, "第四章  实验与结果分析", 1)
    _heading(doc, "4.1  实验环境与设置", 2)
    import platform

    env_rows = [["操作系统", platform.system() + " " + platform.release()],
                ["处理器", platform.processor() or "-"],
                ["Python", platform.python_version()]]
    try:
        import torch

        env_rows.append(["PyTorch", torch.__version__ + ("（CUDA 可用）" if torch.cuda.is_available() else "（CPU）")])
    except Exception:
        pass
    _three_line_table(doc, ["项目", "配置"], env_rows, caption="表 4-1  实验环境")
    for note in _task_methodology_notes(runs):
        _para(doc, note)
    if runs:
        rows = []

        def fmt_val(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        for r in runs:
            cfg = r["config"]; s = r["summary"] or {}
            pm = s.get("primary_metric") or {}
            group = cfg.get("group") or "baseline"
            rows.append([GROUP_LABELS.get(group, group),
                         cfg.get("dataset_name", "-"), s.get("model_label", cfg.get("model", "-")),
                         json_params_short(cfg.get("params")),
                         f"{pm.get('name', '-')} = {fmt_val(pm.get('value', '-'))}",
                         f"{s.get('n_train', '-')}/{s.get('n_test', s.get('n_val', '-'))}"])
        _three_line_table(doc, ["实验分组", "数据集", "模型", "主要超参数", "主指标", "训练/测试样本"],
                          rows, caption="表 4-2  实验结果对比")
        _para(doc, "注：主指标均为未参与训练的评估样本结果。图像任务区分训练/验证/测试集，"
                   "test_* 为独立测试集结果（仅在最终评估时使用一次），val_* 为验证集结果"
                   "（模型与超参数选择依据）；表格任务报告单次测试集拆分的指标，并另附交叉验证"
                   "供稳定性参考。论文分析应避免反复查看同一测试集造成选择偏置，"
                   "必要时补充多次重复实验的均值与标准差。")

        notes = _research_notes(runs, dataset_meta)
        _heading(doc, "4.2  综合对比与研究性解读", 2)
        for note in _evaluation_metrics_text(runs):
            _para(doc, note)
        for note in notes:
            _para(doc, note)

    fig_no = 0
    for ri, r in enumerate(runs):
        s = r["summary"] or {}
        cfg = r["config"]
        rd = Path(r.get("run_dir", ""))
        model_label = s.get("model_label", cfg.get("model", "实验"))
        _heading(doc, f"4.{ri + 3}  {model_label} 实验结果分析", 2)
        figure_start = fig_no + 1
        if s.get("cv_scores"):
            import numpy as np

            _para(doc, f"交叉验证得分为 {', '.join(f'{x:.4f}' for x in s['cv_scores'])}，"
                       f"均值 {np.mean(s['cv_scores']):.4f}、标准差 {np.std(s['cv_scores']):.4f}，模型表现稳定。")
        for fname, cap in [("curves.png", "训练损失与准确率曲线"), ("cv_scores.png", "交叉验证得分"),
                           ("confusion_matrix.png", "混淆矩阵"), ("roc.png", "ROC 曲线"),
                           ("pred_vs_true.png", "预测值与真实值对比"), ("feature_importance.png", "特征重要性")]:
            if _figure(doc, rd / fname, f"图 4-{fig_no + 1}  {cap}（{model_label}）"):
                fig_no += 1
        if cfg.get("task") in ("object_detection", "semantic_segmentation"):
            for fname, cap in [("results.png", "训练/验证损失与指标曲线"),
                               ("confusion_matrix_normalized.png", "归一化混淆矩阵"),
                               ("val_batch0_pred.jpg", "验证集预测可视化"),
                               ("val_batch1_pred.jpg", "验证集预测可视化")]:
                if _figure(doc, rd / fname, f"图 4-{fig_no + 1}  {cap}（{model_label}）"):
                    fig_no += 1
        if cfg.get("task") == "semantic_segmentation":
            for fname, cap in [("segmentation_prediction.png", "测试集分割预测掩码")]:
                if _figure(doc, rd / fname, f"图 4-{fig_no + 1}  {cap}（{model_label}）"):
                    fig_no += 1
        if fig_no >= figure_start:
            _para(doc, f"图 4-{figure_start} 至 图 4-{fig_no} 给出{model_label}的关键实验图。"
                       "分析时可将这些图与表 4-2 的指标结合：先看训练曲线是否稳定收敛，"
                       "再检查混淆矩阵中的主要误分类类别；若包含预测可视化，还应核对边界或预测质量；"
                       "同时交叉检查训练配置、数据划分和验证集表现，避免只凭单张图下结论。")
        if drafts.get(f"analysis:{r['run_id']}"):
            _md_to_paras(doc, drafts[f"analysis:{r['run_id']}"])
        elif s.get("metrics"):
            _para(doc, _run_analysis_text(r, dataset_meta, runs))

    # ---------------- 第五章 总结
    _heading(doc, "第五章  总结与展望", 1)
    _heading(doc, "5.1  工作总结", 2)
    if drafts.get("summary"):
        _md_to_paras(doc, drafts["summary"])
    else:
        _para(doc, _work_summary_text(dataset_meta, runs))

    _heading(doc, "5.2  主要结论", 2)
    if drafts.get("conclusion"):
        _md_to_paras(doc, drafts["conclusion"])
    else:
        conclusion = _conclusion_text(dataset_meta, runs)
        if not conclusion:
            conclusion = ("本文完成了从数据准备、模型训练到实验分析的完整研究流程。实验结果表明，"
                          "所选模型在目标数据集上取得了较好的性能。未来工作可从以下方向展开：引入更多对比模型、"
                          "开展消融实验验证各改进模块的有效性、扩大数据规模并探索模型的可解释性。")
        _para(doc, conclusion)

    _heading(doc, "5.3  研究局限", 2)
    for note in _limitations_text(dataset_meta, runs):
        _para(doc, note)
    for note in _domain_limitations_text(_domain_context(title, dataset_meta)):
        _para(doc, note)

    _heading(doc, "5.4  未来展望", 2)
    if drafts.get("future"):
        _md_to_paras(doc, drafts["future"])
    else:
        _para(doc, _future_work_text(runs))

    # ---------------- 参考文献（GB/T 7714 风格）
    _heading(doc, "参考文献", 1)
    refs = _references_for_runs(runs)
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        run = p.add_run(f"[{i}] {ref}")
        _font(run, east="宋体", size=10.5)

    # ---------------- 致谢
    doc.add_page_break()
    _heading(doc, "致  谢", 1)
    _para(doc, "（此处撰写致谢：感谢导师的指导、同学的帮助与家人的支持。建议亲自撰写，"
               "结合具体的帮助细节，这部分通常不查重但最见真情。）")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def json_params_short(params: dict | None) -> str:
    if not params:
        return "默认"
    return ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
