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
    # 清空所有边框，再逐条加三线
    tbl = t._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", "12"), ("bottom", "12"), ("left", "none"),
                     ("right", "none"), ("insideH", "none"), ("insideV", "none")):
        el = OxmlElement(f"w:{edge}")
        if sz == "none":
            el.set(qn("w:val"), "none")
        else:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), sz)  # 单位 1/8 磅：12 = 1.5 磅
        borders.append(el)
    tblPr.append(borders)
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
    doc.add_paragraph()


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


def _research_notes(runs: list[dict], dataset_meta: dict | None) -> list[str]:
    """从已有实验结果生成研究性解读；不引入外部模型输出，保证可复现。"""
    notes: list[str] = []
    base = _run_by_group(runs, "baseline")
    improved = _run_by_group(runs, "improved")
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
                break

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
        _para(doc, f"本文围绕“{title}”展开研究。研究工作涵盖数据集的获取与预处理、基线模型的构建与训练、"
                   f"算法的改进与对比实验，以及结果的可视化分析。（配置 AI 接口后可自动起草本节，"
                   f"导出后请务必用自己的语言重写。）")
    kw = "、".join(["机器学习", "毕业设计"] + sorted({r["config"].get("model_label", "").split(" (")[0] for r in runs})[:2])
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.5
    _first_line_chars(p, 200)
    _font(p.add_run(f"关键词：{kw}"), "黑体", 12)
    doc.add_page_break()

    # ---------------- 英文摘要
    _heading(doc, "Abstract", 1)
    _para(doc, "This thesis investigates the application of machine learning methods on the selected "
               "dataset, covering data collection and preprocessing, baseline model construction, "
               "comparative experiments, and visualized result analysis. Replace this paragraph with "
               "your own English abstract before submission.", east="Times New Roman")
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.5
    _font(p.add_run("Key words: machine learning; graduation design; model evaluation"), "Times New Roman", 12, bold=True)
    doc.add_page_break()

    # ---------------- 第一章 绪论
    _heading(doc, "第一章  绪论", 1)
    _heading(doc, "1.1  研究背景与意义", 2)
    if drafts.get("background"):
        _md_to_paras(doc, drafts["background"])
    else:
        _para(doc, "随着人工智能技术的快速发展，机器学习与深度学习方法在各行各业得到了广泛应用。"
                   "如何利用数据驱动的方法解决实际问题，是当前研究的热点之一。本文以此为背景展开研究，"
                   "具有较好的理论意义与应用价值。（导出后请补充 2-3 段与选题直接相关的领域背景。）")
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
    _heading(doc, "1.3  论文组织结构", 2)
    _para(doc, "本文共分为五章：第一章绪论；第二章介绍相关技术基础；第三章介绍数据集与预处理方法；"
               "第四章给出实验设置、结果与分析；第五章总结全文并展望未来工作。")

    # ---------------- 第二章 相关技术
    _heading(doc, "第二章  相关技术基础", 1)
    if drafts.get("related"):
        _md_to_paras(doc, drafts["related"])
    else:
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
        for note in notes:
            _para(doc, note)

    fig_no = 0
    for ri, r in enumerate(runs):
        s = r["summary"] or {}
        cfg = r["config"]
        rd = Path(r.get("run_dir", ""))
        model_label = s.get("model_label", cfg.get("model", "实验"))
        _heading(doc, f"4.{ri + 3}  {model_label} 实验结果分析", 2)
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
        if drafts.get(f"analysis:{r['run_id']}"):
            _md_to_paras(doc, drafts[f"analysis:{r['run_id']}"])
        elif s.get("metrics"):
            ms = "；".join(f"{k} = {v:.4f}" if isinstance(v, float) else f"{k} = {v}" for k, v in s["metrics"].items())
            src = {"test": "测试集", "val": "验证集"}.get(s.get("eval_source"), "评估集")
            _para(doc, f"该模型在{src}上的表现为：{ms}。（配置 AI 接口后，此处将自动生成实验解读，"
                       f"建议结合自己的理解重写。）")

    # ---------------- 第五章 总结
    _heading(doc, "第五章  总结与展望", 1)
    if drafts.get("conclusion"):
        _md_to_paras(doc, drafts["conclusion"])
    else:
        _para(doc, "本文完成了从数据准备、模型训练到实验分析的完整研究流程。实验结果表明，"
                   "所选模型在目标数据集上取得了较好的性能。未来工作可从以下方向展开：引入更多对比模型、"
                   "开展消融实验验证各改进模块的有效性、扩大数据规模并探索模型的可解释性。")

    # ---------------- 参考文献（GB/T 7714 风格）
    _heading(doc, "参考文献", 1)
    refs = [
        "PEDREGOSA F, VAROQUAUX G, GRAMFORT A, et al. Scikit-learn: machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.",
        "PASZKE A, GROSS S, MASSA F, et al. PyTorch: an imperative style, high-performance deep learning library[C]//Advances in Neural Information Processing Systems 32. 2019: 8026-8037.",
        "HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.",
        "BREIMAN L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32.",
        "（请按 GB/T 7714-2015 著录规则补充与选题直接相关的文献，一般不少于 15 篇）",
    ]
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
