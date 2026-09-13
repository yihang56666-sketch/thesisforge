"""文本去 AI 味（Humanize）模块。

定位（务必诚实）：
- 这不是"绕过 AIGC 检测"的外挂。各家检测系统（知网、维普、万方、Turnitin 等）
  算法不公开且持续更新，任何工具都无法保证检测结果；
- 本模块做三件实际有用的事：
  1) 规则引擎去掉大模型写作的高频"模板腔"（套话、空洞修饰、排比列表体）；
  2) 打散句长节奏（检测器普遍利用"句长波动小"这一 AI 特征，参考 GPTZero 的
     burstiness 思路与开源中文检测语料 HC3 的对比结论）；
  3) 自检打分：指出文本里最像 AI 的部分，提示作者亲自改写；
- 配置了大模型接口后，可再做一遍 LLM 改写（严格保持事实与数字不变）。
最终仍应以作者本人重写、通读确认为准，并遵守学校关于 AIGC 使用的规定。
"""
from __future__ import annotations

import re
import random

# ---------------------------------------------------------------- 词表
# 高频 AI 套话：直接删除或替换
CLICHE_REMOVE = [
    "值得注意的是，", "值得注意的是：", "需要注意的是，", "需要指出的是，",
    "众所周知，", "毋庸置疑，", "可以说，", "总的来说，",
    "在一定程度上，", "从某种意义上说，", "不难发现，", "可以看出，",
]
CLICHE_REPLACE = {
    "综上所述": ["从上述实验结果看", "结合以上结果", "就本实验而言"],
    "总而言之": ["整体来看", "概括起来"],
    "首先": ["一开始", "第一步"],   # 仅在句首替换，避免破坏逻辑
    "其次": ["接着", "随后"],
    "最后": ["收尾阶段", "到最后一步"],
    "此外": ["另外"],               # 高频词交替
    "然而": ["不过"],
    "因此": ["所以", "由此"],
    "极大地提升": ["明显改善"],
    "极大地提高": ["明显改善"],
    "显著提升": ["提升明显", "有明显提升"],
    "显著提高": ["有明显提高"],
    "有效地提升": ["提升"],
    "有效地提高": ["提高"],
    "有效地": [""],   # 兜底：具体搭配处理完后，剩余的「有效地」直接删
    "全面提升": ["整体改善"],
    "深入探讨": ["讨论"],
    "进行了深入的分析": ["做了分析"],
    "赋能": ["支持"],
    "助力": ["帮助"],
    "深度融合": ["结合"],
    "保驾护航": [],
    "至关重要": ["很重要"],
    "无处不在": ["很常见"],
    "日新月异": ["更新很快"],
    "应运而生": ["随之出现"],
}

# 空洞程度副词：多数情况下直接删掉更平实
INTENSIFIERS = ["非常", "极其", "十分", "相当", "极为", "极大地", "大大地", "高度"]

# AI 常见句式开头（用于开头重复检测）
_CONNS = ["同时，", "此外，", "另外，", "在实验中，", "从结果看，"]

_RND = random.Random(20260913)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])", text)
    return [p for p in parts if p and p.strip()]


def _rand_choice(options: list[str]) -> str:
    return _RND.choice(options) if options else ""


# ---------------------------------------------------------------- 规则改写
def _strip_cliches(text: str, changes: list[str]) -> str:
    for c in CLICHE_REMOVE:
        if c in text:
            text = text.replace(c, "")
            changes.append(f"删除套话「{c.rstrip('，：')}」")
    for word, alts in CLICHE_REPLACE.items():
        if word in text:
            rep = _rand_choice(alts)
            text = text.replace(word, rep)
            changes.append(f"「{word}」→「{rep}」" if rep else f"删除「{word}」")
    return text


def _tone_down_intensifiers(text: str, changes: list[str]) -> str:
    n = 0
    for w in INTENSIFIERS:
        while w in text:
            idx = text.find(w)
            # 保留个别位置以维持自然感，其余删除
            text = text[:idx] + text[idx + len(w):]
            n += 1
            if n > 30:
                break
    if n:
        changes.append(f"删去 {n} 处程度副词（非常/极其/极大地…）")
    return text


def _trim_nominal_verbs(text: str, changes: list[str]) -> str:
    """『对 X 进行了分析』→『分析了 X』一类的冗余动词结构。"""
    pat = re.compile(r"对(.{1,18}?)进行了(分析|研究|讨论|评估|优化|测试)")
    def rep(m):
        return f"{m.group(2)}了{m.group(1)}"
    new, n = pat.subn(rep, text)
    if n:
        changes.append(f"精简 {n} 处「进行了…」结构")
    return new


def _split_long(text: str, changes: list[str]) -> str:
    """超长句在逗号处断开，制造句长波动（burstiness）。"""
    out = []
    for sent in _sentences(text):
        if len(sent) > 90 and sent.count("，") >= 2:
            commas = [m.start() for m in re.finditer("，", sent)]
            mid = min(commas, key=lambda p: abs(p - len(sent) // 2))
            head, tail = sent[: mid + 1], sent[mid + 1:]
            if len(tail.strip()) > 10:
                tail = tail.strip()
                out.append(head + "\n" if text.count("\n") else head)
                out.append(tail if tail.endswith(("。", "；", "！", "？")) else tail + "。")
                changes.append("拆分了一个超长句")
                continue
        out.append(sent)
    return "".join(out)


def _debullet(text: str, changes: list[str]) -> str:
    """连续列表行改为『一是…二是…三是…』的自然段（列表体是明显 AI 特征）。"""
    lines = text.split("\n")
    out, buf = [], []
    def flush():
        if not buf:
            return
        items = [re.sub(r"^(\s*[-*\u2022]|\s*\d+[.、)] )\s*", "", b).strip().rstrip("。；;") for b in buf]
        kv_like = sum(1 for it in items if re.search(r"[\w\u4e00-\u9fff_]+\s*[=:：]", it))
        if len(items) == 1:
            out.append(items[0] + "。")
        elif kv_like >= max(2, len(items) // 2):
            # 指标/键值对列表：自然顿号连接，不套「一是二是」
            out.append("，".join(items) + "。")
        else:
            marks = ["一是", "二是", "三是", "四是", "五是", "六是"][: len(items)]
            joined = "；".join(f"{m}{it}" for m, it in zip(marks, items))
            out.append(joined + "。")
        changes.append(f"把 {len(items)} 条列表改写成连贯段落")
        buf.clear()
    for line in lines:
        if re.match(r"^\s*([-*\u2022]|\d+[.、)])\s+", line):
            buf.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def _vary_openers(text: str, changes: list[str]) -> str:
    """连续句子开头相同时插入衔接词，避免机械排比感。"""
    sents = _sentences(text)
    result = []
    same_streak = 0
    prev_opener = None
    for s in sents:
        opener = s[:2]
        if opener == prev_opener and len(s) > 12:
            same_streak += 1
            if same_streak >= 2:
                s = _rand_choice(_CONNS) + s
                same_streak = 0
                changes.append("调整了连续同头句的节奏")
        else:
            same_streak = 0
        prev_opener = opener
        result.append(s)
    return "".join(result)


def humanize_text(text: str) -> dict:
    """规则引擎主入口：去模板腔 + 打散节奏。返回 {text, changes}。"""
    changes: list[str] = []
    out = text or ""
    before = score_ai_flavor(out)
    out = _debullet(out, changes)
    out = _strip_cliches(out, changes)
    out = _tone_down_intensifiers(out, changes)
    out = _trim_nominal_verbs(out, changes)
    out = _split_long(out, changes)
    out = _vary_openers(out, changes)
    out = re.sub(r"\n{3,}", "\n\n", out)
    after = score_ai_flavor(out)
    return {"text": out, "changes": changes, "score_before": before, "score_after": after}


# ---------------------------------------------------------------- 风险自检
def score_ai_flavor(text: str) -> dict:
    """启发式 AI 特征评分（0-100，越高越像 AI）。模仿检测器常用信号：
    套话密度、列表体占比、句长波动（burstiness）、句首重复、程度副词密度。"""
    text = text or ""
    issues: list[str] = []
    score = 0.0
    n_chars = max(len(text), 1)

    # 1. 套话密度
    hits = [w for w in list(CLICHE_REMOVE) + list(CLICHE_REPLACE) if w in text]
    density = len(hits) / n_chars * 1000
    s = min(30.0, density * 8)
    score += s
    if hits:
        issues.append(f"出现 {len(hits)} 处 AI 高频套话（如「{hits[0]}」等）")

    # 2. 列表体占比
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        list_lines = sum(1 for l in lines if re.match(r"^\s*([-*\u2022]|\d+[.、)])\s+", l))
        ratio = list_lines / len(lines)
        s = min(25.0, ratio * 60)
        score += s
        if ratio > 0.25:
            issues.append(f"列表行占比 {ratio:.0%}，列表体过重（AI 生成文本的典型结构）")

    # 3. 句长波动（burstiness）：AI 文本句长往往过于均匀
    sents = [len(x.strip()) for x in _sentences(text) if len(x.strip()) >= 4]
    if len(sents) >= 4:
        mean = sum(sents) / len(sents)
        var = sum((x - mean) ** 2 for x in sents) / len(sents)
        cv = (var ** 0.5) / mean if mean else 0
        if cv < 0.35:
            score += 25
            issues.append(f"句长过于均匀（波动系数 {cv:.2f} < 0.35），建议长短句交错")
        elif cv < 0.5:
            score += 12
            issues.append(f"句长波动偏小（{cv:.2f}），可再增加一些短句")

    # 4. 句首重复
    sents_text = [x.strip() for x in _sentences(text) if len(x.strip()) >= 6]
    if len(sents_text) >= 4:
        openers = [x[:2] for x in sents_text]
        top = max(set(openers), key=openers.count)
        share = openers.count(top) / len(openers)
        if share > 0.3:
            score += 15
            issues.append(f"「{top}」开头的句子占 {share:.0%}，开头过于单一")

    # 5. 程度副词
    n_int = sum(text.count(w) for w in INTENSIFIERS)
    if n_int / n_chars * 1000 > 2:
        score += min(10.0, n_int)
        issues.append(f"程度副词 {n_int} 处（非常/极其…），显得空洞")

    score = round(min(100.0, score))
    level = "低" if score < 35 else ("中" if score < 60 else "高")
    return {"score": score, "level": level, "issues": issues}


# ---------------------------------------------------------------- LLM 改写
REWRITE_SYSTEM = (
    "你是一名严谨的论文改写助手。请重写用户给出的论文段落，规则："
    "1) 所有事实、数字、模型名、指标、结论必须与原文完全一致，不得增删；"
    "2) 长短句交错，打破均匀节奏；"
    "3) 删去空洞套话和多余的程度副词，用具体表述替代空泛形容；"
    "4) 少用列表，改用自然段；连续句子不要用相同词语开头；"
    "5) 保持平实的中文学术文风。直接输出改写后的正文，不要任何解释或前后缀。"
)


async def llm_rewrite(text: str) -> str:
    from . import ai

    chunks = []
    # 分段改写，每段约 1200 字，避免超出输出限制
    paras = text.split("\n")
    buf, cur = [], 0
    for p in paras:
        buf.append(p)
        cur += len(p)
        if cur > 1200:
            chunks.append("\n".join(buf))
            buf, cur = [], 0
    if buf:
        chunks.append("\n".join(buf))

    out = []
    for ch in chunks:
        if not ch.strip():
            out.append(ch)
            continue
        rewritten = await ai.chat(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": ch}],
            max_tokens=2000, temperature=0.7,
        )
        out.append(rewritten.strip())
    return "\n".join(out)
