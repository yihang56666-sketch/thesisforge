# -*- coding: utf-8 -*-
"""Delivery readiness checks for the report workshop."""


DEFAULT_TITLE = "基于机器学习的毕业设计研究"


def report_readiness(title: str, author: dict, dataset_meta: dict | None, runs: list[dict]) -> dict:
    """Turn manual delivery checks into visible pass/warn/fail items."""
    checks: list[dict] = []

    def add(label: str, status: str, message: str):
        checks.append({"label": label, "status": status, "message": message})

    clean_title = (title or "").strip()
    if not clean_title or clean_title == DEFAULT_TITLE or len(clean_title) < 6:
        add("论文题目", "fail", "请填写具体题目，不要保留默认题目；题目应说明研究对象、方法和任务。")
    else:
        add("论文题目", "ok", "题目已具体到研究对象与方法。")

    name = (author or {}).get("name", "").strip()
    school = (author or {}).get("school", "").strip()
    missing_author = []
    if not name:
        missing_author.append("姓名")
    if not school:
        missing_author.append("学校")
    if missing_author:
        add("作者信息", "fail", "缺少" + "、".join(missing_author) + "，封面与页眉会不完整。")
    else:
        missing_optional = [
            key for key in ("student_id", "advisor")
            if not (author or {}).get(key, "").strip()
        ]
        if missing_optional:
            add("作者信息", "warn", "缺少" + "、".join(missing_optional) + "，封面信息建议补齐。")
        else:
            add("作者信息", "ok", "封面作者信息完整。")

    if not dataset_meta or not (dataset_meta or {}).get("name"):
        add("数据集", "fail", "请选择或导入用于报告的数据集。")
    else:
        n_rows = int((dataset_meta.get("stats") or {}).get("n_rows") or dataset_meta.get("n_rows") or 0)
        is_image = dataset_meta.get("type") == "image"
        size_bad = n_rows < 100 if is_image else n_rows < 200
        warnings = dataset_meta.get("quality_warnings") or []
        if size_bad:
            unit = "张" if is_image else "条"
            ds_message = (f"当前样本约 {n_rows} {unit}，规模偏小，结论应说明数据划分敏感性；"
                          + (("；".join(warnings) + "。") if warnings else "质量问题需处理或写入研究局限。"))
            add("数据集", "warn", ds_message)
        elif warnings:
            add("数据集", "warn", "；".join(warnings))
        else:
            add("数据集", "ok", "数据集已选择，当前未检测到未处理的质量提醒。")

    if not runs:
        add("实验完成", "fail", "至少需要一个已完成且有摘要的实验写入第四章。")
    else:
        if len(runs) < 3:
            add("实验完成", "warn", f"已有 {len(runs)} 个实验，毕业设计通常至少需要基线、改进和一组对照。")
        else:
            add("实验完成", "ok", f"已有 {len(runs)} 个已完成实验。")

    groups = {
        ((r.get("config") or {}).get("group") or r.get("group") or "custom")
        for r in runs
    }
    missing_groups = [g for g in ("baseline", "improved", "ablation") if g not in groups]
    if not runs:
        add("实验设计", "fail", "没有可分析实验，无法完成对比、消融和研究性解读。")
    elif missing_groups:
        labels = {"baseline": "基线", "improved": "改进", "ablation": "消融"}
        included = [labels[g] for g in ("baseline", "improved", "ablation") if g in groups]
        add("实验设计", "warn", "当前包含" + "、".join(included) + "；缺少" + "、".join(labels[g] for g in missing_groups) + "，研究性结论说服力不足。")
    else:
        add("实验设计", "ok", "基线、改进和消融实验齐备。")

    repeat_groups = {
        str((r.get("config") or {}).get("batch_id") or r.get("name") or r.get("run_id"))
        for r in runs
        if (r.get("config") or {}).get("batch_kind") == "repeats"
        or "repeat" in str(r.get("name") or "").lower()
        or "重复" in str(r.get("name") or "")
    }
    if len(repeat_groups) < 3:
        add("重复实验", "warn", "建议对最优方案重复 3 次以上，报告均值±标准差，避免单次划分造成结论不稳。")
    else:
        add("重复实验", "ok", f"已有 {len(repeat_groups)} 个重复实验。")

    missing_test = [
        str(r.get("name") or r.get("run_id"))
        for r in runs
        if not (r.get("summary") or {}).get("metrics")
        or (r.get("summary") or {}).get("eval_source") != "test"
    ]
    if not runs:
        add("最终评估", "fail", "没有实验指标可写入结果表。")
    elif len(missing_test) == len(runs):
        add("最终评估", "warn", "缺少测试集指标；当前指标可能只来自验证集，不能作为最终结论。")
    elif missing_test:
        add("最终评估", "warn", "、".join(missing_test) + " 缺少测试集指标或摘要不完整。")
    else:
        add("最终评估", "ok", "所选实验都有测试集指标，可用于最终结论。")

    no_history = [
        str(r.get("name") or r.get("run_id"))
        for r in runs
        if not (r.get("summary") or {}).get("epochs")
    ]
    if not runs:
        add("训练曲线", "fail", "没有训练过程可分析。")
    elif no_history:
        add("训练曲线", "warn", "、".join(no_history) + " 缺少训练曲线，难以判断收敛与过拟合。")
    else:
        add("训练曲线", "ok", "所选实验包含训练曲线。")

    no_seed = [
        str(r.get("name") or r.get("run_id"))
        for r in runs
        if not any([
            (r.get("config") or {}).get("seed") is not None,
            (r.get("config") or {}).get("random_state") is not None,
            ((r.get("config") or {}).get("params") or {}).get("seed") is not None,
            (r.get("summary") or {}).get("random_state") is not None,
        ])
    ]
    if no_seed:
        add("可复现记录", "warn", "、".join(no_seed) + " 未记录随机种子，报告应补写数据划分与可复现策略。")
    else:
        add("可复现记录", "ok", "实验配置包含随机种子。")

    blocking = any(c["status"] == "fail" for c in checks)
    return {"checks": checks, "blocking": blocking}
