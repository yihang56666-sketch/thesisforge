/* 毕设工坊 —— 10 步引导式毕设向导 */
"use strict";

const WIZARD_STEPS = [
  {
    key: "1", name: "立项", tip: "先告诉软件你的毕设方向，后面每一步都会沿用这些信息。",
    fields: ["title", "direction", "author", "advisor", "goal"],
  },
  {
    key: "2", name: "数据", tip: "没有自己的数据就选内置数据集；有 CSV / Excel / 图像 zip 也可以上传。",
  },
  {
    key: "3", name: "数据预处理与划分", tip: "这部分保证实验可复现、结论可信：先划分再预处理，测试集从头到尾只碰一次。",
  },
  {
    key: "4", name: "任务与网络架构", tip: "根据你的数据选任务，再选能完成任务的神经网络；表格、图像、文本三类都有覆盖。",
  },
  {
    key: "5", name: "优化与训练策略", tip: "SGD、Momentum、Adam、AdamW、RMSprop 都可以选；不会调参就保持推荐值，先跑通再优化。",
  },
  {
    key: "6", name: "训练与监控", tip: "给实验起名、选分组，然后一键开跑；训练过程中可以随时看实时日志和曲线。",
    checks: [
      "实验命名与分组（基线 / 改进 / 消融），后面对比不猜谜",
      "训练在后台运行，切走也能继续，可随时回来看日志与曲线",
      "中断/失败会保留现场并写日志，不会白屏或直接消失",
    ],
    href: "#/train", cta: "去训练页开跑",
  },
  {
    key: "7", name: "评估分析", tip: "用独立测试集看模型到底行不行，再让软件帮你解读，结论才有说服力。",
    checks: [
      "独立测试集指标：准确率 / F1 / AUC 等",
      "混淆矩阵、ROC 曲线、损失曲线自动生成",
      "AI 解读结果并提出可验证的改进方向",
      "图表可直接插入论文第四章",
    ],
    href: "#/runs", cta: "去实验记录看结果",
  },
  {
    key: "8", name: "消融实验", tip: "逐组件关掉改进点来验证它是否真的有用，这是答辩时最常被追问的部分。",
    checks: [
      "基线实验：最简单的可用方法",
      "改进实验：每次只改一个变量",
      "消融实验：逐个关闭改进组件",
      "对比必须有解释，不只是数字",
    ],
    href: "#/train", cta: "去训练页补实验",
  },
  {
    key: "9", name: "实验对比", tip: "把基线、改进、消融放进同一张表，直接生成论文用的三线表。",
    checks: [
      "基线 / 改进 / 消融自动分组",
      "同一张主指标表，最佳值自动高亮",
      "CSV / Markdown 一键导出论文三线表",
      "同一配置重复 3-5 次，统计均值 ± 标准差",
    ],
    href: "#/compare", cta: "去实验对比页",
  },
  {
    key: "10", name: "报告工坊", tip: "把实验和图表组装成 Word 初稿，再做 AI 自检与降 AI 味。",
    checks: [
      "按毕业论文章节结构生成 Word 初稿",
      "实验三线表与图表自动插入",
      "AIGC 自检与降 AI 味",
      "导出后 Ctrl+A 全选 → F9 更新目录",
    ],
    href: "#/report", cta: "去报告工坊",
  },
];

const WIZARD_DIRECTIONS = ["图像分类", "目标检测", "语义分割", "文本分类", "表格预测", "时间序列预测", "算法优化与消融", "其他"];

function wizardInferDirection(task, type) {
  if (type === "image") {
    if (task === "object_detection") return "目标检测";
    if (task === "semantic_segmentation") return "语义分割";
    return "图像分类";
  }
  if (task === "text_classification") return "文本分类";
  if (task === "time_series_forecasting") return "时间序列预测";
  if (task === "tabular_regression") return "表格预测";
  if (task === "tabular_classification") return "表格预测";
  return "其他";
}

function wizardTaskType(task, type) {
  return type || (String(task || "").startsWith("image") ? "image" : "tabular");
}

function wizardDirectionMatches(direction, task, type) {
  const inferred = wizardInferDirection(task, type);
  return direction === "其他" || direction === "算法优化与消融" || direction === inferred;
}

const DATA_PREP_FIELDS = [
  { key: "missing", label: "缺失值处理", kind: "choice", options: [["impute", "自动填充"], ["drop", "删除含缺失行"], ["keep", "保留缺失"]], note: "样本少时优先填充而不是删行" },
  { key: "impute", label: "填充方式", kind: "choice", options: [["median", "中位数"], ["mean", "均值"], ["most_frequent", "众数"], ["zero", "填 0"], ["none", "不填充"]], note: "数值列用中位数更稳健" },
  { key: "scale", label: "标准化", kind: "choice", options: [["standard", "Z-score 标准化"], ["minmax", "Min-Max 归一化"], ["robust", "稳健缩放"], ["none", "不缩放"]], note: "MLP/CNN/距离类模型通常需要" },
  { key: "encode", label: "类别编码", kind: "choice", options: [["onehot", "独热编码"], ["label", "标签编码"], ["none", "不编码"]], note: "树模型可不编码" },
  { key: "augment", label: "数据增强", kind: "choice", options: [["none", "关闭"], ["flip_rotate", "翻转 / 旋转"], ["crop", "随机裁剪"], ["color_jitter", "颜色扰动"], ["all", "组合增强"]], note: "图像任务常用；文本任务建议关闭" },
];

const TRAINING_FIELDS = [
  { key: "optimizer", label: "优化器", kind: "choice", options: [["sgd", "SGD"], ["sgd_momentum", "SGD + Momentum"], ["adam", "Adam（推荐）"], ["adamw", "AdamW"], ["rmsprop", "RMSprop"]] },
  { key: "lr", label: "学习率", kind: "float", min: 0.00001, max: 1, step: 0.0001, fallback: 0.001 },
  { key: "batch_size", label: "批大小", kind: "int", min: 1, max: 256, step: 1, fallback: 32 },
  { key: "epochs", label: "训练轮次", kind: "int", min: 1, max: 300, step: 1, fallback: 15 },
  { key: "scheduler", label: "学习率调度", kind: "choice", options: [["cosine", "余弦退火"], ["step", "阶梯下降"], ["plateau", "自适应下降"], ["none", "关闭"]] },
  { key: "weight_decay", label: "权重衰减 L2", kind: "float", min: 0, max: 0.5, step: 0.0001, fallback: 0 },
  { key: "early_stop_patience", label: "早停耐心（0=关闭）", kind: "int", min: 0, max: 100, step: 1, fallback: 0 },
  { key: "grad_clip", label: "梯度裁剪（0=关闭）", kind: "float", min: 0, max: 100, step: 0.1, fallback: 0 },
  { key: "device", label: "训练设备", kind: "choice", options: [["auto", "自动"], ["cpu", "仅 CPU"], ["gpu", "优先 GPU"]] },
];

const TRAINING_DEFAULT = { optimizer: "adam", lr: 0.001, batch_size: 32, epochs: 15, scheduler: "cosine", weight_decay: 0, early_stop_patience: 0, grad_clip: 0, device: "auto" };
const DATA_PREP_DEFAULT = { missing: "impute", impute: "median", scale: "standard", encode: "onehot", augment: "flip_rotate", split_first: true, test_size: 0.2, val_split: 0.2, seed: 42 };
const ARCH_PARAM_SKIP = new Set(["optimizer", "lr", "batch_size", "epochs", "scheduler", "weight_decay", "early_stop_patience", "grad_clip", "seed", "device"]);

let wizardState = null;
let wizardSaveQueue = Promise.resolve();
let wizardCtx = { dsR: null, modelsR: null };
let wizardEstimateTimer = null;

function wizardStepEntry(key) {
  if (!wizardState.steps[key]) wizardState.steps[key] = { state: "todo", completed: false, saved_at: null, template: null };
  return wizardState.steps[key];
}

function wizardTemplate(key) {
  const st = wizardStepEntry(key);
  if (!st.template || typeof st.template !== "object") st.template = {};
  return st.template;
}

function wizardTemplateDefaults(key) {
  const t = wizardTemplate(key);
  const src = key === "3" ? DATA_PREP_DEFAULT : key === "5" ? TRAINING_DEFAULT : null;
  if (src) {
    for (const [k, v] of Object.entries(src)) {
      if (!(k in t)) t[k] = v;
    }
  }
  return t;
}

function wizardTaskOptions(ds, catalog) {
  const keys = Object.keys(catalog || {});
  if (!ds) return keys;
  if (ds.type === "image") {
    if (ds.task === "object_detection" && keys.includes("object_detection")) return ["object_detection"];
    if (ds.task === "semantic_segmentation" && keys.includes("semantic_segmentation")) return ["semantic_segmentation"];
    return keys.includes("image_classification") ? ["image_classification"] : keys;
  }
  if (ds.task && keys.includes(ds.task)) return [ds.task];
  return keys.filter((k) => k.startsWith("tabular_") || k === "time_series_forecasting");
}

function wizardDatasetBar(dsR) {
  const ds = ((dsR && dsR.datasets) || [])[0] || null;
  if (!ds) {
    return `<div class="wz-ds-current warn"><div class="name">还没有数据集</div><div class="desc">先到步骤 2 载入内置示例或上传自己的数据，这里会自动根据数据集推断任务类型。</div></div>`;
  }
  const sizeText = ds.type === "image"
    ? `${ds.n_images || "?"} 张 / ${ds.n_classes || "?"} 类`
    : `${ds.n_rows || "?"} 行 × ${(ds.columns || []).length || "?"} 列`;
  const matched = wizardDirectionMatches((wizardState.project || {}).direction, ds.task, ds.type);
  const warn = matched ? "" : `<div class="muted small mt4">注意：这个数据集和当前立项方向不一致，建议回步骤 2 重新选择，或在步骤 1 调整研究大类。</div>`;
  return `<div class="wz-ds-current ${matched ? "" : "warn"}"><span class="tag">${ds.type === "image" ? "图像" : "表格"}</span><div class="name">${esc(ds.name)}</div><div class="desc">规模：${esc(sizeText)}</div>${warn}</div>`;
}

async function loadWizard() {
  wizardState = await api("/api/wizard");
}

async function saveWizard(patch) {
  if (!wizardState) return;
  if (patch) Object.assign(wizardState, patch);
  const snapshot = JSON.parse(JSON.stringify(wizardState));
  const task = wizardSaveQueue.then(async () => {
    try {
      return await api("/api/wizard", { method: "PUT", body: snapshot });
    } catch (e) {
      toast(e.message, "error");
      throw e;
    }
  });
  wizardSaveQueue = task.catch(() => {});
  return task;
}

async function markWizardStep(key, action) {
  const savedAt = action === "done" || action === "skipped"
    ? new Date().toLocaleString("zh-CN", { hour12: false })
    : null;
  const prev = wizardState.steps[key] || {};
  wizardState.steps[key] = {
    state: action,
    completed: action === "done",
    saved_at: savedAt,
    template: prev.template || null,
  };
  await saveWizard();
}

async function wizardGoTo(stepNo) {
  const n = Math.max(1, Math.min(10, stepNo));
  wizardState.active_step = n;
  await saveWizard();
  renderWizard().catch((e) => toast(e.message, "error"));
}

/* ================= 渲染 ================= */
function wizardStepsBar(activeNo) {
  return `<div class="wizard-steps">${WIZARD_STEPS.map((s, i) => {
    const st = (wizardState.steps[s.key] || {});
    const cls = ["wizard-step-chip"];
    if (i + 1 === activeNo) cls.push("active");
    if (st.state === "done") cls.push("done");
    if (st.state === "skipped") cls.push("skipped");
    return `<button class="${cls.join(" ")}" data-wgoto="${s.key}">${i + 1}. ${esc(s.name)}</button>`;
  }).join("")}</div>`;
}

function wizardProjectFields() {
  const p = wizardState.project || {};
  return `
    <b class="t">项目基本信息</b>
    <div class="form-grid">
      <div class="form-row"><label>论文题目</label>
        <input data-wfield="title" maxlength="120" value="${esc(p.title)}" placeholder="如：基于深度学习的×××研究与优化"></div>
      <div class="form-row"><label>研究大类</label>
        <select data-wfield="direction">
          ${WIZARD_DIRECTIONS.map((d) => `<option ${d === p.direction ? "selected" : ""}>${esc(d)}</option>`).join("")}
        </select></div>
      <div class="form-row"><label>作者姓名</label><input data-wfield="author" maxlength="40" value="${esc(p.author)}" placeholder="你的姓名"></div>
      <div class="form-row"><label>指导教师</label><input data-wfield="advisor" maxlength="40" value="${esc(p.advisor)}" placeholder="指导教师"></div>
    </div>
    <div class="form-row"><label>一句话说明你要解决的问题与数据</label>
      <textarea data-wfield="goal" rows="3" maxlength="500" placeholder="例如：使用某数据集完成图像分类，基线用 CNN，改进方向是注意力机制与数据增强。">${esc(p.goal)}</textarea></div>
    <div class="hint">这些信息会写入最后的论文封面与开题字段；现在不确定也可以先跳过，后面在「报告工坊」再补。</div>`;
}

function wizardDatasetPane(dsR) {
  const builtin = (dsR && dsR.builtin) || [];
  const datasets = (dsR && dsR.datasets) || [];
  const direction = (wizardState.project || {}).direction;
  const taskName = (t) => {
    if (t === "object_detection") return "目标检测";
    if (t === "semantic_segmentation") return "语义分割";
    if (t === "time_series_forecasting") return "时间序列";
    return t === "image_classification" ? "图像" : t === "text_classification" ? "文本" : "表格";
  };
  return `
    <b class="t">内置数据集（可直接载入，自动完成 EDA 与 AI 解读）</b>
    <div class="wz-ds-grid">
      ${builtin.map((b) => `
        <div class="wz-ds-card">
          <div class="name">${esc(b.name)}</div>
          <div class="desc">${esc(b.desc || "")}</div>
          <div class="foot"><span class="tag ${wizardDirectionMatches(direction, b.task, wizardTaskType(b.task, b.type)) ? "" : "warn"}">${taskName(b.task)} · ${wizardDirectionMatches(direction, b.task, wizardTaskType(b.task, b.type)) ? "匹配" : "不匹配"}</span>
          <button class="btn small accent" data-builtin="${esc(b.key)}">载入</button></div>
        </div>`).join("")}
    </div>
    <hr class="sep">
    <b class="t">当前已载入 ${datasets.length} 个数据集</b>
    ${datasets.length ? `
      <div class="table-scroll"><table class="data"><thead><tr><th>名称</th><th>类型</th><th>规模</th><th>状态</th></tr></thead>
      <tbody>${datasets.map((d) => `<tr>
        <td><b>${esc(d.name)}</b></td>
        <td>${d.type === "image" ? (d.task === "object_detection" ? "检测" : d.task === "semantic_segmentation" ? "分割" : "图像") : "表格"}</td>
        <td>${d.type === "image" ? (d.n_images || "?") + " 张 / " + (d.n_classes || "?") + " 类" : esc((d.n_rows || "?") + " 行 × " + (d.columns || []).length + " 列")}</td>
        <td><span class="status done">已就绪</span>${datasetQualityHtml(d, true)}</td></tr>`).join("")}</tbody></table></div>`
      : `<div class="empty">还没有数据集。先载入一个内置数据集，或到「数据集」页上传你自己的文件。</div>`}
    <div class="row-flex mt14">
      <button class="btn" data-wgoto-page="#/datasets">去数据集页上传 / 下载</button>
    </div>`;
}

function wizardDataPrepPane(dsR) {
  const t = wizardTemplateDefaults("3");
  const ds = ((dsR && dsR.datasets) || [])[0] || null;
  const isImage = !!ds && ds.type === "image";
  const isText = !!ds && ds.task === "text_classification";
  const hasAug = Object.prototype.hasOwnProperty.call(t, "augment");
  if (isText && !hasAug) t.augment = "none";
  let fields = DATA_PREP_FIELDS;
  if (isImage) fields = DATA_PREP_FIELDS.filter((f) => f.key === "augment");
  else if (isText) fields = DATA_PREP_FIELDS.filter((f) => ["missing", "impute"].includes(f.key));
  const choiceRow = (f) => {
    const v = t[f.key] ?? f.options[0][0];
    return `<div class="form-row"><label>${esc(f.label)}</label>
      <select data-wprep="${esc(f.key)}">${f.options.map(([val, lab]) =>
        `<option value="${esc(val)}" ${String(v) === val ? "selected" : ""}>${esc(lab)}</option>`).join("")}</select>
      ${f.note ? `<div class="hint">${esc(f.note)}</div>` : ""}</div>`;
  };
  return `
    ${wizardDatasetBar(dsR)}
    <b class="t">预处理策略</b>
    <div class="form-grid">${fields.map(choiceRow).join("")}</div>
    <b class="t mt14">数据划分（先划分再预处理，测试集只评估一次）</b>
    <div class="form-grid">
      <div class="form-row"><label>划分顺序</label>
        <select data-wprep="split_first">
          <option value="true" ${t.split_first ? "selected" : ""}>先划分，再在训练集上拟合预处理</option>
          <option value="false" ${!t.split_first ? "selected" : ""}>先预处理，再划分</option>
        </select>
        <div class="hint">推荐先划分，避免测试集信息泄漏进预处理统计量。</div></div>
      <div class="form-row"><label>测试集比例</label>
        <input data-wprep-num="test_size" type="number" min="0.05" max="0.5" step="0.05" value="${esc(t.test_size)}"></div>
      ${isImage ? `<div class="form-row"><label>验证集比例</label>
        <input data-wprep-num="val_split" type="number" min="0.05" max="0.5" step="0.05" value="${esc(t.val_split)}">
        <div class="hint">图像任务按 训练/验证/测试 三份划分。</div></div>` : ""}
      <div class="form-row"><label>随机种子</label>
        <input data-wprep-num="seed" type="number" min="0" step="1" value="${esc(t.seed)}">
        <div class="hint">固定种子，实验可复现。</div></div>
    </div>
    <div class="hint mt8">${isImage ? "图像数据集自动按类别文件夹组织，重点是增强、划分与种子。" : isText ? "文本数据不推荐增强，数值列缺失会按所选策略处理。" : "数值列缺失自动填充，类别列自动编码，标准化默认开启。"}</div>`;
}

function wizardNetworkPane(dsR, modelsR) {
  const catalog = (modelsR && modelsR.catalog) || {};
  const ds = ((dsR && dsR.datasets) || [])[0] || null;
  const t = wizardTemplate("4");
  const tasks = wizardTaskOptions(ds, catalog);
  const task = tasks.includes(t.task) ? t.task : tasks[0];
  t.task = task;
  const taskDef = catalog[task];
  const modelKeys = taskDef ? Object.keys(taskDef.models) : [];
  const arch = modelKeys.includes(t.arch) ? t.arch : (modelKeys[0] || "");
  t.arch = arch;
  if (!t.params || typeof t.params !== "object") t.params = {};
  const spec = taskDef && taskDef.models[arch];
  const structureParams = spec ? Object.entries(spec.params).filter(([k]) => !ARCH_PARAM_SKIP.has(k)) : [];
  const row = (k, ps) => {
    const v = k in t.params ? t.params[k] : ps.default;
    let input;
    if (ps.type === "choice") {
      input = `<select data-warch-param="${esc(k)}">${ps.options.map((o) =>
        `<option value="${esc(o)}" ${String(o) === String(v) ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
    } else if (ps.type === "bool") {
      input = `<select data-warch-param="${esc(k)}">
        <option value="true" ${v ? "selected" : ""}>开启</option>
        <option value="false" ${!v ? "selected" : ""}>关闭</option></select>`;
    } else if (ps.type === "string") {
      input = `<input data-warch-param="${esc(k)}" type="text" value="${esc(v)}">`;
    } else {
      input = `<input data-warch-param="${esc(k)}" type="number" step="${ps.type === "float" ? "0.001" : "1"}"
        min="${esc(ps.min ?? "")}" max="${esc(ps.max ?? "")}" value="${esc(v)}">`;
    }
    return `<div class="form-row"><label>${esc(ps.label || k)}</label>${input}</div>`;
  };
  return `
    ${wizardDatasetBar(dsR)}
    <b class="t">任务类型（根据你的数据自动推断，可修改）</b>
    <div class="form-row"><select data-wtask>
      ${tasks.map((k) => `<option value="${esc(k)}" ${k === task ? "selected" : ""}>${esc((catalog[k] || {}).label || k)}</option>`).join("")}
    </select></div>
    <b class="t mt14">选择网络架构</b>
    ${taskDef ? `<div class="model-grid">${modelKeys.map((k) => {
      const m = taskDef.models[k];
      return `<div class="model-item ${k === arch ? "selected" : ""}" data-warch="${esc(k)}">
        <div class="m-name">${esc(m.label)}</div>
        <div class="m-desc">${esc(m.desc)}</div>
        <div class="m-tag">${m.engine === "torch" ? "神经网络" : "机器学习基线"}</div></div>`;
    }).join("")}</div>` : '<div class="hint">当前任务类型没有可用模型，请先检查数据集。</div>'}
    <b class="t mt14">网络结构参数</b>
    <div class="form-grid">${structureParams.length ? structureParams.map(([k, ps]) => row(k, ps)).join("")
      : '<div class="hint">这个模型没有结构参数，使用默认结构即可。</div>'}</div>
    <b class="t mt14">网络代价预览</b>
    <div id="wz-estimate" class="wz-estimate loading">正在估算参数量与设备友好度…</div>
    <div class="hint mt8">优化器、学习率、轮次等训练参数放到下一步设置，这里只看网络结构。</div>`;
}

function wizardTrainingPane() {
  const t = wizardTemplateDefaults("5");
  const row = (f) => {
    if (f.kind === "choice") {
      const v = t[f.key] ?? f.options[0][0];
      return `<div class="form-row"><label>${esc(f.label)}</label>
        <select data-wtrain="${esc(f.key)}">${f.options.map(([val, lab]) =>
          `<option value="${esc(val)}" ${String(v) === val ? "selected" : ""}>${esc(lab)}</option>`).join("")}</select></div>`;
    }
    return `<div class="form-row"><label>${esc(f.label)}</label>
      <input data-wtrain-num="${esc(f.key)}" type="number" min="${esc(f.min)}" max="${esc(f.max)}" step="${esc(f.step)}" value="${esc(t[f.key] ?? f.fallback)}"></div>`;
  };
  return `
    <b class="t">训练策略</b>
    <div class="wz-note">不会调参就保持推荐值：Adam + 学习率 0.001 + 余弦退火，先跑通再优化；每个选项之后都能在训练页再改。</div>
    <div class="form-grid">${TRAINING_FIELDS.map(row).join("")}</div>
    <b class="t mt14">环境自检（训练前先看这里，缺依赖会得到人话提示）</b>
    <div id="wz-env" class="wz-estimate loading">正在检查 PyTorch / CPU / GPU…</div>`;
}

async function wizardEstimate() {
  const t = wizardState && wizardState.steps["4"] && wizardState.steps["4"].template;
  const box = $("#wz-estimate");
  if (!t || !box || !t.task || !t.arch) return;
  const ds = ((wizardCtx.dsR && wizardCtx.dsR.datasets) || [])[0] || null;
  box.className = "wz-estimate loading";
  box.innerHTML = "正在估算参数量与设备友好度…";
  try {
    const r = await api("/api/networks/estimate", {
      method: "POST",
      body: { task: t.task, model: t.arch, params: Object.assign({}, t.params), dataset_id: ds ? ds.id : null },
    });
    if (!box.isConnected) return;
    if (!r.ok) {
      box.className = "wz-estimate warn";
      box.innerHTML = `<div class="t">估算失败</div>${(r.warnings || []).map((w) =>
        `<div class="muted small mt8">${esc(w)}</div>`).join("")}`;
      return;
    }
    const chips = [
      ["参数量", r.params_display || "—"],
      ["建议轮次", r.recommended_epochs ?? "—"],
      ["训练速度", r.speed || "—"],
      ["设备友好度", r.device_friendly || "—"],
    ];
    box.className = "wz-estimate ok";
    box.innerHTML = `<div class="wz-env-grid">${chips.map(([k, v]) =>
      `<div class="kv"><b>${esc(v)}</b>${esc(k)}</div>`).join("")}</div>
      ${(r.warnings || []).map((w) => `<div class="muted small mt8">注意：${esc(w)}</div>`).join("")}`;
  } catch (e) {
    box.className = "wz-estimate warn";
    box.innerHTML = `估算失败：${esc(e.message)}`;
  }
}

async function wizardEnv() {
  const box = $("#wz-env");
  if (!box) return;
  box.className = "wz-estimate loading";
  box.innerHTML = "正在检查 PyTorch / CPU / GPU…";
  try {
    const r = await api("/api/environment/check");
    if (!box.isConnected) return;
    const chips = [
      ["Python", r.python || "—"],
      ["PyTorch", r.torch || (r.torch_available ? "已安装" : "未安装")],
      ["训练设备", r.device || "未检测"],
      ["内存 / 显存", r.memory_gb ? `${r.memory_gb} GB` : "—"],
      ["torchvision", r.torchvision || "—"],
      ["scikit-learn", r.sklearn || "—"],
    ];
    box.className = `wz-estimate ${r.warnings && r.warnings.length ? "warn" : "ok"}`;
    box.innerHTML = `<div class="wz-env-grid">${chips.map(([k, v]) =>
      `<div class="kv"><b>${esc(v)}</b>${esc(k)}</div>`).join("")}</div>
      ${(r.warnings || []).map((w) => `<div class="muted small mt8">注意：${esc(w)}</div>`).join("")}
      <button class="btn small ghost mt8" data-wenv-run>重新检查</button>`;
    const rerun = box.querySelector("[data-wenv-run]");
    if (rerun) rerun.onclick = () => wizardEnv();
  } catch (e) {
    box.className = "wz-estimate warn";
    box.innerHTML = `环境自检失败：${esc(e.message)}`;
  }
}

async function wizardWriting(stepKey) {
  const out = $("#wz-write-out");
  const btn = document.querySelector("[data-wsnippet]");
  if (!out) return;
  if (btn) { btn.disabled = true; btn.textContent = "生成中…"; }
  try {
    const r = await api(`/api/writing/snippet?step=${encodeURIComponent(stepKey)}`);
    out.innerHTML = `<div class="wz-write-out"><div class="ai-source">论文提示 · 步骤 ${esc(stepKey)} 可直接扩展</div>
      <div class="ai-out">${mdToHtml(r.text)}</div></div>`;
    out.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    toast(e.message, "error");
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "这一步论文怎么写"; }
  }
}

function wizardChecksPane(step) {
  return `
    <b class="t">这一节软件会帮你完成</b>
    <div class="wz-check-list">${step.checks.map((c, i) => `<div class="wz-check"><i>${i + 1}</i><span>${esc(c)}</span></div>`).join("")}</div>
    <div class="row-flex mt14">
      <a class="btn" href="${step.href}">${esc(step.cta)}</a>
    </div>`;
}

function wizardRunsPane(runsR) {
  const runs = (runsR && runsR.runs) || [];
  const done = runs.filter((r) => r.state === "done").length;
  if (!runs.length) {
    return `
      <b class="t">还没有实验</b>
      <div class="empty">先在「模型训练」页给实验命名、选择分组（基线 / 改进 / 消融），再用前面的设置跑通第一个实验；这里会显示实时状态。</div>
      <div class="row-flex"><a class="btn accent" href="#/train">开始第一个实验</a></div>`;
  }
  return `
    <b class="t">实验状态（${done} / ${runs.length} 已完成）</b>
    <div class="table-scroll">${runsTable(runs.slice(0, 8))}</div>
    <div class="hint mt8">实验命名与分组（基线 / 改进 / 消融）在训练页填写，后面对比和报告会自动按组整理。</div>
    <div class="row-flex mt14">
      <a class="btn" href="#/train">再开一个实验</a>
      <a class="btn accent" href="#/runs">查看全部实验</a>
    </div>`;
}

function wizardAutoNote(stepKey, dsR, runsR) {
  const dsCount = ((dsR && dsR.datasets) || []).length;
  const runs = (runsR && runsR.runs) || [];
  const done = runs.filter((r) => r.state === "done").length;
  let msg = "";
  if (stepKey === "2" && dsCount > 0) {
    msg = `系统已检测到 ${dsCount} 个已载入数据集，本步条件满足，可以保存并继续。`;
  } else if (stepKey === "6" && runs.length > 0) {
    msg = `系统已检测到 ${runs.length} 个实验记录（${done} 个已完成）。至少有 1 个完成实验即满足本步条件，仍建议手动标记完成以留下记录。`;
  } else if (stepKey === "7" && done > 0) {
    msg = `系统已检测到 ${done} 个完成实验，评估图表会随实验详情自动生成。`;
  } else if (stepKey === "8" && done > 0) {
    msg = `已有 ${done} 个完成实验可作为消融的起点，到训练页或实验记录页一键生成消融清单。`;
  } else if (stepKey === "9" && done >= 2) {
    msg = `已有 ${done} 个完成实验满足对比条件，可直接生成对比表。`;
  } else if (stepKey === "10" && done > 0) {
    msg = `已有 ${done} 个完成实验可写入报告第四章初稿。`;
  }
  return msg ? `<div class="wz-auto-note"><b>系统自动检测：</b>${esc(msg)}</div>` : "";
}

async function renderWizard() {
  await loadWizard();
  const stepNo = Math.max(1, Math.min(10, wizardState.active_step || 1));
  const step = WIZARD_STEPS[stepNo - 1];
  const [dsR, runsR, modelsR] = await Promise.all([
    api("/api/datasets").catch(() => null),
    api("/api/runs").catch(() => null),
    step.key === "4" ? api("/api/models").catch(() => null) : Promise.resolve(null),
  ]);
  wizardCtx = { dsR, modelsR };
  const st = wizardState.steps[step.key] || {};
  const p = wizardState.project || {};
  const doneCount = WIZARD_STEPS.filter((s) => (wizardState.steps[s.key] || {}).completed).length;
  const page = $("#page");
  page.className = "page wizard-page";
  page.innerHTML = `
    ${wizardStepsBar(stepNo)}
    <div class="wizard-body">
      <div class="wizard-main">
        <div class="wizard-kicker-row">
          <div class="wizard-kicker">毕设向导 · 第 ${stepNo} 步 / 共 10 步</div>
          <button class="btn small" data-wsnippet>这一步论文怎么写</button>
        </div>
        <h1>${esc(step.name)}</h1>
        <p class="wizard-tip">${esc(step.tip)}</p>
        <div class="panel wizard-panel">
          ${step.key === "1" ? wizardProjectFields()
            : step.key === "2" ? wizardDatasetPane(dsR)
            : step.key === "3" ? wizardDataPrepPane(dsR)
            : step.key === "4" ? wizardNetworkPane(dsR, modelsR)
            : step.key === "5" ? wizardTrainingPane()
            : step.key === "6" ? wizardRunsPane(runsR)
            : step.checks ? wizardChecksPane(step)
            : '<div class="hint">这一步正在补充详细表单，先按推荐流程继续。</div>'}
        </div>
        ${wizardAutoNote(step.key, dsR, runsR)}
        <div id="wz-write-out" class="mt14"></div>
        <div class="wizard-actions">
          <div class="row-flex">
            ${stepNo > 1 ? '<button class="btn" data-wstep="prev">上一步</button>' : ""}
            <button class="btn ghost" data-wstep="skip">跳过本步</button>
            ${st.state === "done" ? '<span class="wizard-done-label">本步已完成</span>' : ""}
          </div>
          <div class="row-flex">
            ${stepNo < 10 ? '<button class="btn" data-wstep="next">下一步</button>' : ""}
            <button class="btn accent" data-wstep="done">${stepNo === 10 ? "完成向导" : "保存并继续"}</button>
          </div>
        </div>
      </div>
      <aside class="wizard-rail">
        <div class="panel">
          <div class="muted small">向导进度</div>
          <div class="wizard-progress"><span style="width:${doneCount * 10}%"></span></div>
          <div class="small" style="margin-top:6px">${doneCount} / 10 步已完成</div>
          <hr class="sep">
          <div class="muted small">当前项目</div>
          <div style="margin-top:6px"><b>${esc(p.title || "未填写题目")}</b></div>
          <div class="muted small mt4">${esc(p.direction || "未选择方向")}</div>
          <hr class="sep">
          <a class="btn" style="width:100%;justify-content:center" href="#/guide">查看毕设指南</a>
          <a class="btn ghost" style="width:100%;justify-content:center;margin-top:8px" href="#/dashboard">进入完整工作台</a>
        </div>
      </aside>
    </div>`;
  bindWizardPage(step, stepNo, dsR, modelsR);
}

/* ================= 交互 ================= */
function bindWizardPage(step, stepNo, dsR, modelsR) {
  bindWizardTemplates(step, dsR, modelsR);
  const writeBtn = document.querySelector("[data-wsnippet]");
  if (writeBtn) writeBtn.onclick = () => wizardWriting(step.key);
  document.querySelectorAll("[data-wgoto]").forEach((b) => {
    b.onclick = () => wizardGoTo(Number(b.dataset.wgoto));
  });
  document.querySelectorAll("[data-wfield]").forEach((el) => {
    el.onchange = () => updateWizardProject(el.dataset.wfield, el.value);
    if (el.tagName !== "SELECT") {
      el.oninput = () => updateWizardProject(el.dataset.wfield, el.value);
    }
  });
  document.querySelectorAll("[data-builtin]").forEach((b) => {
    b.onclick = async () => {
      b.disabled = true;
      b.textContent = "载入中…";
      try {
        const r = await api("/api/datasets/builtin", { method: "POST", body: { name: b.dataset.builtin } });
        const ds = r.dataset || {};
        const inferred = wizardInferDirection(ds.task, ds.type);
        const direction = (wizardState.project || {}).direction;
        if (!wizardDirectionMatches(direction, ds.task, ds.type)) {
          wizardState.project.direction = inferred;
          await saveWizard();
          toast(`已载入，并把研究大类调整为「${inferred}」以匹配数据集`, "success");
        } else {
          toast(`已载入：${r.dataset.name}`, "success");
        }
        renderWizard().catch((e) => toast(e.message, "error"));
      } catch (e) {
        toast(e.message, "error");
        b.disabled = false;
        b.textContent = "载入";
      }
    };
  });
  document.querySelectorAll("[data-wgoto-page]").forEach((a) => {
    a.onclick = () => (location.hash = a.dataset.wgotoPage);
  });
  const doneBtn = document.querySelector('[data-wstep="done"]');
  if (doneBtn) {
    doneBtn.onclick = async () => {
      doneBtn.disabled = true;
      try {
        await markWizardStep(step.key, "done");
        if (stepNo === 10) toast("10 步已全部完成，可以开始写论文了", "success");
        wizardGoTo(stepNo + 1);
      } catch (e) { doneBtn.disabled = false; }
    };
  }
  const skipBtn = document.querySelector('[data-wstep="skip"]');
  if (skipBtn) {
    skipBtn.onclick = async () => {
      skipBtn.disabled = true;
      try {
        await markWizardStep(step.key, "skipped");
        wizardGoTo(stepNo + 1);
      } catch (e) { skipBtn.disabled = false; }
    };
  }
  const prevBtn = document.querySelector('[data-wstep="prev"]');
  if (prevBtn) prevBtn.onclick = () => wizardGoTo(stepNo - 1);
  const nextBtn = document.querySelector('[data-wstep="next"]');
  if (nextBtn) nextBtn.onclick = () => wizardGoTo(stepNo + 1);
}

function bindWizardTemplates(step, dsR, modelsR) {
  if (step.key === "3") {
    document.querySelectorAll("[data-wprep]").forEach((el) => {
      el.onchange = () => {
        const t = wizardTemplate("3");
        t[el.dataset.wprep] = el.value === "true" ? true : el.value === "false" ? false : el.value;
        saveWizard().catch(() => {});
      };
    });
    document.querySelectorAll("[data-wprep-num]").forEach((el) => {
      el.onchange = () => {
        const t = wizardTemplate("3");
        t[el.dataset.wprepNum] = parseFloat(el.value);
        saveWizard().catch(() => {});
      };
    });
    return;
  }
  if (step.key === "4") {
    const taskSel = document.querySelector("[data-wtask]");
    if (taskSel) {
      taskSel.onchange = async (e) => {
        const t = wizardTemplate("4");
        t.task = e.target.value;
        const def = (modelsR && modelsR.catalog && modelsR.catalog[t.task]) || {};
        t.arch = Object.keys(def.models || {})[0] || "";
        t.params = {};
        await saveWizard();
        renderWizard().catch((err) => toast(err.message, "error"));
      };
    }
    document.querySelectorAll("[data-warch]").forEach((el) => {
      el.onclick = async () => {
        const t = wizardTemplate("4");
        t.arch = el.dataset.warch;
        t.params = {};
        await saveWizard();
        renderWizard().catch((err) => toast(err.message, "error"));
      };
    });
    document.querySelectorAll("[data-warch-param]").forEach((el) => {
      el.onchange = () => {
        const t = wizardTemplate("4");
        t.params[el.dataset.warchParam] = el.value;
        saveWizard().catch(() => {});
        clearTimeout(wizardEstimateTimer);
        wizardEstimateTimer = setTimeout(wizardEstimate, 400);
      };
    });
    wizardEstimate();
    return;
  }
  if (step.key === "5") {
    document.querySelectorAll("[data-wtrain]").forEach((el) => {
      el.onchange = () => {
        const t = wizardTemplate("5");
        t[el.dataset.wtrain] = el.value;
        saveWizard().catch(() => {});
      };
    });
    document.querySelectorAll("[data-wtrain-num]").forEach((el) => {
      el.onchange = () => {
        const t = wizardTemplate("5");
        t[el.dataset.wtrainNum] = parseFloat(el.value);
        saveWizard().catch(() => {});
      };
    });
    wizardEnv();
  }
}

function updateWizardProject(field, value) {
  if (!wizardState) return;
  wizardState.project[field] = value;
  saveWizard().catch(() => {});
}
