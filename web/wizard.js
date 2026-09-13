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
    checks: [
      "固定随机种子（默认 42），保证实验可复现",
      "先划分训练 / 验证 / 测试，再执行预处理",
      "缺失值填充与标准化只在训练集上拟合",
      "测试集只评估一次，杜绝数据泄漏",
    ],
    href: "#/train", cta: "到训练页设置划分",
  },
  {
    key: "4", name: "任务与网络架构", tip: "根据你的数据选任务，再选能完成任务的神经网络；表格、图像、文本三类都有覆盖。",
    checks: [
      "表格数据：MLP 多层感知机",
      "图像数据：CNN / ResNet18",
      "文本数据：LSTM / GRU / TextCNN / 轻量 Transformer",
      "按任务自动给出推荐架构与参数量预览",
    ],
    href: "#/train", cta: "去训练页选网络",
  },
  {
    key: "5", name: "优化与训练策略", tip: "SGD、Momentum、Adam、AdamW、RMSprop 都可以选；不会调参就保持推荐值，先跑通再优化。",
    checks: [
      "SGD / Momentum / Adam / AdamW / RMSprop 全部可选",
      "学习率、权重衰减、梯度裁剪、批次大小可调",
      "学习率调度器与早停按推荐值自动开启",
      "统一用同一个随机种子和评价口径，实验才有可比性",
    ],
    href: "#/train", cta: "去训练页调策略",
  },
  { key: "6", name: "训练与监控", tip: "给实验起名、选分组，然后一键开跑；训练过程中可以随时看实时日志和曲线。" },
  {
    key: "7", name: "评估分析", tip: "用独立测试集看模型到底行不行，再让软件帮你解读，结论才有说服力。",
    checks: [
      "独立测试集指标：准确率 / F1 / AUC 等",
      "混淆矩阵、PR / ROC 曲线、损失曲线自动生成",
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

const WIZARD_DIRECTIONS = ["图像分类", "文本分类", "表格预测", "算法优化与消融", "其他"];

let wizardState = null;
let wizardSaveQueue = Promise.resolve();

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
  wizardState.steps[key] = {
    state: action,
    completed: action === "done",
    saved_at: savedAt,
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
  const taskName = (t) => (t === "image_classification" ? "图像" : t === "text_classification" ? "文本" : "表格");
  return `
    <b class="t">内置数据集（可直接载入，自动完成 EDA 与 AI 解读）</b>
    <div class="wz-ds-grid">
      ${builtin.map((b) => `
        <div class="wz-ds-card">
          <div class="name">${esc(b.name)}</div>
          <div class="desc">${esc(b.desc || "")}</div>
          <div class="foot"><span class="tag">${taskName(b.task)}</span>
          <button class="btn small accent" data-builtin="${esc(b.key)}">载入</button></div>
        </div>`).join("")}
    </div>
    <hr class="sep">
    <b class="t">当前已载入 ${datasets.length} 个数据集</b>
    ${datasets.length ? `
      <div class="table-scroll"><table class="data"><thead><tr><th>名称</th><th>类型</th><th>规模</th><th>状态</th></tr></thead>
      <tbody>${datasets.map((d) => `<tr>
        <td><b>${esc(d.name)}</b></td>
        <td>${d.type === "image" ? "图像" : "表格"}</td>
        <td>${d.type === "image" ? (d.n_images || "?") + " 张 / " + (d.n_classes || "?") + " 类" : esc((d.n_rows || "?") + " 行 × " + (d.columns || []).length + " 列")}</td>
        <td><span class="status done">已就绪</span></td></tr>`).join("")}</tbody></table></div>`
      : `<div class="empty">还没有数据集。先载入一个内置数据集，或到「数据集」页上传你自己的文件。</div>`}
    <div class="row-flex mt14">
      <button class="btn" data-wgoto-page="#/datasets">去数据集页上传 / 下载</button>
    </div>`;
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
      <div class="empty">先去「模型训练」页用前面的设置跑通第一个实验，这里会显示实时状态。</div>
      <div class="row-flex"><a class="btn accent" href="#/train">开始第一个实验</a></div>`;
  }
  return `
    <b class="t">实验状态（${done} / ${runs.length} 已完成）</b>
    <div class="table-scroll">${runsTable(runs.slice(0, 8))}</div>
    <div class="row-flex mt14">
      <a class="btn" href="#/train">再开一个实验</a>
      <a class="btn accent" href="#/runs">查看全部实验</a>
    </div>`;
}

async function renderWizard() {
  await loadWizard();
  const [dsR, runsR] = await Promise.all([
    api("/api/datasets").catch(() => null),
    api("/api/runs").catch(() => null),
  ]);
  const stepNo = Math.max(1, Math.min(10, wizardState.active_step || 1));
  const step = WIZARD_STEPS[stepNo - 1];
  const st = wizardState.steps[step.key] || {};
  const p = wizardState.project || {};
  const doneCount = WIZARD_STEPS.filter((s) => (wizardState.steps[s.key] || {}).completed).length;
  const page = $("#page");
  page.className = "page wizard-page";
  page.innerHTML = `
    ${wizardStepsBar(stepNo)}
    <div class="wizard-body">
      <div class="wizard-main">
        <div class="wizard-kicker">毕设向导 · 第 ${stepNo} 步 / 共 10 步</div>
        <h1>${esc(step.name)}</h1>
        <p class="wizard-tip">${esc(step.tip)}</p>
        <div class="panel wizard-panel">
          ${step.key === "1" ? wizardProjectFields()
            : step.key === "2" ? wizardDatasetPane(dsR)
            : step.key === "6" ? wizardRunsPane(runsR)
            : step.checks ? wizardChecksPane(step)
            : '<div class="hint">这一步正在补充详细表单，先按推荐流程继续。</div>'}
        </div>
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
  bindWizardPage(step, stepNo, dsR);
}

/* ================= 交互 ================= */
function bindWizardPage(step, stepNo, dsR) {
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
        toast(`已载入：${r.dataset.name}`, "success");
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

function updateWizardProject(field, value) {
  if (!wizardState) return;
  wizardState.project[field] = value;
  saveWizard().catch(() => {});
}
