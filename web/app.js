/* 毕设工坊 ThesisForge 前端（原生 JS，无构建依赖） */
"use strict";

/* ================= 工具 ================= */
const $ = (s, p = document) => p.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const timeoutMs = Number(opts.timeoutMs || 0);
  const controller = timeoutMs ? new AbortController() : null;
  if (controller) setTimeout(() => controller.abort(), timeoutMs);
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    signal: controller ? controller.signal : undefined,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  let data = null;
  try { data = await res.json(); } catch { /* ignore */ }
  if (!res.ok) throw new Error((data && data.detail) || `请求失败 (${res.status})`);
  return data;
}

function toast(msg, type = "") {
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}

/* pywebview 不弹原生 confirm，统一用应用内面板，避免“弹窗啥也没有”。 */
function confirmInApp(title, message, okText = "确认删除") {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.innerHTML = `<div class="modal-box" role="alertdialog" aria-modal="true">
      <div class="modal-title">${esc(title)}</div>
      <div class="modal-msg">${esc(message)}</div>
      <div class="row-flex mt8 righted">
        <button class="btn small" data-act="cancel">取消</button>
        <button class="btn danger small ml8" data-act="ok">${esc(okText)}</button>
      </div>
    </div>`;
    const close = (result) => { overlay.remove(); resolve(result); };
    overlay.querySelector('[data-act="cancel"]').onclick = () => close(false);
    overlay.querySelector('[data-act="ok"]').onclick = () => close(true);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(false); });
    document.body.appendChild(overlay);
    const ok = overlay.querySelector('[data-act="ok"]');
    if (ok) ok.focus();
  });
}

function mdToHtml(text) {
  const lines = String(text || "").split("\n");
  const out = [];
  let inList = false;
  for (const raw of lines) {
    // 先整体转义再插入标签：顺序反了会把 <b>/<code> 一起转义掉，
    // 也才不会让模型输出里的原始 HTML 进入 DOM。
    const line = esc(raw).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");
    const isLi = /^[-*] /.test(line) || /^\d+\. /.test(line);
    if (isLi && !inList) { out.push("<ul>"); inList = true; }
    if (!isLi && inList) { out.push("</ul>"); inList = false; }
    if (/^#{1,4} /.test(line)) out.push(`<h4>${line.replace(/^#{1,4} /, "")}</h4>`);
    else if (isLi) out.push(`<li>${line.replace(/^([-*]|\d+\.) /, "")}</li>`);
    else if (line.startsWith("&gt; ")) out.push(`<div class="muted small">${line.slice(5)}</div>`);
    else if (line.trim()) out.push(`<div>${line}</div>`);
  }
  if (inList) out.push("</ul>");
  return out.join("");
}

function fmtNum(v) {
  if (typeof v !== "number") return String(v);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  return v.toPrecision(4);
}

function loadReportSel() {
  try { const v = JSON.parse(localStorage.getItem("tf_report_sel") || "[]"); return Array.isArray(v) ? v : []; }
  catch { return []; }
}

function saveReportSel() {
  localStorage.setItem("tf_report_sel", JSON.stringify(state.reportSelected || []));
}

function loadCompareSel() {
  try { const v = JSON.parse(localStorage.getItem("tf_compare_sel") || "[]"); return Array.isArray(v) ? v : []; }
  catch { return []; }
}

function saveCompareSel() {
  localStorage.setItem("tf_compare_sel", JSON.stringify(state.compareSel || []));
}

function tfBridge() {
  return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
}

const GROUPS = {
  baseline: { label: "基线", cls: "baseline" },
  improved: { label: "改进", cls: "improved" },
  ablation: { label: "消融", cls: "ablation" },
  custom: { label: "自定义", cls: "custom" },
};

function groupBadge(g) {
  const d = GROUPS[g] || { label: g || "基线", cls: "custom" };
  return `<span class="badge ${d.cls}">${esc(d.label)}</span>`;
}

function datasetTaskLabel(d) {
  if (d.type !== "image") return "表格";
  if (d.task === "object_detection") return "检测";
  if (d.task === "semantic_segmentation") return "分割";
  return "图像";
}

function datasetQualityHtml(meta, compact = false) {
  const warnings = (meta && meta.quality_warnings) || [];
  if (!warnings.length) return "";
  return `<div class="quality-warning ${compact ? "compact" : ""}">
    <b>数据质量提醒</b>
    <ul>${warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>
  </div>`;
}

/* 折线图（纯 SVG） */
function lineChart(series, opts = {}) {
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) return `<div class="empty">暂无数据</div>`;
  const W = opts.width || 620, H = opts.height || 240, PL = 48, PR = 14, PT = 12, PB = 30;
  let min = Math.min(...pts), max = Math.max(...pts);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.08;
  min = min - pad;
  max = max + pad;
  if (Math.min(...pts) >= 0 && min < 0) min = 0;
  const n = Math.max(...series.map((s) => s.points.length));
  const X = (i) => PL + (n <= 1 ? (W - PL - PR) / 2 : (i * (W - PL - PR)) / (n - 1));
  const Y = (v) => PT + (1 - (v - min) / (max - min)) * (H - PT - PB);
  let grid = "";
  for (let t = 0; t <= 4; t++) {
    const v = min + ((max - min) * t) / 4;
    grid += `<line x1="${PL}" x2="${W - PR}" y1="${Y(v)}" y2="${Y(v)}" stroke="#eceee9"/>` +
      `<text x="${PL - 7}" y="${Y(v) + 3.5}" text-anchor="end" font-size="10" fill="#8a9095">${fmtNum(v)}</text>`;
  }
  const step = Math.ceil(n / 8);
  for (let i = 0; i < n; i += step) grid += `<text x="${X(i)}" y="${H - 14}" text-anchor="middle" font-size="10" fill="#8a9095">${i + 1}</text>`;
  if (opts.xLabel) grid += `<text x="${W - PR}" y="${H - 2}" text-anchor="end" font-size="10" fill="#a6acb0">${esc(opts.xLabel)}</text>`;
  const lines = series.map((s) =>
    `<polyline fill="none" stroke="${s.color}" stroke-width="1.8" stroke-linejoin="round" points="${s.points.map((p, i) => `${X(i).toFixed(1)},${Y(p).toFixed(1)}`).join(" ")}"/>` +
    s.points.map((p, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(p).toFixed(1)}" r="2" fill="${s.color}"/>`).join("")
  ).join("");
  const legend = series.map((s) =>
    `<span style="display:inline-flex;align-items:center;gap:5px;margin-left:14px;font-size:11.5px;color:#6e757b"><span style="width:9px;height:2.5px;background:${s.color}"></span>${esc(s.name)}</span>`
  ).join("");
  return `<div style="margin-bottom:2px">${legend}</div>
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${grid}${lines}</svg>`;
}

const C = { teal: "#0d6e63", orange: "#c2410c", ink: "#1d2023", gray: "#9aa1a6", purple: "#6d5aa8" };

const state = {
  health: null, config: null, datasets: [], builtin: [], runs: [], reports: [],
  train: { datasetId: "", task: "", model: "", params: {}, prep: null, testSize: 0.2, valSplit: 0.2, seed: 42, textColumn: "", target: "", name: "", group: "baseline", note: "", tuning: { n_trials: 8, metric: "primary_metric", search_mode: "tpe", results: null, loading: false } },
  openRun: null, openDataset: null, pollTimer: null,
  reportSelected: loadReportSel(), compareSel: loadCompareSel(),
};

/* ================= 路由 ================= */
const ROUTES = { dashboard: pageDashboard, datasets: pageDatasets, train: pageTrain, runs: pageRuns, compare: pageCompare, report: pageReport, ai: pageAI, guide: pageGuide };

function route() {
  const name = (location.hash || "#/wizard").replace(/^#\//, "") || "wizard";
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === name));
  if (name === "wizard") {
    renderWizard().catch((e) => { $("#page").innerHTML = `<div class="panel"><div class="empty">页面加载失败：${esc(e.message)}</div></div>`; });
    return;
  }
  const fn = ROUTES[name] || pageDashboard;
  fn().catch((e) => { $("#page").innerHTML = `<div class="panel"><div class="empty">页面加载失败：${esc(e.message)}</div></div>`; });
}
window.addEventListener("hashchange", route);

/* ================= 引导教程 ================= */
const TOUR = [
  ["欢迎使用毕设工坊",
   "这是一个<b>本地运行</b>的毕业设计工作台：找数据、训练模型、分析结果、写论文初稿，都在网页里完成，不需要写命令行。<br><br>花一分钟看完这个引导，然后照着<span class=\"hl\">总览页的清单</span>一步步做即可。"],
  ["整体流程：四步串起来",
   "<span class='hl'>1</span> 数据集页：载入内置数据集或导入自己的 CSV / 图像打包；<br><span class='hl'>2</span> 模型训练页：选模型、表单调参、一键训练；<br><span class='hl'>3</span> 实验记录页：看曲线、混淆矩阵，点「AI 分析」拿改进建议；<br><span class='hl'>4</span> 报告工坊：勾选实验，导出论文 Word 初稿。"],
  ["建议的第一组实验",
   "不知道从哪开始？照这个来：<br><br>数据集页载入「<span class='hl'>乳腺癌威斯康星</span>」→ 训练页先用<span class='hl'>逻辑回归</span>跑一个基线 → 再换<span class='hl'>随机森林</span>、<span class='hl'>梯度提升树</span>各跑一次 → 实验记录里对比三者指标。全流程约五分钟。"],
  ["AI 分析怎么用",
   "每个数据集和实验详情页都有「AI 分析」按钮。<br><br>不配置也能用（内置规则分析器）；在<span class='hl'>AI 设置</span>页填入大模型接口后，分析会更有深度，报告工坊还能自动起草各章节正文。智谱 glm-4-flash 有免费额度，学生够用。"],
  ["写论文时",
   "「报告工坊」按毕业论文的结构（摘要、绪论、数据与方法、实验分析、总结、参考文献）生成 Word 初稿，实验表格和图表自动插入，导出后按学校模板微调即可。<br><br>「毕设指南」页有完整的九步路线图和避坑清单。"],
];

function showTour(from = 0) {
  const root = $("#tour-root");
  let i = from;
  const render = () => {
    const [title, body] = TOUR[i];
    root.innerHTML = `
      <div class="tour-mask">
        <div class="tour-panel">
          <div class="tour-step-no">新手引导 · ${i + 1} / ${TOUR.length}</div>
          <div class="tour-title">${title}</div>
          <div class="tour-body">${body}</div>
          <div class="tour-foot">
            <div class="tour-dots">${TOUR.map((_, k) => `<i class="${k === i ? "on" : ""}"></i>`).join("")}</div>
            <div class="row-flex">
              <button class="btn small" id="tour-skip">跳过</button>
              ${i > 0 ? '<button class="btn small" id="tour-prev">上一步</button>' : ""}
              <button class="btn primary small" id="tour-next">${i === TOUR.length - 1 ? "开始使用" : "下一步"}</button>
            </div>
          </div>
        </div>
      </div>`;
    $("#tour-skip").onclick = close;
    if ($("#tour-prev")) $("#tour-prev").onclick = () => { i--; render(); };
    $("#tour-next").onclick = () => {
      if (i === TOUR.length - 1) return close();
      i++; render();
    };
  };
  const close = () => {
    root.innerHTML = "";
    localStorage.setItem("tf_onboarded", "1");
  };
  render();
}

/* ================= 总览：进度清单驱动 ================= */
function pageHead(no, title, desc) {
  return `<div class="page-head"><h1 class="page-title"><span class="no">${no}</span>${title}</h1><div class="page-desc">${desc}</div></div>`;
}

async function pageDashboard() {
  const [health, runsR, dsR, cfg, repR] = await Promise.all([
    api("/api/health"), api("/api/runs"), api("/api/datasets"), api("/api/config"), api("/api/report/list"),
  ]);
  Object.assign(state, { health, config: cfg, runs: runsR.runs, datasets: dsR.datasets, builtin: dsR.builtin, reports: repR.reports });
  updateSidebar();
  const done = state.runs.filter((r) => r.state === "done");
  const checklist = [
    { done: state.datasets.length > 0, t: "准备一个数据集", d: "载入内置数据集，或导入你的 CSV / 图像打包 zip", href: "#/datasets", cta: "去数据集" },
    { done: state.runs.length > 0, t: "训练第一个基线模型", d: "先跑最简单的模型拿到第一组指标", href: "#/train", cta: "去训练" },
    { done: done.length >= 2, t: "再训练 1-2 个对比模型", d: "论文需要对比表：基线 vs 更强的模型", href: "#/train", cta: "继续训练" },
    { done: done.length > 0 && (state.config || {}).configured !== null && done.length > 0 && state.runs.some((r) => r.state === "done"), t: "查看结果并生成 AI 分析", d: "实验详情页查看曲线与混淆矩阵，点「AI 分析」拿改进建议", href: "#/runs", cta: "看结果" },
    { done: state.reports.length > 0, t: "导出论文初稿", d: "勾选实验自动生成五章结构的 Word 文档", href: "#/report", cta: "去报告" },
  ];
  const nextIdx = checklist.findIndex((c) => !c.done);
  $("#page").innerHTML = `
    ${pageHead("00", "总览", "从数据到论文初稿的工作台。按顺序完成下面的清单，就走出了一条完整的毕设实验线。")}
    <div class="panel check-list">
      ${checklist.map((c, i) => `
        <div class="check-item ${c.done ? "done" : ""}">
          <div class="mark">${c.done ? "✓" : i + 1}</div>
          <div class="txt"><div class="t">${c.t}</div><div class="d">${c.d}</div></div>
          ${!c.done && i === nextIdx ? `<a class="btn accent small" href="${c.href}">${c.cta}</a>` : ""}
        </div>`).join("")}
    </div>
    ${(state.config || {}).configured ? "" : `<div class="panel mt14" style="border-left:3px solid var(--warn)">
      <b>AI 分析尚未配置</b><div class="muted small mt8">不配置也能用（内置规则分析器）。配置大模型接口后，可对结果做深度解读、自动起草论文章节。<button class="btn-link" id="goto-ai">去配置</button>　<button class="btn-link" id="re-tour">重看新手引导</button></div>
    </div>`}
    <div class="section"><h2>环境</h2>
      <div class="panel"><div class="kv-row" style="margin-top:0">
        <div class="kv"><b>${state.datasets.length}</b>数据集</div>
        <div class="kv"><b>${state.runs.length}</b>实验（${done.length} 个完成）</div>
        <div class="kv"><b>${state.reports.length}</b>已导出报告</div>
        <div class="kv"><b>${health.cuda ? "可用" : "CPU"}</b>GPU ${health.gpu_name ? esc(health.gpu_name) : ""}</div>
        <div class="kv"><b>${(state.config || {}).configured ? "已配置" : "未配置"}</b>AI 接口</div>
      </div></div>
    </div>
    <div class="section"><h2>最近实验</h2>
      <div class="panel">${state.runs.length ? runsTable(state.runs.slice(0, 6)) : '<div class="empty">还没有实验。完成上面清单的第 2 步，这里就会出现记录。</div>'}</div>
    </div>`;
  const g = $("#goto-ai"); if (g) g.onclick = () => (location.hash = "#/ai");
  const r2 = $("#re-tour"); if (r2) r2.onclick = () => showTour(0);
}

/* ================= 数据集页 ================= */
async function pageDatasets() {
  const dsR = await api("/api/datasets");
  state.datasets = dsR.datasets; state.builtin = dsR.builtin;
  $("#page").innerHTML = `
    ${pageHead("01", "数据集", "内置经典数据集可直接载入；自己的数据支持 CSV / Excel 表格与按「类别文件夹/图片」打包的图像 zip；也可以粘贴公网直链下载。")}
    <div class="panel">
      <div class="form-grid">
        <div class="form-row"><label>载入内置数据集</label>
          <div class="row-flex"><select id="ds-builtin-pick" style="flex:1">
            ${state.builtin.map((b) => `<option value="${esc(b.key)}">${esc(b.name)}</option>`).join("")}</select>
          <button class="btn accent" id="btn-builtin">载入</button></div>
          <div class="hint" id="ds-builtin-desc"></div>
        </div>
        <div>
          <div class="form-row"><label>导入本地文件（.csv / .xlsx / .zip）</label>
            <div class="row-flex"><input type="file" id="ds-file" accept=".csv,.xlsx,.xls,.zip" style="padding:5px">
            <button class="btn accent" id="btn-import">导入</button></div>
          </div>
          <div class="form-row"><label>从 URL 下载（公网 http/https 直链，最大 1GB）</label>
            <div class="row-flex"><input id="ds-url" placeholder="https://example.com/dataset.zip" style="flex:1">
            <button class="btn" id="btn-download">下载</button></div>
          </div>
        </div>
      </div>
    </div>
    <div class="section"><h2>已载入的数据集（点击查看详情）</h2>
      <div class="panel">${state.datasets.length ? `
        <table class="data"><thead><tr><th>名称</th><th>类型</th><th>规模</th><th>来源</th><th>载入时间</th><th></th></tr></thead>
        <tbody>${state.datasets.map((d) => `<tr class="clickable" data-ds="${esc(d.id)}">
          <td><b>${esc(d.name)}</b></td><td>${esc(datasetTaskLabel(d))}</td>
          <td>${d.type === "image" ? (d.n_images || "?") + " 张 / " + (d.n_classes || "?") + " 类" : esc((d.n_rows || "?") + " 行 × " + (d.columns || []).length + " 列")}</td>
          <td class="muted">${{ builtin: "内置", imported: "导入", download: "下载" }[d.source] || d.source}</td>
          <td class="muted small">${esc(d.created_at || "")}</td>
          <td><button class="btn danger small" data-del-ds="${esc(d.id)}">删除</button></td></tr>`).join("")}
        </tbody></table>` : '<div class="empty">还没有数据集。用上面的「载入内置数据集」添加第一个。</div>'}</div>
    </div>
    <div id="ds-detail"></div>`;

  const pick = $("#ds-builtin-pick");
  const showBuiltinDesc = () => {
    const b = state.builtin.find((x) => x.key === pick.value);
    $("#ds-builtin-desc").textContent = b ? b.desc : "";
  };
  pick.onchange = showBuiltinDesc; showBuiltinDesc();
  $("#btn-builtin").onclick = async () => {
    $("#btn-builtin").disabled = true;
    try {
      const r = await api("/api/datasets/builtin", { method: "POST", body: { name: pick.value } });
      toast(`已载入并完成分析：${r.dataset.name}`, "success");
      pageDatasets();
    } catch (e) { toast(e.message, "error"); $("#btn-builtin").disabled = false; }
  };
  $("#btn-import").onclick = async () => {
    const f = $("#ds-file").files[0];
    if (!f) return toast("请先选择文件", "error");
    const fd = new FormData(); fd.append("file", f);
    toast("上传解析中…");
    try { const r = await api("/api/datasets/import", { method: "POST", body: fd }); toast(`已导入：${r.dataset.name}`, "success"); pageDatasets(); }
    catch (e) { toast(e.message, "error"); }
  };
  $("#btn-download").onclick = async () => {
    const url = $("#ds-url").value.trim();
    if (!url) return toast("请输入下载链接", "error");
    toast("下载解析中，视网速可能需要一些时间…");
    try { const r = await api("/api/datasets/download", { method: "POST", body: { url } }); toast(`已下载：${r.dataset.name}`, "success"); pageDatasets(); }
    catch (e) { toast(e.message, "error"); }
  };
  document.querySelectorAll("[data-ds]").forEach((tr) => tr.onclick = (e) => { if (e.target.closest("[data-del-ds]")) return; openDataset(tr.dataset.ds); });
  document.querySelectorAll("[data-del-ds]").forEach((b) => b.onclick = async () => {
    if (!(await confirmInApp("删除数据集", "确定删除该数据集？相关实验记录不会自动删除。"))) return;
    try { await api("/api/datasets/" + b.dataset.delDs, { method: "DELETE" }); toast("已删除", "success"); if (state.openDataset === b.dataset.delDs) state.openDataset = null; pageDatasets(); } catch (e) { toast(e.message, "error"); }
  });
  if (state.openDataset) openDataset(state.openDataset);
}

async function openDataset(id) {
  state.openDataset = id;
  document.querySelectorAll("[data-ds]").forEach((tr) => tr.classList.toggle("selected", tr.dataset.ds === id));
  const box = $("#ds-detail");
  box.innerHTML = '<div class="panel mt14 loading">加载详情…</div>';
  try {
    const [meta, prev] = await Promise.all([api("/api/datasets/" + id), api(`/api/datasets/${id}/preview?rows=12`)]);
    const stats = meta.stats || {};
    const statChips = meta.type === "image"
      ? [["类别数", meta.n_classes || "-"], ["图像总数", meta.n_images || "-"]]
      : [["行数", stats.n_rows || meta.n_rows || "-"], ["列数", (meta.columns || []).length], ["类别数", meta.n_classes ?? "-"],
         ["缺失值", Object.values(stats.missing || {}).reduce((a, b) => a + b, 0) || 0]];
    box.innerHTML = `
      <div class="panel mt14">
        <div class="row-flex spread"><b>${esc(meta.name)}</b>
          <button class="btn small" id="btn-ds-ai">AI 分析该数据集</button></div>
        <div class="muted small mt8">${esc(meta.desc || "")}</div>
        <div class="kv-row">${statChips.map(([k, v]) => `<div class="kv"><b>${esc(v)}</b>${k}</div>`).join("")}</div>
        ${datasetQualityHtml(meta)}
        ${meta.eda_files && meta.eda_files.length ? `<div class="figure-grid mt14">${meta.eda_files.map((f) => `
          <figure><img src="/api/datasets/${esc(id)}/eda/${esc(f)}" loading="lazy"><figcaption>${esc(f.replace(".png", ""))}</figcaption></figure>`).join("")}</div>` : ""}
        <hr class="sep"><b class="small">数据预览</b>
        <div style="overflow:auto;max-height:280px" class="mt8">${prev.type === "tabular" ? `
          <table class="data"><thead><tr>${prev.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
          <tbody>${prev.rows.map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table>` :
          `<table class="data"><thead><tr><th>类别</th><th>示例文件</th></tr></thead><tbody>
          ${prev.sample.map((s) => `<tr><td>${esc(s["class"])}</td><td>${esc(s.file)}</td></tr>`).join("")}</tbody></table>`}
        </div>
        <div id="ds-ai-out" class="mt14"></div>
      </div>`;
    $("#btn-ds-ai").onclick = async (e) => {
      e.target.disabled = true; e.target.textContent = "分析中…";
      try {
        const r = await api(`/api/datasets/${id}/analyze`, { method: "POST" });
        $("#ds-ai-out").innerHTML = `<hr class="sep"><div class="ai-source">AI 分析 · 来源：${r.source === "llm" ? "大模型" : "内置规则分析器"}</div><div class="ai-out">${mdToHtml(r.text)}</div>`;
        e.target.textContent = "重新分析"; e.target.disabled = false;
      } catch (err) { toast(err.message, "error"); e.target.disabled = false; e.target.textContent = "AI 分析该数据集"; }
    };
  } catch (e) { box.innerHTML = `<div class="panel mt14 empty">加载失败：${esc(e.message)}</div>`; }
}

/* ================= 训练页 ================= */
async function pageTrain() {
  if (typeof loadWizard === "function") {
    try {
      await loadWizard();
      const prep = wizardState && wizardState.steps && wizardState.steps["3"] && wizardState.steps["3"].template;
      if (prep && typeof prep === "object") {
        state.train = { ...state.train, prep: { ...prep } };
      }
    } catch { /* 向导数据加载失败时不阻塞训练页 */ }
  }
  const [dsR, modelsR] = await Promise.all([api("/api/datasets"), api("/api/models")]);
  state.datasets = dsR.datasets; state.models = modelsR.catalog;
  const t = state.train;
  if (!t.datasetId || !state.datasets.find((d) => d.id === t.datasetId)) {
    t.datasetId = state.datasets.length ? state.datasets[0].id : "";
    t.task = ""; t.model = "";
  }
  const ds = state.datasets.find((d) => d.id === t.datasetId);
  if (ds) {
    const tasks = ds.type === "image"
      ? (ds.task === "object_detection" ? ["object_detection"] : ds.task === "semantic_segmentation" ? ["semantic_segmentation"] : ["image_classification"])
      : ["tabular_classification", "tabular_regression", "text_classification", "time_series_forecasting"];
    if (!tasks.includes(t.task)) t.task = ds.task && tasks.includes(ds.task) ? ds.task : tasks[0];
  }
  const taskDef = state.models[t.task];
  const modelKeys = taskDef ? Object.keys(taskDef.models) : [];
  if (!modelKeys.includes(t.model)) t.model = modelKeys[0] || "";
  const isImage = t.task === "image_classification";
  const prepTxt = t.prep && typeof t.prep === "object"
    ? `${t.prep.missing === "impute" ? "自动填充缺失" : t.prep.missing === "drop" ? "删除缺失行" : "保留缺失"}
       ｜ 编码 ${({onehot:"独热",label:"标签",none:"关闭"})[t.prep.encode] || t.prep.encode || "关闭"}
       ｜ 标准化 ${({standard:"Z-score",minmax:"Min-Max",robust:"稳健",none:"关闭"})[t.prep.scale] || t.prep.scale || "关闭"}
       ｜ 测试集 ${t.prep.test_size ?? 0.2}`
    : "使用系统默认预处理（自动填充、独热编码、标准化）。";

  $("#page").innerHTML = `
    ${pageHead("02", "模型训练", "选数据集，选模型，表单里调参数，点一次按钮开始训练。训练在后台运行，可回到总览再做别的事。")}
    <div class="train-summary">
      <div class="summary-item"><span>数据集</span><b>${esc(ds ? ds.name : "未选择")}</b></div>
      <div class="summary-item"><span>任务 / 模型</span><b>${esc(taskDef ? taskDef.label : "未选择")} · ${esc(taskDef && taskDef.models[t.model] ? taskDef.models[t.model].label : "默认")}</b></div>
      <div class="summary-item"><span>划分</span><b>${t.testSize ?? 0.2}${isImage ? ` / ${t.valSplit ?? 0.2}` : ""} 测试${isImage ? " / 验证" : ""}</b></div>
      <div class="summary-item"><span>随机种子</span><b>${esc(t.seed)}</b></div>
    </div>
    <div class="steps">
      <span class="step-chip ${t.datasetId ? "active" : ""}"><span class="n">一</span>数据集</span>
      <span class="step-chip ${t.task ? "active" : ""}"><span class="n">二</span>任务类型</span>
      <span class="step-chip ${t.model ? "active" : ""}"><span class="n">三</span>模型与参数</span>
    </div>
    <div class="panel">
      <div class="form-grid">
        <div class="form-row"><label>数据集</label>
          <select id="tr-ds">${state.datasets.map((d) => `<option value="${esc(d.id)}" ${d.id === t.datasetId ? "selected" : ""}>${esc(d.name)}（${esc(datasetTaskLabel(d))}）</option>`).join("") || "<option>请先到数据集页载入</option>"}</select></div>
        <div class="form-row"><label>任务类型</label>
          <select id="tr-task">${Object.entries(state.models).filter(([k]) => {
            if (ds && ds.task === "object_detection") return k === "object_detection";
            if (ds && ds.task === "semantic_segmentation") return k === "semantic_segmentation";
            return (ds && ds.type === "image") === (k === "image_classification");
          })
            .map(([k, v]) => `<option value="${esc(k)}" ${k === t.task ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></div>
      </div>
      <div id="tr-textcol"></div>
      <div class="form-grid">
        <div class="form-row"><label>标签列（预测目标）</label>
          <select id="tr-target">${ds && ds.columns ? ds.columns.map((c, i) => `<option value="${esc(c)}" ${(t.target || ds.columns[ds.columns.length - 1]) === c ? "selected" : ""}>${esc(c)}${i === ds.columns.length - 1 ? "（默认）" : ""}</option>`).join("") : "<option value=\"\">使用最后一列</option>"}</select></div>
        <div class="form-row"><label>数据划分与随机种子</label>
          <div class="row-flex">
            <label class="split-field">测试集<input id="tr-testsize" type="number" step="0.05" min="0" max="0.5" value="${t.testSize}"></label>
            <label class="split-field" id="tr-valwrap" ${isImage ? "" : "hidden"}>验证集<input id="tr-valsize" type="number" step="0.05" min="0.05" max="0.5" value="${t.valSplit}"></label>
            <label class="split-field">种子<input id="tr-seed" type="number" value="${t.seed}" title="随机种子"></label>
          </div>
          <div class="hint" id="tr-split-hint"></div></div>
      </div>
      <div class="form-row" style="margin-top:10px"><label>向导预处理策略（来自第 3 步，可直接沿用）</label>
        <div class="hint" id="tr-prep-note">${esc(prepTxt)}</div></div>
    </div>
    <div class="section"><div class="section-head"><h2>选择模型</h2><p>先选一个可解释的基线，再换更强模型对比。浅色高亮表示当前选中。</p></div>
      <div class="model-grid">${modelKeys.map((k) => {
        const m = taskDef.models[k];
        return `<div class="model-item ${k === t.model ? "selected" : ""}" data-model="${esc(k)}">
          <div class="m-row"><div class="m-name">${esc(m.label)}</div>${k === t.model ? '<span class="m-check">已选</span>' : ""}</div>
          <div class="m-desc">${esc(m.desc)}</div>
          <div class="m-tag">${m.local ? "本机权重" : (m.engine === "torch" ? "神经网络" : "模型")}</div></div>`;
      }).join("")}</div>
      <div class="hint">本机 YOLO 权重放到 models/ 或 data/models/ 后重启即可扫描；也可用 THESISFORGE_MODEL_DIR 指定其他目录。</div>
    </div>
    <div class="section"><div class="section-head"><h2>超参数</h2><p>不改也能直接训练；想只改一两个变量时，其他字段保持推荐值即可。</p></div>
      <div class="panel"><div class="form-grid" id="tr-params-form"></div></div>
    </div>
    <div class="section"><div class="section-head"><h2>自动调参</h2><p>适合不确定超参数范围时使用。每个候选都会真实训练一次，结果不会自动加入实验列表。</p></div>
      <div class="panel">
        <div class="form-grid">
          <div class="form-row"><label>搜索轮数</label>
            <input id="tu-trials" type="number" min="2" max="200" value="${esc(t.tuning.n_trials)}" title="每个候选参数会真实训练一次"></div>
          <div class="form-row"><label>评估指标</label>
            <select id="tu-metric">
              <option value="primary_metric" ${t.tuning.metric === "primary_metric" ? "selected" : ""}>主指标（推荐）</option>
              <option value="accuracy" ${t.tuning.metric === "accuracy" ? "selected" : ""}>Accuracy</option>
              <option value="f1" ${t.tuning.metric === "f1" ? "selected" : ""}>F1</option>
              <option value="auc" ${t.tuning.metric === "auc" ? "selected" : ""}>AUC</option>
              <option value="r2" ${t.tuning.metric === "r2" ? "selected" : ""}>R²</option>
              <option value="rmse" ${t.tuning.metric === "rmse" ? "selected" : ""}>RMSE</option>
            </select></div>
          <div class="form-row"><label>搜索模式</label>
            <select id="tu-mode">
              <option value="tpe" ${t.tuning.search_mode === "tpe" ? "selected" : ""}>TPE 贝叶斯搜索（推荐）</option>
              <option value="hyperband" ${t.tuning.search_mode === "hyperband" ? "selected" : ""}>Hyperband 剪枝</option>
              <option value="grid" ${t.tuning.search_mode === "grid" ? "selected" : ""}>小规模网格</option>
            </select></div>
          <div class="form-row"><label>操作</label>
            <button class="btn accent" id="btn-tuning">${t.tuning.loading ? "搜索中…" : "开始自动调参"}</button></div>
        </div>
        <div class="hint">自动调参会按当前模型和参数范围做本地搜索，每个候选都真实训练一次，结果不会自动加入普通实验列表。</div>
        <div id="tu-results"></div>
      </div>
    </div>
    <div class="section"><div class="section-head"><h2>实验命名与分组</h2><p>论文对比表和消融表会按分组整理，起名越具体后面越好找。</p></div>
      <div class="panel">
        <div class="form-grid">
          <div class="form-row"><label>实验名称（可选）</label><input id="tr-name" maxlength="80" value="${esc(t.name || "")}" placeholder="如：基线：逻辑回归"></div>
          <div class="form-row"><label>实验分组</label>
            <select id="tr-group">${Object.entries(GROUPS).map(([k, v]) => `<option value="${k}" ${k === t.group ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select>
          </div>
          <div class="form-row"><label>备注（可选，最多 500 字）</label><input id="tr-note" maxlength="500" value="${esc(t.note || "")}" placeholder="如：固定种子 42，仅改 n_estimators"></div>
        </div>
      </div>
    </div>
    <div class="row-flex mt14" style="margin-top:22px"><button class="btn accent" id="btn-launch" style="font-size:14px;padding:9px 26px">开始训练</button>
    <span class="hint">提交后立即返回，可到「实验记录」看实时进度</span></div>`;

  const renderParams = () => {
    const spec = taskDef && taskDef.models[t.model];
    const form = $("#tr-params-form");
    if (!spec || !Object.keys(spec.params).length) { form.innerHTML = '<div class="hint">该模型没有需要设置的超参数，使用默认配置即可。</div>'; return; }
    form.innerHTML = Object.entries(spec.params).map(([k, ps]) => {
      const v = state.train.params[k] ?? ps.default;
      let input;
      if (ps.type === "choice") input = `<select data-p="${esc(k)}">${ps.options.map((o) => `<option ${o === v ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
      else if (ps.type === "bool") input = `<select data-p="${esc(k)}"><option value="true" ${v ? "selected" : ""}>开启</option><option value="false" ${!v ? "selected" : ""}>关闭</option></select>`;
      else input = `<input data-p="${esc(k)}" type="number" step="${ps.type === "float" ? "0.001" : "1"}" value="${esc(v)}">`;
      return `<div class="form-row"><label>${esc(ps.label || k)}</label>${input}</div>`;
    }).join("");
  };
  renderParams();
  const renderTuning = () => {
    const tu = state.train.tuning;
    const box = $("#tu-results");
    if (!box) return;
    if (!tu.results) { box.innerHTML = ""; return; }
    const rows = (tu.results.history || []).map((h) => `
      <tr><td>${h.number + 1}</td><td>${h.value === null || h.value === undefined ? "-" : esc(fmtNum(h.value))}</td><td>${esc(fmtParams(h.params))}</td></tr>
    `).join("");
    box.innerHTML = `
      <div class="kv-row mt12">
        <div class="kv"><b>${esc(fmtNum(tu.results.best_value))}</b>最佳分数</div>
        <div class="kv"><b>#${tu.results.best_trial + 1}</b>最佳轮次</div>
        <div class="kv"><b>${tu.results.history.length}</b>有效轮次</div>
      </div>
      <div class="table-scroll mt12"><table><thead><tr><th>轮次</th><th>分数</th><th>参数</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  };
  renderTuning();

  $("#tr-ds").onchange = (e) => { state.train = { ...state.train, datasetId: e.target.value, task: "", model: "", target: "", textColumn: "" }; pageTrain(); };
  $("#tr-task").onchange = (e) => { state.train.task = e.target.value; state.train.model = ""; pageTrain(); };
  document.querySelectorAll("[data-model]").forEach((d) => d.onclick = () => { state.train.model = d.dataset.model; state.train.params = {}; pageTrain(); });
  document.querySelectorAll("[data-p]").forEach((inp) => inp.onchange = () => { state.train.params[inp.dataset.p] = inp.value; });
  $("#tr-testsize").onchange = (e) => (state.train.testSize = parseFloat(e.target.value) || 0.2);
  $("#tr-seed").onchange = (e) => (state.train.seed = parseInt(e.target.value) || 42);
  $("#tr-name").oninput = (e) => (state.train.name = e.target.value);
  $("#tr-group").onchange = (e) => (state.train.group = e.target.value);
  $("#tr-note").oninput = (e) => (state.train.note = e.target.value);
  const vs = $("#tr-valsize");
  if (vs) vs.onchange = (e) => (state.train.valSplit = parseFloat(e.target.value) || 0.2);
  $("#tr-split-hint").textContent = isImage
    ? "图像任务按 训练 / 验证 / 测试 三份划分：权重只在训练集更新，best.pt 按验证集挑选，最终指标来自测试集（评估一次，不再回调）。固定随机种子保证可复现，论文里要写。"
    : "先划分再预处理：填充与标准化只在训练集上拟合，测试集只评估一次。固定随机种子保证实验可复现，论文里要写。";
  $("#tr-target").onchange = (e) => (state.train.target = e.target.value);
  $("#tu-trials").onchange = (e) => (state.train.tuning.n_trials = parseInt(e.target.value) || 8);
  $("#tu-metric").onchange = (e) => (state.train.tuning.metric = e.target.value);
  $("#tu-mode").onchange = (e) => (state.train.tuning.search_mode = e.target.value);
  $("#btn-tuning").onclick = async () => {
    const btn = $("#btn-tuning");
    btn.disabled = true; btn.textContent = "搜索中…";
    state.train.tuning.loading = true;
    try {
      const body = {
        dataset_id: state.train.datasetId, task: state.train.task, model: state.train.model,
        params: state.train.params, target: state.train.target || null,
        text_column: state.train.textColumn || null,
        n_trials: state.train.tuning.n_trials,
        metric: state.train.tuning.metric,
        search_mode: state.train.tuning.search_mode,
        seed: state.train.seed,
      };
      const r = await api("/api/tuning/search", { method: "POST", body, timeoutMs: 600000 });
      state.train.tuning.results = r;
      toast(`自动调参完成，最佳分数 ${fmtNum(r.best_value)}`, "success");
    } catch (e) { toast(e.message, "error"); }
    state.train.tuning.loading = false;
    btn.disabled = false; btn.textContent = "开始自动调参";
    renderTuning();
  };

  if (t.task === "text_classification") {
    const box = $("#tr-textcol");
    box.innerHTML = '<div class="form-row"><label>文本列</label><select id="tr-textsel"><option>加载中…</option></select></div>';
    try {
      const prev = await api(`/api/datasets/${t.datasetId}/preview?rows=3`);
      box.innerHTML = `<div class="form-row"><label>文本列（内容为一段文字的列）</label><select id="tr-textsel">
        ${prev.columns.map((c) => `<option ${c === state.train.textColumn ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></div>`;
      $("#tr-textsel").onchange = (e) => (state.train.textColumn = e.target.value);
      if (!state.train.textColumn) state.train.textColumn = prev.columns[0];
    } catch { /* ignore */ }
  }

  $("#btn-launch").onclick = async () => {
    const btn = $("#btn-launch");
    btn.disabled = true; btn.textContent = "提交中…";
    try {
      const body = {
        dataset_id: state.train.datasetId, task: state.train.task, model: state.train.model,
        params: state.train.params, target: state.train.target || null,
        prep: state.train.prep || {},
        text_column: state.train.textColumn || null, test_size: state.train.testSize,
        val_split: state.train.valSplit, random_state: state.train.seed,
        name: state.train.name, group: state.train.group, note: state.train.note,
      };
      const r = await api("/api/runs", { method: "POST", body });
      toast(`实验已启动：${r.run_id}`, "success");
      state.openRun = r.run_id;
      location.hash = "#/runs";
    } catch (e) { toast(e.message, "error"); btn.disabled = false; btn.textContent = "开始训练"; }
  };
}

/* ================= 实验记录页 ================= */
function statusText(s) { return `<span class="status ${esc(s)}">${{ running: "训练中", done: "已完成", failed: "失败", cancelled: "已取消" }[s] || s}</span>`; }

/* 指标来自哪一份数据：test_* 才是独立测试集，val_* 只是调参用的验证集 */
function metricSourceNote(s) {
  if (s.split_scheme === "train/val/test" || (s.n_test || 0) > 0) return "（测试集）";
  if (s.eval_source === "val" || s.n_test === 0) return "（验证集，非测试集）";
  return "";
}

function runsTable(runs) {
  return `<table class="data"><thead><tr><th>实验 ID</th><th>名称 / 分组</th><th>数据集</th><th>模型</th><th>状态</th><th>主指标</th><th>时间</th></tr></thead>
    <tbody>${runs.map((r) => `<tr class="clickable" data-run="${esc(r.run_id)}">
      <td class="small muted">${esc(r.run_id)}</td>
      <td><div class="cell-main">${esc(r.name || r.model_label || r.model || r.run_id)}</div><div class="cell-sub mt4">${groupBadge(r.group)}${r.batch_kind ? `<span class="batch-chip small">${r.batch_kind === "repeats" ? "重复" : "消融"}</span>` : ""}</div></td>
      <td>${esc(r.dataset_name || "-")}</td><td>${esc(r.model_label || r.model || "-")}</td>
      <td>${statusText(r.state)}</td>
      <td>${r.primary_metric ? `<b>${esc(r.primary_metric.name)}</b> = ${fmtNum(r.primary_metric.value)}` : "-"}</td>
      <td class="muted small">${esc(r.created_at || "")}</td></tr>`).join("")}</tbody></table>`;
}

async function pageRuns() {
  const r = await api("/api/runs");
  state.runs = r.runs;
  $("#page").innerHTML = `
    ${pageHead("03", "实验记录", "每次训练自动留档：配置、日志、指标、图表、模型文件。点行查看详情，详情可直接选入报告。")}
    <div class="panel">${state.runs.length ? runsTable(state.runs) : '<div class="empty">暂无实验记录。到「模型训练」页发起第一次训练。</div>'}</div>
    <div id="run-detail"></div>`;
  document.querySelectorAll("[data-run]").forEach((tr) => tr.onclick = () => openRun(tr.dataset.run));
  if (state.openRun) openRun(state.openRun);
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const anyRunning = state.runs.some((r) => r.state === "running");
  if (anyRunning && location.hash.includes("runs")) {
    state.pollTimer = setTimeout(async () => {
      if (!location.hash.includes("runs")) return;
      try { const r = await api("/api/runs"); state.runs = r.runs; if (state.openRun) await openRun(state.openRun, true); } catch { /* ignore */ }
    }, 2500);
  }
}

function fmtParams(params) {
  if (!params || !Object.keys(params).length) return "默认";
  return Object.entries(params).map(([k, v]) => `${k}=${v}`).join(" · ");
}

async function openRun(id, silent = false) {
  state.openRun = id;
  if (!location.hash.includes("runs")) { location.hash = "#/runs"; return; }
  document.querySelectorAll("[data-run]").forEach((tr) => tr.classList.toggle("selected", tr.dataset.run === id));
  const box = $("#run-detail");
  if (!box) return;
  if (!silent) box.innerHTML = '<div class="panel mt14 loading">加载详情…</div>';
  try {
    const d = await api("/api/runs/" + id);
    const s = d.summary || {};
    const metricsHtml = s.metrics ? Object.entries(s.metrics).filter(([, v]) => typeof v === "number")
      .map(([k, v]) => `<div class="kv"><b>${fmtNum(v)}</b>${esc(k)}</div>`).join("") : "";
    const epochRows = (s.epochs || (d.metrics_log || []).filter((e) => e.type === "epoch"));
    const cvScores = s.cv_scores || (d.metrics_log.find((e) => e.type === "cv") || {}).scores;
    const running = d.state === "running";
    box.innerHTML = `
      <div class="panel mt14">
        <div class="row-flex spread">
          <div class="row-flex"><b>${esc(d.name || ("实验 " + id))}</b>${groupBadge(d.group)}${statusText(d.state)}</div>
          <div class="row-flex">
            ${running ? '<button class="btn danger small" id="btn-cancel">取消训练</button>' : ""}
            ${d.state === "done" ? '<button class="btn small" id="btn-repeat">重复 3 次</button><button class="btn small" id="btn-ablation">一键消融</button>' : ""}
            <button class="btn small" id="btn-reuse">复用参数</button>
            <button class="btn small" id="btn-run-ai">AI 分析结果</button>
            <button class="btn danger small" id="btn-del">删除</button>
            <button class="btn small" id="btn-back">返回列表</button>
          </div>
        </div>
        ${d.error ? `<div class="mt8 small" style="color:var(--danger)">训练失败：${esc(d.error).slice(0, 300)}</div>` : ""}
        ${d.failure_hint ? `<div class="panel small mt8" style="background:#fff6e9;border-color:var(--warn);color:#5b4a1f">${esc(d.failure_hint)}</div>` : ""}
        <div class="meta-box mt14">
          <div class="form-grid">
            <div class="form-row"><label>实验名称</label><input id="meta-name" maxlength="80" value="${esc(d.name || "")}" placeholder="${esc(d.config.model_label || d.config.model || "实验名称")}"></div>
            <div class="form-row"><label>实验分组</label>
              <select id="meta-group">${Object.entries(GROUPS).map(([k, v]) => `<option value="${k}" ${k === d.group ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></div>
            <div class="form-row"><label>备注（可选）</label><input id="meta-note" maxlength="500" value="${esc(d.note || "")}"></div>
            <div class="form-row"><label>保存</label><button class="btn small" id="btn-meta-save">保存实验信息</button></div>
          </div>
        </div>
        <div class="row-flex mt8 small muted">
          数据集 ${esc(d.config.dataset_name || "-")} ｜ 模型 ${esc(d.config.model_label || d.config.model || "-")}
          ｜ 参数 ${esc(fmtParams(d.config.params))}
        </div>
        ${d.batch_kind ? `<div class="row-flex mt8 small">${groupBadge(d.group)}<span class="batch-chip">${d.batch_kind === "repeats" ? `重复批次 #${esc(d.repeat_index || "")}` : "消融批次"}</span><span class="muted">${esc(d.batch_id || "")}</span></div>` : ""}
        ${d.state === "done" && s.primary_metric ? `<div class="kv-row"><div class="kv"><b style="color:var(--accent);font-size:23px">${fmtNum(s.primary_metric.value)}</b>主指标 ${esc(s.primary_metric.name)}${esc(metricSourceNote(s))}</div>${metricsHtml}
          <div class="kv"><b>${esc(s.n_train || "-")}/${esc(s.n_val || "-")}/${esc(s.n_test || "-")}</b>训练/验证/测试</div>
          <div class="kv"><b>${esc(s.train_time_sec || "-")}s</b>训练用时</div></div>` : ""}
        ${epochRows.length > 1 ? `<div class="mt14">${lineChart([
          { name: "train_loss", color: C.teal, points: epochRows.map((e) => e.train_loss) },
          { name: "val_loss", color: C.orange, points: epochRows.map((e) => e.val_loss) },
          { name: "val_acc", color: C.ink, points: epochRows.map((e) => e.val_acc) },
        ], { xLabel: "epoch" })}</div>` : ""}
        ${cvScores && cvScores.length ? `<div class="mt14">${lineChart([{ name: "交叉验证得分", color: C.teal, points: cvScores }], { xLabel: "折" })}</div>` : ""}
        ${d.artifacts && d.artifacts.length ? `<div class="figure-grid mt14">${d.artifacts.map((f) => f.endsWith(".png") ? `
          <figure><img src="/api/runs/${esc(id)}/artifacts/${esc(f)}" loading="lazy"><figcaption>${esc(f.replace(".png", ""))}</figcaption></figure>` : `
          <figure><a class="btn small" href="/api/runs/${esc(id)}/artifacts/${esc(f)}" download>下载 ${esc(f)}</a><figcaption>${esc(f)}</figcaption></figure>`).join("")}</div>` : ""}
        <div class="mt14" id="run-ai-out"></div>
        <hr class="sep"><b class="small">运行日志</b> <span class="muted small">${running ? "每 2.5 秒自动刷新" : ""}</span>
        <div class="log-box mt8" id="run-log">${esc(d.log_tail || "（暂无日志）")}</div>
      </div>`;
    $("#btn-back").onclick = () => { state.openRun = null; box.innerHTML = ""; };
    $("#btn-reuse").onclick = () => {
      const cfg = d.config || {};
      state.train = {
        datasetId: cfg.dataset_id || "", task: cfg.task || "", model: cfg.model || "",
        params: { ...(cfg.params || {}) }, prep: (cfg.prep && typeof cfg.prep === "object") ? { ...cfg.prep } : {},
        target: cfg.target || "", textColumn: cfg.text_column || "",
        testSize: cfg.test_size ?? 0.2, valSplit: cfg.val_split ?? 0.2, seed: cfg.random_state ?? 42,
        name: cfg.name || "", group: cfg.group || "baseline", note: cfg.note || "",
      };
      toast("参数已填入训练页", "success");
      location.hash = "#/train";
    };
    const launchBatch = async (count) => {
      const btn = $("#btn-repeat");
      if (btn) { btn.disabled = true; btn.textContent = "提交中…"; }
      try {
        const r = await api("/api/experiments/repeats", { method: "POST", body: { run_id: id, count } });
        toast(`已创建 ${r.run_ids.length} 个重复实验`, "success");
        state.openRun = null;
        pageRuns();
      } catch (e) {
        toast(e.message, "error");
        if (btn) { btn.disabled = false; btn.textContent = "重复 3 次"; }
      }
    };
    if ($("#btn-repeat")) $("#btn-repeat").onclick = () => {
      const old = box.querySelector(".repeat-panel");
      if (old) old.remove();
      const panel = document.createElement("div");
      panel.className = "panel mt14 repeat-panel";
      panel.innerHTML = `<b>重复实验</b><div class="hint">克隆当前实验，仅随机种子不同，跑多次后取均值 ± 标准差。</div>
        <div class="form-grid mt8"><div class="form-row">
          <label for="repeat-count">重复次数（2-10）</label>
          <input id="repeat-count" type="number" min="2" max="10" step="1" value="3">
        </div></div>
        <div class="row-flex mt8"><button class="btn accent small" id="btn-repeat-go">启动重复实验</button><button class="btn small" id="btn-repeat-cancel">取消</button></div>`;
      const anchor = $(".meta-box", box) || box.lastElementChild;
      anchor.after(panel);
      $("#btn-repeat-cancel").onclick = () => panel.remove();
      $("#btn-repeat-go").onclick = () => {
        const n = parseInt($("#repeat-count").value, 10);
        if (!n || n < 2 || n > 10) return toast("次数需在 2-10 之间", "error");
        panel.remove();
        launchBatch(n);
      };
    };
    if ($("#btn-ablation")) $("#btn-ablation").onclick = () => {
      if (d.state !== "done") return;
      const params = d.config.params || {};
      const spec = (((state.models || {})[d.config.task]) || {}).models?.[d.config.model] || {};
      const rows = Object.entries(params).map(([k, base]) => {
        const ps = (spec.params || {})[k] || {};
        let opts;
        if (ps.type === "bool") opts = [true, false].filter((v) => v !== base);
        else if (ps.type === "choice") opts = (ps.options || []).filter((v) => v !== base);
        else if (typeof base === "number") {
          const step = (typeof ps.step === "number" && ps.step > 0) ? ps.step : (Math.abs(base) >= 1 ? 1 : 0.05);
          const scale = Math.abs(base) >= 1 ? Math.max(step, Math.abs(base) * 0.1) : step;
          opts = [base * 0.5, base * 1.5, base + scale, base - scale]
            .map((v) => Math.round(v * 10000) / 10000)
            .filter((v) => (typeof ps.min !== "number" || v >= ps.min) &&
                            (typeof ps.max !== "number" || v <= ps.max))
            .filter((v, i, arr) => arr.indexOf(v) === i)
            .filter((v) => v !== base);
        } else opts = [];
        if (!opts.length) return "";
        return `<div class="form-row"><label>${esc(ps.label || k)}（当前 ${esc(String(base))}）</label>
          <div class="ablation-opts">${opts.map((v) => `<label class="chip-check"><input type="checkbox" data-p="${esc(k)}" value="${esc(v)}">${esc(String(v))}</label>`).join("")}</div></div>`;
      }).filter(Boolean).join("");
      if (!rows) return toast("该实验没有可消融的超参数", "error");
      const panel = document.createElement("div");
      panel.className = "panel mt14 ablation-panel";
      panel.innerHTML = `<b>一键消融</b><div class="hint">每个勾选项派生一个全新实验，只改所选参数，其余参数与源实验保持一致（含随机种子）。</div>
        <div class="form-grid mt8">${rows}</div>
        <div class="row-flex mt8"><button class="btn accent small" id="btn-ablation-go">启动消融实验</button><button class="btn small" id="btn-ablation-cancel">取消</button></div>`;
      const old = box.querySelector(".ablation-panel");
      if (old) old.remove();
      const anchor = $(".meta-box", box) || box.lastElementChild;
      anchor.after(panel);
      $("#btn-ablation-cancel").onclick = () => panel.remove();
      $("#btn-ablation-go").onclick = async () => {
        const overrides = {};
        panel.querySelectorAll("input[type=checkbox]:checked").forEach((c) => {
          let v = c.value;
          if (v === "true") v = true; else if (v === "false") v = false;
          else if (v !== "" && !Number.isNaN(Number(v))) v = Number(v);
          (overrides[c.dataset.p] = overrides[c.dataset.p] || []).push(v);
        });
        if (!Object.keys(overrides).length) return toast("请至少勾选一个消融值", "error");
        const btn = $("#btn-ablation-go");
        btn.disabled = true; btn.textContent = "提交中…";
        try {
          const r = await api("/api/experiments/ablation", { method: "POST", body: { run_id: id, overrides } });
          toast(`已创建 ${r.run_ids.length} 个消融实验`, "success");
          state.openRun = null;
          pageRuns();
        } catch (e) { toast(e.message, "error"); btn.disabled = false; btn.textContent = "启动消融实验"; }
      };
    };
    $("#btn-meta-save").onclick = async () => {
      const btn = $("#btn-meta-save");
      btn.disabled = true;
      try {
        const r = await api(`/api/runs/${id}/meta`, { method: "PATCH", body: {
          name: $("#meta-name").value, group: $("#meta-group").value, note: $("#meta-note").value,
        }});
        Object.assign(d, r.meta);
        const row = state.runs.find((x) => x.run_id === id);
        if (row) Object.assign(row, r.meta);
        toast("实验信息已保存", "success");
        await openRun(id, true);
      } catch (e) { toast(e.message, "error"); }
      if (btn) btn.disabled = false;
    };
    $("#btn-run-ai").onclick = async (e) => {
      if (d.state !== "done") return toast("请等训练完成后再分析", "error");
      e.target.disabled = true; e.target.textContent = "分析中…";
      try {
        const r = await api(`/api/runs/${id}/analyze`, { method: "POST" });
        $("#run-ai-out").innerHTML = `<hr class="sep"><div class="ai-source">AI 分析 · 来源：${r.source === "llm" ? "大模型" : "内置规则分析器"}</div><div class="ai-out">${mdToHtml(r.text)}</div>`;
        e.target.textContent = "重新分析"; e.target.disabled = false;
      } catch (err) { toast(err.message, "error"); e.target.disabled = false; e.target.textContent = "AI 分析结果"; }
    };
    if ($("#btn-cancel")) $("#btn-cancel").onclick = async () => { try { await api(`/api/runs/${id}/cancel`, { method: "POST" }); toast("已发送取消请求", "success"); openRun(id); } catch (e) { toast(e.message, "error"); } };
    $("#btn-del").onclick = async () => {
      if (!(await confirmInApp("删除实验", "确定删除该实验及其所有文件？此操作不可恢复。"))) return;
      try { await api("/api/runs/" + id, { method: "DELETE" }); toast("已删除", "success"); state.openRun = null; pageRuns(); } catch (e) { toast(e.message, "error"); }
    };
    if (running) {
      const logBox = $("#run-log");
      try { const lg = await api(`/api/runs/${id}/log`); logBox.textContent = lg.log || "（暂无日志）"; logBox.scrollTop = logBox.scrollHeight; } catch { /* ignore */ }
    }
  } catch (e) { box.innerHTML = `<div class="panel mt14 empty">加载失败：${esc(e.message)}</div>`; }
  schedulePoll();
}

/* ================= 实验对比 ================= */
function defaultCompareSel(doneRuns) {
  const counts = {};
  for (const r of doneRuns) counts[r.group] = (counts[r.group] || 0) + 1;
  const best = Object.keys(counts).sort((a, b) => (counts[b] || 0) - (counts[a] || 0))[0] || "baseline";
  return doneRuns.filter((r) => r.group === best).slice(0, 4).map((r) => r.run_id);
}

async function pageCompare() {
  const r = await api("/api/runs");
  const done = (r.runs || []).filter((x) => x.state === "done");
  state.runs = r.runs;
  state.compareSel = state.compareSel.filter((rid) => done.some((x) => x.run_id === rid));
  if (!state.compareSel.length) state.compareSel = defaultCompareSel(done);
  saveCompareSel();

  $("#page").innerHTML = `
    ${pageHead("03B", "实验对比", "把已完成实验放到同一张表里做横向比较：主指标条形图、全指标表、热力矩阵，一处看清基线/改进/消融的差异。CSV 和 Markdown 可直接下载或复制进论文。")}
    <div class="panel">
      <b class="t">选择要对比的已完成实验（最多 20 个）</b>
      <div class="compare-picker">${done.length ? done.map((x) => `
        <label class="cmp-item"><input type="checkbox" class="cmp-run" value="${esc(x.run_id)}" ${state.compareSel.includes(x.run_id) ? "checked" : ""}>
          <span class="cmp-name">${esc(x.name || x.model_label || x.model || x.run_id)}</span>${groupBadge(x.group)}
          <span class="muted small">${esc(x.model_label || x.model || "")}</span>
          <span class="muted small">${x.primary_metric ? `${esc(x.primary_metric.name)}=${fmtNum(x.primary_metric.value)}` : ""}</span></label>`).join("")
        : '<div class="empty">还没有已完成实验。先到「模型训练」页完成几组实验，再回来对比。</div>'}</div>
      <div class="row-flex mt14">
        <button class="btn small" id="cmp-same">同组推荐</button>
        <button class="btn small" id="cmp-clear">清空选择</button>
        <button class="btn accent" id="cmp-build">生成对比</button>
        <span class="hint">默认选中最近一组同分组的已完成实验；同组推荐会按基线→改进→消融补满一组。</span>
      </div>
    </div>
    <div id="cmp-result"></div>`;

  const syncSel = () => {
    state.compareSel = [...document.querySelectorAll(".cmp-run:checked")].map((c) => c.value).slice(0, 20);
    saveCompareSel();
  };
  document.querySelectorAll(".cmp-run").forEach((cb) => cb.onchange = syncSel);
  $("#cmp-clear").onclick = () => {
    state.compareSel = [];
    saveCompareSel();
    document.querySelectorAll(".cmp-run").forEach((c) => (c.checked = false));
    $("#cmp-result").innerHTML = "";
  };
  $("#cmp-same").onclick = async () => {
    if (!done.length) return;
    const first = state.compareSel.find((rid) => done.some((x) => x.run_id === rid));
    const anchor = done.find((x) => x.run_id === first) || done[0];
    state.compareSel = done.filter((x) => x.group === anchor.group).slice(0, 20).map((x) => x.run_id);
    saveCompareSel();
    document.querySelectorAll(".cmp-run").forEach((c) => (c.checked = state.compareSel.includes(c.value)));
    toast("已选择同组实验", "success");
    $("#cmp-build").click();
  };
  $("#cmp-build").onclick = async () => {
    syncSel();
    if (!state.compareSel.length) return toast("请先选择至少一个已完成实验", "error");
    const btn = $("#cmp-build");
    btn.disabled = true; btn.textContent = "生成中…";
    try {
      const result = await api("/api/experiments/compare", { method: "POST", body: { run_ids: state.compareSel } });
      renderCompare(result);
      toast(`对比完成：${result.count} 个实验`, "success");
    } catch (e) { toast(e.message, "error"); }
    btn.disabled = false; btn.textContent = "生成对比";
  };
}

function renderCompare(result) {
  const box = $("#cmp-result");
  if (!result || !result.count) {
    box.innerHTML = '<div class="panel mt14 empty">没有可对比的已完成实验，请勾选左上方的实验后重新生成。</div>';
    return;
  }
  const rows = result.metric_rows || [];
  const cols = result.columns || [];
  const primary = rows.find((x) => x.is_primary) || rows[0] || {};
  const pvals = (primary.values || []).map((v) => (typeof v === "number" ? v : null));
  const pmin = Math.min(...pvals.filter((v) => v !== null));
  const pmax = Math.max(...pvals.filter((v) => v !== null));
  const pspan = (pmax - pmin) || 1;
  const barHtml = cols.map((c, i) => {
    const v = pvals[i];
    const h = v === null ? 0 : 14 + Math.round(((v - pmin) / pspan) * 86);
    return `<div class="bar-col" title="${esc(c.label)}：${v === null ? "-" : fmtNum(v)}（${esc(c.group_label)}）">
      <div class="bar-val">${v === null ? "-" : fmtNum(v)}</div>
      <div class="bar-track"><div class="bar-fill ${esc(c.group)}" style="height:${h}%"></div></div>
      <div class="bar-label" title="${esc(c.label)}">${esc((c.label || c.model_label || "").slice(0, 10))}</div>
      <div class="bar-group">${groupBadge(c.group)}</div></div>`;
  }).join("");

  function heatStyle(mr, v) {
    if (typeof v !== "number") return "";
    const nums = (mr.values || []).filter((x) => typeof x === "number");
    const lo = Math.min(...nums), hi = Math.max(...nums);
    const span = (hi - lo) || 1;
    const p = (v - lo) / span;
    const good = mr.higher_is_better ? p : 1 - p;
    const alpha = 0.04 + good * 0.42;
    return `style="background:rgba(13,110,99,${alpha.toFixed(3)})"`;
  }
  const bestCell = (mr, v) => {
    if (typeof v !== "number") return '<td class="muted">-</td>';
    const nums = (mr.values || []).filter((x) => typeof x === "number");
    const best = mr.higher_is_better ? Math.max(...nums) : Math.min(...nums);
    return `<td class="${v === best ? "best" : ""}" ${heatStyle(mr, v)}>${fmtNum(v)}</td>`;
  };

  box.innerHTML = `
    <div class="panel mt14">
      <div class="row-flex spread">
        <b>主指标：${esc(primary.label || "主指标")}</b>
        <div class="row-flex">
          <button class="btn small" id="cmp-csv">下载 CSV</button>
          <button class="btn small" id="cmp-md">复制 Markdown</button>
        </div>
      </div>
      <div class="bar-chart mt14">${barHtml || '<div class="empty">暂无可绘制的数值</div>'}</div>
    </div>
    <div class="section"><h2>指标对比表（最优值加粗高亮）</h2>
      <div class="panel table-scroll">
        <table class="data compare-table"><thead><tr><th>指标</th>${cols.map((c) => `<th>${esc(c.label)}<div class="mt4">${groupBadge(c.group)}</div></th>`).join("")}</tr></thead>
        <tbody>${rows.map((mr) => `<tr><td class="metric-cell">${esc(mr.label)}${mr.higher_is_better ? "" : '<span class="muted small">（越低越好）</span>'}</td>${(mr.values || []).map((v) => bestCell(mr, v)).join("")}</tr>`).join("")}</tbody></table>
      </div>
    </div>
    <div class="section"><h2>热力矩阵</h2>
      <div class="panel table-scroll">
        <div class="heat-grid">
          <div class="heat-col heat-label"><div class="heat-head">指标</div>${rows.map((mr) => `<div class="heat-cell" title="${esc(mr.label)}">${esc(mr.label)}</div>`).join("")}</div>
          ${cols.map((c, ci) => `<div class="heat-col"><div class="heat-head">${esc(c.label)}${groupBadge(c.group)}</div>${rows.map((mr) => {
            const v = (mr.values || [])[ci];
            return `<div class="heat-cell" ${heatStyle(mr, v)} title="${esc(c.label)} · ${esc(mr.label)} = ${typeof v === "number" ? fmtNum(v) : "-"}">${typeof v === "number" ? fmtNum(v) : "-"}</div>`;
          }).join("")}</div>`).join("")}
        </div>
      </div>
    </div>
    <div class="hint mt8">表头按 基线→改进→消融→自定义 排序；每行最优值已高亮，热力颜色代表该行内的相对高低。</div>`;

  $("#cmp-csv").onclick = () => {
    const blob = new Blob(["\ufeff" + (result.csv || "")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "ThesisForge-实验对比.csv";
    document.body.appendChild(a); a.click(); a.remove();
    toast("CSV 已下载", "success");
  };
  $("#cmp-md").onclick = async () => {
    const text = result.markdown || "";
    try { await navigator.clipboard.writeText(text); toast("Markdown 已复制", "success"); }
    catch {
      const ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); ta.remove(); toast("Markdown 已复制", "success");
    }
  };
}

/* ================= 报告工坊 ================= */
async function pageReport() {
  const [runsR, dsR, repR] = await Promise.all([api("/api/runs"), api("/api/datasets"), api("/api/report/list")]);
  const doneRuns = runsR.runs.filter((r) => r.state === "done");
  state.reportSelected = state.reportSelected.filter((rid) => doneRuns.some((r) => r.run_id === rid));
  saveReportSel();
  state.datasets = dsR.datasets; state.reports = repR.reports;
  const readiness = async () => {
    const ids = [...document.querySelectorAll(".rp-run:checked")].map((c) => c.value);
    const box = $("#rp-readiness");
    if (!box) return;
    try {
      const r = await api("/api/report/readiness", { method: "POST", body: {
        title: $("#rp-title").value.trim() || "基于机器学习的毕业设计研究",
        run_ids: ids, dataset_id: $("#rp-ds").value || null,
        author: { school: $("#rp-school").value, college: $("#rp-college").value, major: $("#rp-major").value, name: $("#rp-name").value, student_id: $("#rp-sid").value, advisor: $("#rp-advisor").value },
      }});
      const labels = { ok: "就绪", warn: "建议", fail: "阻断" };
      const colors = { ok: "var(--ok)", warn: "var(--warn)", fail: "var(--danger)" };
      box.innerHTML = `<div class="panel ${r.blocking ? "readiness-blocking" : "readiness-ok"}">
        <div class="row-flex"><b class="t">交付前自检</b>
          <span class="muted">${r.blocking ? "有阻断项，先补齐再生成最终版" : "无阻断项，生成前可参考建议"}</span></div>
        <div class="readiness-grid mt12">${r.checks.map((c) => `
          <div class="readiness-item" style="border-left-color:${colors[c.status]}">
            <b>${esc(c.label)}</b><span class="muted">${labels[c.status]}</span>
            <div class="small">${esc(c.message)}</div>
          </div>`).join("")}</div>
      </div>`;
    } catch (e) { box.innerHTML = `<div class="small" style="color:var(--danger)">交付自检失败：${esc(e.message)}</div>`; }
  };
  $("#page").innerHTML = `
    ${pageHead("04", "报告工坊", "勾选实验与数据集，按毕业论文的章节结构生成 Word 初稿：目录、中英文摘要、绪论、相关技术、数据与预处理、实验与结果分析（三线表 + 自动插图）、总结、参考文献（GB/T 7714）、致谢。")}
    <div id="rp-readiness" class="mt14"></div>
    <div class="form-grid mt14">
      <div class="panel">
        <b class="t">论文信息</b>
        <div class="form-row"><label>论文题目</label><input id="rp-title" value="基于机器学习的毕业设计研究"></div>
        <div class="form-grid">
          <div class="form-row"><label>学校（用于封面与页眉）</label><input id="rp-school" placeholder="如：XX 大学"></div>
          <div class="form-row"><label>学院</label><input id="rp-college" placeholder="如：计算机与人工智能学院"></div>
          <div class="form-row"><label>专业</label><input id="rp-major" placeholder="如：人工智能"></div>
          <div class="form-row"><label>姓名</label><input id="rp-name"></div>
          <div class="form-row"><label>学号</label><input id="rp-sid"></div>
          <div class="form-row"><label>指导教师</label><input id="rp-advisor"></div>
        </div>
        <div class="form-row"><label>关联数据集（写入第三章）</label>
          <select id="rp-ds"><option value="">（不使用）</option>${state.datasets.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("")}</select></div>
      </div>
      <div class="panel">
        <b class="t">写入第四章的实验（可多选）</b>
        <div class="mt8" style="max-height:180px;overflow:auto">${doneRuns.length ? doneRuns.map((r) => `
          <label class="row-flex small report-run" style="padding:6px 2px"><input type="checkbox" class="rp-run" value="${esc(r.run_id)}" style="width:auto" ${state.reportSelected.includes(r.run_id) ? "checked" : ""}>
          <span>${esc(r.name || r.model_label || r.model || r.run_id)}</span>${groupBadge(r.group)}
          <span class="muted">${r.primary_metric ? `${esc(r.primary_metric.name)}=${fmtNum(r.primary_metric.value)}` : r.run_id}</span></label>`).join("") : '<div class="empty">暂无已完成实验，先去训练一个模型。</div>'}</div>
        <label class="row-flex mt14 small"><input type="checkbox" id="rp-ai" style="width:auto" ${state.config && state.config.configured ? "checked" : "disabled"}>
          用 AI 起草各章节正文（自动去 AI 味）${state.config && !state.config.configured ? "——需先配置 AI 接口" : ""}</label>
        <button class="btn accent mt14" id="btn-report">生成论文初稿（.docx）</button>
        <div id="rp-result" class="mt14"></div>
      </div>
    </div>
    <div class="section"><h2>论文格式（自动应用，导出后请对照学校模板微调）</h2>
      <div class="panel small" style="line-height:2.0">
        A4，页边距上/下 2.6cm、左 3.0cm、右 2.5cm ｜ 正文宋体小四（西文 Times New Roman），1.5 倍行距，首行缩进 2 字符<br>
        一级标题黑体三号居中，二级黑体四号，三级黑体小四 ｜ 页眉「学校 + 本科毕业设计（论文）」，页脚居中页码，封面不显示<br>
        表格为三线表，图表注释宋体五号居中（图注在下、表注在上）｜ 目录为 Word 自动域：打开文档后 <b>Ctrl+A 全选 → F9 更新</b><br>
        <span class="muted">学校的具体要求（字号/行距/编号样式/封面样式）各校不同，生成后以学院下发的模板为准逐项核对。</span>
      </div>
    </div>
    <div class="section"><h2>AIGC 自检与降 AI 味</h2>
      <div class="panel">
        <div class="hint mb14" style="line-height:1.9">
          把（自己写的或 AI 生成的）段落粘进来，先「检测」看 AI 特征在哪些地方，再「降 AI 味改写」。
          规则引擎会：删模板套话、打散均匀句长、把列表改成连贯段落、精简空洞修饰——这些正是检测系统（句长波动、模板短语等信号）重点盯的特征。
          配置 AI 接口后可再叠一层 LLM 深度改写（保证数字与结论不变）。
        </div>
        <textarea id="hu-text" rows="8" placeholder="粘贴需要检查的论文段落……"></textarea>
        <div class="row-flex mt8">
          <button class="btn" id="hu-check">检测 AI 特征</button>
          <button class="btn accent" id="hu-rewrite">降 AI 味改写</button>
          <label class="row-flex small" style="gap:4px"><input type="checkbox" id="hu-llm" style="width:auto" ${state.config && state.config.configured ? "" : "disabled"}> 叠加 LLM 深度改写${state.config && !state.config.configured ? "（需配置 AI）" : ""}</label>
        </div>
        <div id="hu-out" class="mt14"></div>
        <div class="hint mt14" style="border-top:1px solid var(--line);padding-top:10px;line-height:1.9">
          说明：各校 AIGC 检测系统（知网/维普/万方等）算法不公开且持续更新，<b>任何工具都不能保证检测结果</b>。
          本功能做的是让文字更自然、更有个人风格；最可靠的做法始终是：把 AI 初稿当参考资料，<b>用自己的话重写</b>，并遵守学校关于 AIGC 使用的规定。
        </div>
      </div>
    </div>
    <div class="section"><h2>已导出的报告</h2>
      <div class="panel">${state.reports.length ? `<table class="data"><tbody>
        ${state.reports.map((r) => `<tr><td><a href="/api/report/download?filename=${encodeURIComponent(r.filename)}">${esc(r.filename)}</a></td>
        <td class="muted small">${(r.size / 1024).toFixed(0)} KB</td><td class="muted small">${esc(r.created_at)}</td></tr>`).join("")}
      </tbody></table>` : '<div class="empty">还没有导出过报告。</div>'}</div>
    </div>`;
  document.querySelectorAll(".rp-run").forEach((cb) => cb.onchange = () => {
    state.reportSelected = [...document.querySelectorAll(".rp-run:checked")].map((c) => c.value);
    saveReportSel();
    readiness();
  });
  ["#rp-title", "#rp-school", "#rp-college", "#rp-major", "#rp-name", "#rp-sid", "#rp-advisor", "#rp-ds"].forEach((sel) => {
    $(sel).onchange = readiness;
  });
  $("#rp-title").oninput = readiness;
  await readiness();
  $("#btn-report").onclick = async () => {
    const ids = [...document.querySelectorAll(".rp-run:checked")].map((c) => c.value);
    if (!ids.length) return toast("请至少勾选一个已完成实验写入第四章", "error");
    const btn = $("#btn-report");
    btn.disabled = true; btn.textContent = "生成中（AI 起草约需 1-2 分钟）…";
    try {
      const r = await api("/api/report/generate", { method: "POST", body: {
        title: $("#rp-title").value.trim() || "基于机器学习的毕业设计研究",
        run_ids: ids, dataset_id: $("#rp-ds").value || null,
        author: { school: $("#rp-school").value, college: $("#rp-college").value, major: $("#rp-major").value, name: $("#rp-name").value, student_id: $("#rp-sid").value, advisor: $("#rp-advisor").value },
        ai_draft: $("#rp-ai").checked,
      }});
      $("#rp-result").innerHTML = `<div class="panel" style="background:var(--accent-soft);border-color:var(--accent)">
        已生成${r.ai_sections.length ? `（AI 起草 ${r.ai_sections.length} 个章节，已自动去 AI 味）` : ""}
        <div class="mt8"><a class="btn small" href="/api/report/download?filename=${encodeURIComponent(r.filename)}">下载 ${esc(r.filename)}</a></div>
        <div class="hint mt8">打开后记得：Ctrl+A 全选 → F9 更新目录域；摘要与结论建议自己重写。</div></div>`;
      toast("报告生成成功", "success");
      state.reports = (await api("/api/report/list")).reports;
    } catch (e) { toast(e.message, "error"); }
    btn.disabled = false; btn.textContent = "生成论文初稿（.docx）";
  };
  $("#hu-check").onclick = async () => {
    const text = $("#hu-text").value.trim();
    if (!text) return toast("请先粘贴文本", "error");
    const r = await api("/api/humanize/check", { method: "POST", body: { text } });
    renderHuResult(r, null);
  };
  $("#hu-rewrite").onclick = async () => {
    const text = $("#hu-text").value.trim();
    if (!text) return toast("请先粘贴文本", "error");
    const btn = $("#hu-rewrite");
    btn.disabled = true; btn.textContent = "改写中…";
    try {
      const r = await api("/api/humanize/rewrite", { method: "POST", body: { text, use_llm: $("#hu-llm").checked } });
      if (r.llm_note && r.llm_note !== "ok") toast(r.llm_note, "error");
      renderHuResult(r.score_after, r);
      $("#hu-text").value = r.text;
      toast("已改写并回填到输入框", "success");
    } catch (e) { toast(e.message, "error"); }
    btn.disabled = false; btn.textContent = "降 AI 味改写";
  };

  function renderHuResult(score, rewrite) {
    const levelColor = { "低": "var(--ok)", "中": "var(--warn)", "高": "var(--danger)" }[score.level];
    $("#hu-out").innerHTML = `
      <div class="panel" style="background:#fbfbf9">
        <div class="kv-row" style="margin-top:0">
          <div class="kv"><b style="color:${levelColor}">${score.score}</b>AI 特征分（${score.level}）</div>
          ${rewrite ? `<div class="kv"><b>${rewrite.changes.length}</b>处修改</div>` : ""}
        </div>
        ${score.issues && score.issues.length ? `<div class="mt8 small">${score.issues.map((i) => `<div>· ${esc(i)}</div>`).join("")}</div>` : '<div class="small" style="color:var(--ok)">未检出明显 AI 特征。</div>'}
        ${rewrite ? `<div class="mt8 small muted">修改明细：${esc(rewrite.changes.slice(0, 8).join("；") || "无")}<br>改写后的文本已回填到输入框，请通读确认语义未变。</div>` : ""}
      </div>`;
  }
}

/* ================= AI 设置 ================= */
async function pageAI() {
  const cfg = await api("/api/config");
  state.config = cfg;
  $("#page").innerHTML = `
    ${pageHead("05", "AI 设置", "填写任意 OpenAI 兼容接口（DeepSeek、智谱、通义、Kimi、OpenAI 等），用于结果分析与论文章节起草。Key 只保存在本机 data/runtime_config.json，也可改用环境变量 " + esc(cfg.llm_api_key_env) + "。")}
    <div class="form-grid">
      <div class="panel">
        <div class="form-row"><label>接口地址 base_url（一般以 /v1 结尾）</label>
          <input id="ai-url" placeholder="https://api.deepseek.com/v1" value="${esc(cfg.llm_base_url)}"></div>
        <div class="form-row"><label>模型名称</label>
          <input id="ai-model" placeholder="如 deepseek-chat / glm-4-flash / qwen-plus" value="${esc(cfg.llm_model)}"></div>
        <div class="form-row"><label>API Key ${cfg.env_key_present ? "（检测到环境变量已配置，优先使用）" : "（留空则沿用已保存或环境变量）"}</label>
          <input id="ai-key" type="password" placeholder="${cfg.stored_key_present ? "已保存（输入可覆盖）" : "sk-..."}"></div>
        <div class="row-flex mt8">
          <button class="btn primary" id="ai-save">保存配置</button>
          <button class="btn" id="ai-test">测试连接</button>
          <span class="hint">当前状态：${cfg.configured ? "已配置" : "未配置（使用内置规则分析）"}</span>
        </div>
        <div id="ai-test-out" class="mt14"></div>
      </div>
      <div class="panel">
        <b class="t">常用接口参考</b>
        <table class="data"><tbody>
          <tr><td>DeepSeek</td><td class="small muted">https://api.deepseek.com/v1</td><td class="small">deepseek-chat</td></tr>
          <tr><td>智谱 AI</td><td class="small muted">https://open.bigmodel.cn/api/paas/v4</td><td class="small">glm-4-flash（免费）</td></tr>
          <tr><td>阿里通义</td><td class="small muted">https://dashscope.aliyuncs.com/compatible-mode/v1</td><td class="small">qwen-plus</td></tr>
          <tr><td>Moonshot</td><td class="small muted">https://api.moonshot.cn/v1</td><td class="small">moonshot-v1-8k</td></tr>
          <tr><td>OpenAI</td><td class="small muted">https://api.openai.com/v1</td><td class="small">gpt-4o-mini</td></tr>
        </tbody></table>
        <div class="hint mt14">学生推荐智谱 glm-4-flash（有免费额度）或 DeepSeek（低价）。Key 到各官网注册获取。</div>
      </div>
    </div>`;
  $("#ai-save").onclick = async () => {
    try {
      const body = { llm_base_url: $("#ai-url").value.trim(), llm_model: $("#ai-model").value.trim() };
      const key = $("#ai-key").value.trim();
      if (key) body.llm_api_key = key;
      const r = await api("/api/config", { method: "POST", body });
      toast(r.configured ? "已保存，AI 功能已激活" : "已保存（信息不完整，AI 功能未激活）", r.configured ? "success" : "");
      pageAI();
    } catch (e) { toast(e.message, "error"); }
  };
  $("#ai-test").onclick = async () => {
    const out = $("#ai-test-out");
    out.innerHTML = '<div class="hint">连接测试中…</div>';
    try {
      const r = await api("/api/config/test", { method: "POST" });
      out.innerHTML = `<div class="small" style="color:var(--ok)">连接成功，模型回复：${esc(r.reply)}</div>`;
    } catch (e) { out.innerHTML = `<div class="small" style="color:var(--danger)">${esc(e.message)}</div>`; }
  };
}

/* ================= 毕设指南 ================= */
const ROADMAP = [
  ["选题与开题", "确定研究方向：数据好获取、计算量可控（CPU/单卡可跑）、有一定改进空间", "选题大小适中，避免纯调包也避免做不出来；写清「用什么数据 + 什么模型 + 改进什么」", "开题报告、任务书", "浏览本页其余步骤验证可行性；到「数据集」页确认数据可得"],
  ["文献调研", "检索 10-20 篇相关论文（知网 / Google Scholar / Papers With Code），确定 1-2 个基线方法", "优先近 3-5 年文献；记录每篇的数据集、指标、核心思路", "文献综述初稿、基线清单", "把基线模型记下来，到「模型训练」页复现"],
  ["数据获取与 EDA", "从 Kaggle / HuggingFace / UCI / 官方源下载数据；检查规模、类别平衡、缺失值、分布", "数据质量决定上限；防止数据泄漏（先划分再预处理）；留出测试集不动", "数据集卡片、EDA 图表", "「数据集」页一键载入内置集 / 导入 / URL 下载，自动生成 EDA + AI 解读"],
  ["基线模型", "先跑通最简单的基线（逻辑回归 / CNN 等），拿到第一组可信指标", "固定随机种子；统一评价指标；结果要能复现", "基线指标 + 日志", "「模型训练」页选择基线模型，默认参数先跑一轮"],
  ["算法改进", "在基线上做 1-2 个改进点：网络结构、损失函数、数据增强、集成、调参搜索", "改进点要能消融验证（逐个开关）；每次只改一个变量", "改进模型 + 消融实验表", "「模型训练」页改参数重跑，多轮实验在「实验记录」对比"],
  ["实验管理", "统一实验设置，完成对比表（基线 vs 改进 vs SOTA）与曲线图", "每组实验记录超参数与种子；指标用均值±方差更可信", "实验对比表格、训练曲线", "「实验记录」页自动留档所有实验，可直接选入报告"],
  ["结果分析", "混淆矩阵、ROC、特征重要性、误差案例分析；解释为什么有效/无效", "分析要有因果解释，不是罗列数字；图要有编号并在文中引用", "实验分析章节", "点「AI 分析结果」生成解读，所有图表可直接插入论文"],
  ["论文撰写", "按学校模板撰写：摘要最后写、图表规范编号、参考文献核对", "先搭骨架再填肉；实验章节用数据说话；查重前自己先通读", "论文初稿（.docx）", "「报告工坊」一键生成含图表的 Word 初稿，AI 起草各章"],
  ["查重与答辩", "查重降重、盲审修改、准备答辩 PPT 与现场演示", "PPT 少字多图；准备好「为什么这个改进有效」的答案；演练 3 遍", "终稿、答辩 PPT", "报告导出后本地精修；用演示实验录屏做 backup"],
];

async function pageGuide() {
  $("#page").innerHTML = `
    ${pageHead("06", "毕设指南", "人工智能专业毕业设计的完整拆解。核心结论：毕设是一条流水线——数据决定上限、实验决定说服力、分析决定深度、写作决定呈现。")}
    <div class="panel">
      <div class="hint mb14">整体串联：选题 → 文献定基线 → 数据 → 基线可复现 → 改进可消融 → 实验可对比 → 分析有解释 → 论文有图表 → 答辩有底气</div>
      ${ROADMAP.map(([t, doWhat, focus, output, tool], i) => `
        <div class="guide-step" data-n="${i + 1}"><h3>${esc(t)}</h3>
          <div class="row"><span class="k">做什么</span><span class="v">${esc(doWhat)}</span></div>
          <div class="row"><span class="k">注重</span><span class="v">${esc(focus)}</span></div>
          <div class="row"><span class="k">产出</span><span class="v">${esc(output)}</span></div>
          <div class="row"><span class="k">工具衔接</span><span class="v">${esc(tool)}</span></div>
        </div>`).join("")}
    </div>
    <div class="section"><h2>一套可以直接照抄的实验方案模板</h2>
      <div class="panel table-scroll">
        <table class="data"><thead><tr><th>编号</th><th>实验名称（写入分组）</th><th>对比要回答的问题</th><th>做法</th></tr></thead>
        <tbody>
          <tr><td>A1</td><td>基线-逻辑回归（基线）</td><td>最简单、可复现的下限是多少</td><td>默认参数先跑通，固定随机种子</td></tr>
          <tr><td>A2</td><td>改进-随机森林（改进）</td><td>更强的非线性模型能否超过基线</td><td>固定种子，调树数量/最大深度，只改关键参数</td></tr>
          <tr><td>A3</td><td>改进-梯度提升树（改进）</td><td>集成与逐棵纠错是否再提升</td><td>与 A2 同一数据划分与评价口径</td></tr>
          <tr><td>A4</td><td>消融-关闭某一改进项（消融）</td><td>改进点是否真的有用</td><td>在最优配置上单独关掉一个改进项，其余不动</td></tr>
          <tr><td>A5</td><td>自定义-重复实验或外部方案（自定义）</td><td>稳定性与扩展对比</td><td>同一配置重复 3-5 次记录均值±标准差</td></tr>
        </tbody></table>
        <div class="hint mt14">建议至少完成 A1、A2、A3、A4。训练前在训练页填好实验名称与分组，报告会自动按组排序，对比页可直接生成论文用的三线表。</div>
      </div>
    </div>`;
}

/* ================= 侧栏状态 ================= */
function updateSidebar() {
  const h = state.health;
  if (!h) return;
  $("#sidebar-status").innerHTML =
    `v${esc(h.version)}<br>GPU <b>${h.cuda ? "可用" : h.cuda === false ? "CPU 模式" : "未知"}</b><br>AI <b>${state.config && state.config.configured ? "已配置" : "未配置"}</b>`;
}

/* ================= 启动 ================= */
(async function init() {
  try {
    const [h, c] = await Promise.all([api("/api/health"), api("/api/config")]);
    state.health = h; state.config = c;
    updateSidebar();
  } catch (e) {
    $("#sidebar-status").textContent = "后端连接失败";
  }
  route();
  // 有了 10 步毕设向导后，不再自动弹旧版新手引导弹窗；侧栏「新手引导」仍可手动重看。
  $("#btn-tour").onclick = () => showTour(0);
  $("#btn-browser").onclick = async () => {
    const bridge = tfBridge();
    if (bridge && bridge.open_external_browser) {
      try { await bridge.open_external_browser(); toast("已用系统浏览器打开", "success"); return; } catch { /* fallthrough */ }
    }
    window.open(location.href, "_blank");
  };
})();
