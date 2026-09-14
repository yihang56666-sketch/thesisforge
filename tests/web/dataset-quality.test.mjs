import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const appSource = fs.readFileSync(path.join(root, "web", "app.js"), "utf8");
const wizardSource = fs.readFileSync(path.join(root, "web", "wizard.js"), "utf8");

const sandbox = {
  window: { addEventListener: () => {} },
  location: { hash: "" },
  console,
  localStorage: { getItem: () => null, setItem: () => {} },
  document: {
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: () => {},
  },
  fetch: async () => ({ ok: true, json: async () => ({}) }),
};
vm.createContext(sandbox);
vm.runInContext(appSource.split("/* ================= 路由 ================= */")[0], sandbox);
vm.runInContext(wizardSource, sandbox);

test("dataset detail renders backend quality warnings as a visible warning block", () => {
  const html = vm.runInContext(
    "datasetQualityHtml({ quality_warnings: ['检测到 3 个缺失值', '检测到 2 个重复样本'] })",
    sandbox,
  );
  assert.match(html, /quality-warning/);
  assert.match(html, /数据质量提醒/);
  assert.match(html, /检测到 3 个缺失值/);
  assert.match(html, /检测到 2 个重复样本/);
});

test("wizard dataset pane shows loaded dataset quality warnings", () => {
  const html = vm.runInContext(
    "wizardState = { project: { direction: '其他' } }; wizardDatasetPane({ builtin: [], datasets: [{ id: 'd1', name: 'demo', type: 'tabular', quality_warnings: ['样本规模较小'] }] })",
    sandbox,
  );
  assert.match(html, /数据质量提醒/);
  assert.match(html, /样本规模较小/);
});
