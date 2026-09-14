import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const source = fs.readFileSync(path.join(root, "web", "wizard.js"), "utf8");
const sandbox = { window: {}, console };
vm.createContext(sandbox);
vm.runInContext(source, sandbox);

test("wizard maps each supported task family to a precise research direction", () => {
  assert.equal(vm.runInContext("wizardInferDirection('image_classification', 'image')", sandbox), "图像分类");
  assert.equal(vm.runInContext("wizardInferDirection('object_detection', 'image')", sandbox), "目标检测");
  assert.equal(vm.runInContext("wizardInferDirection('semantic_segmentation', 'image')", sandbox), "语义分割");
  assert.equal(vm.runInContext("wizardInferDirection('text_classification', 'tabular')", sandbox), "文本分类");
  assert.equal(vm.runInContext("wizardInferDirection('time_series_forecasting', 'tabular')", sandbox), "时间序列预测");
  assert.equal(vm.runInContext("wizardInferDirection('tabular_classification', 'tabular')", sandbox), "表格预测");
});

test("wizard keeps legacy directions compatible with precise task families", () => {
  assert.equal(vm.runInContext("wizardDirectionMatches('目标检测', 'object_detection', 'image')", sandbox), true);
  assert.equal(vm.runInContext("wizardDirectionMatches('语义分割', 'semantic_segmentation', 'image')", sandbox), true);
  assert.equal(vm.runInContext("wizardDirectionMatches('算法优化与消融', 'object_detection', 'image')", sandbox), true);
});

test("wizard explains where to place local YOLO weights", () => {
  const catalog = {
    object_detection: {
      label: "图像 · 目标检测",
      models: {
        yolo26n: {
          label: "YOLO26n 目标检测",
          desc: "测试模型",
          engine: "ultralytics",
          params: {},
        },
      },
    },
  };
  vm.runInContext(
    `esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
    wizardState = { steps: {} }; catalog = ${JSON.stringify(catalog)};`,
    sandbox,
  );
  const html = vm.runInContext(
    "wizardNetworkPane({ datasets: [{ type: 'image' }] }, { catalog })",
    sandbox,
  );
  assert.match(html, /models\/ 或 data\/models\//);
  assert.match(html, /THESISFORGE_MODEL_DIR/);
});
