# Sprint 1（向导框架与状态后端）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 上线可持久化的 10 步毕设向导：开始页、步骤导航、跳过/完成状态、全工作台切换，并把步骤 1/2/6 接上真实数据。

**Architecture:** 后端用 `data/project.json` 保存向导进度，提供 `/api/wizard` 读写接口；前端新增 `web/wizard.js`，在现有 `#/wizard` 路由下渲染向导，其余侧栏页面保持不变。

**Tech Stack:** FastAPI、原生 JS、现有 REST API、pytest。

---

## 文件结构

- Create: `app/onboarding.py` — 步骤定义、进度读写、完成检测、项目信息默认值。
- Modify: `app/config.py` — `PROJECT_FILE` 常量。
- Modify: `app/main.py` — `/api/wizard` GET/PUT。
- Create: `tests/test_onboarding.py` — 后端测试。
- Create: `web/wizard.js` — 向导渲染与交互。
- Modify: `web/index.html` — 侧栏“毕设向导”入口。
- Modify: `web/app.js` — 路由注册与状态获取。
- Modify: `web/style.css` — 向导样式。
- Create: `docs/superpowers/plans/2026-09-13-sprint2-train-engine.md` — 下一阶段占位（在最后提交时创建）。

## 数据约定

`data/project.json`：

```json
{
  "version": 1,
  "project": {
    "title": "",
    "direction": "",
    "author": "",
    "advisor": "",
    "goal": ""
  },
  "steps": {
    "1": {"state": "todo", "completed": false, "saved_at": null},
    "2": {"state": "todo", "completed": false, "saved_at": null}
  },
  "active_step": 1,
  "updated_at": "2026-09-13 12:00:00"
}
```

`state` 取值：`todo`（未开始）、`doing`（当前/编辑过）、`done`、`skipped`。

步骤 1-10 的 key 固定为字符串 `"1"` 到 `"10"`。

## Task 1: 后端向导状态

**Files:**
- Create: `app/onboarding.py`
- Modify: `app/config.py:44` 附近
- Test: `tests/test_onboarding.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_onboarding.py`：

```python
import os
import json
from app import onboarding


def _tmp_project(tmp_path, monkeypatch):
    p = tmp_path / "project.json"
    monkeypatch.setattr(onboarding, "PROJECT_FILE", p)
    return p


def test_default_progress(tmp_path, monkeypatch):
    _tmp_project(tmp_path, monkeypatch)
    data = onboarding.load_progress()
    assert data["version"] == 1
    assert data["active_step"] == 1
    assert data["steps"]["1"]["state"] == "todo"
    assert set(data["steps"]) == {str(i) for i in range(1, 11)}


def test_save_and_reload(tmp_path, monkeypatch):
    p = _tmp_project(tmp_path, monkeypatch)
    data = onboarding.load_progress()
    data["project"]["title"] = "神经网络优化研究"
    data["steps"]["4"]["state"] = "done"
    data["steps"]["4"]["completed"] = True
    data["active_step"] = 5
    onboarding.save_progress(data)
    reloaded = onboarding.load_progress()
    assert reloaded["project"]["title"] == "神经网络优化研究"
    assert reloaded["steps"]["4"]["state"] == "done"
    assert reloaded["active_step"] == 5


def test_validate_known_steps(tmp_path, monkeypatch):
    _tmp_project(tmp_path, monkeypatch)
    ok = onboarding.update_step(2, "done")
    assert ok is True
    bad = onboarding.update_step("99", "done")
    assert bad is False
    bad_state = onboarding.update_step(2, "weird")
    assert bad_state is False


def test_project_completion_detection(tmp_path, monkeypatch):
    p = _tmp_project(tmp_path, monkeypatch)
    p.write_text(json.dumps({
        "version": 1,
        "project": {"title": "t", "direction": "image"},
        "steps": {str(i): {"state": "todo", "completed": False, "saved_at": None} for i in range(1, 11)},
        "active_step": 1,
    }), encoding="utf-8")
    assert onboarding.looks_completed_manually(onboarding.load_progress()) is False
    data = onboarding.load_progress()
    for i in range(1, 11):
        data["steps"][str(i)]["state"] = "done"
        data["steps"][str(i)]["completed"] = True
    onboarding.save_progress(data)
    assert onboarding.looks_completed_manually(onboarding.load_progress()) is True
```

- [ ] **Step 2: 运行测试确认失败**

运行：`python -m pytest tests/test_onboarding.py -q`

预期：`ModuleNotFoundError: No module named 'app.onboarding'`。

- [ ] **Step 3: 实现 onboarding 模块**

创建 `app/onboarding.py`：

```python
"""毕设向导进度：步骤定义、持久化与完成状态。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import DATA_DIR

PROJECT_FILE = DATA_DIR / "project.json"
STEP_NAMES = {
    "1": "立项",
    "2": "数据",
    "3": "数据预处理与划分",
    "4": "任务与网络架构",
    "5": "优化与训练策略",
    "6": "训练与监控",
    "7": "评估分析",
    "8": "消融实验",
    "9": "实验对比",
    "10": "报告工坊",
}
VALID_STATES = {"todo", "doing", "done", "skipped"}


def _blank_step() -> dict:
    return {"state": "todo", "completed": False, "saved_at": None}


def default_progress() -> dict:
    return {
        "version": 1,
        "project": {"title": "", "direction": "", "author": "", "advisor": "", "goal": ""},
        "steps": {key: _blank_step() for key in STEP_NAMES},
        "active_step": 1,
        "updated_at": "",
    }


def load_progress() -> dict:
    try:
        raw = json.loads(PROJECT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    data = default_progress()
    if isinstance(raw, dict) and raw.get("version") == 1:
        for key in STEP_NAMES:
            step = raw.get("steps", {}).get(key, {})
            if isinstance(step, dict) and step.get("state") in VALID_STATES:
                data["steps"][key] = {
                    "state": step.get("state", "todo"),
                    "completed": bool(step.get("completed", False)),
                    "saved_at": step.get("saved_at"),
                }
        if isinstance(raw.get("project"), dict):
            data["project"].update({k: str(v) for k, v in raw["project"].items()
                                    if k in data["project"]})
        if isinstance(raw.get("active_step"), int) and 1 <= raw["active_step"] <= 10:
            data["active_step"] = raw["active_step"]
    return data


def save_progress(data: dict) -> dict:
    clean = default_progress()
    clean["project"].update({k: str(v) for k, v in data.get("project", {}).items()
                             if k in clean["project"]})
    for key, step in data.get("steps", {}).items():
        if key in STEP_NAMES and isinstance(step, dict) and step.get("state") in VALID_STATES:
            clean["steps"][key] = {"state": step["state"],
                                   "completed": bool(step.get("completed", False)),
                                   "saved_at": step.get("saved_at")}
    if isinstance(data.get("active_step"), int) and 1 <= data["active_step"] <= 10:
        clean["active_step"] = data["active_step"]
    clean["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    PROJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean


def update_step(step_key: str | int, state: str) -> bool:
    data = load_progress()
    key = str(step_key)
    if key not in STEP_NAMES or state not in VALID_STATES:
        return False
    data["steps"][key] = {"state": state, "completed": state == "done",
                          "saved_at": time.strftime("%Y-%m-%d %H:%M:%S") if state in ("done", "skipped") else None}
    save_progress(data)
    return True


def looks_completed_manually(data: dict) -> bool:
    return all(step.get("completed", False) for step in data.get("steps", {}).values())
```

修改 `app/config.py`，在 `CONFIG_FILE` 后加：

```python
PROJECT_FILE = DATA_DIR / "project.json"
```

- [ ] **Step 4: 运行测试确认通过**

运行：`python -m pytest tests/test_onboarding.py -q`

预期：4 passed。注意 `app/config.py` 的 `DATA_DIR` 在测试环境中由 conftest 指向 tmp；若测试隔离需要，在 onboarding 模块顶部按现有 config 测试模式处理。

- [ ] **Step 5: 提交**

```bash
git add app/onboarding.py app/config.py tests/test_onboarding.py
git commit -m "feat: 向导进度状态后端与持久化"
```

## Task 2: 向导 API

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_onboarding.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_onboarding.py`：

```python
from fastapi.testclient import TestClient
from app import main


def test_wizard_api_roundtrip(monkeypatch, tmp_path):
    import app.onboarding as onboarding
    monkeypatch.setattr(onboarding, "PROJECT_FILE", tmp_path / "project.json")
    client = TestClient(main.app)
    r = client.get("/api/wizard")
    assert r.status_code == 200
    assert r.json()["active_step"] == 1
    body = {"project": {"title": "测试题目"}, "steps": {"1": {"state": "done", "completed": True, "saved_at": None}}, "active_step": 2}
    r2 = client.put("/api/wizard", json=body)
    assert r2.status_code == 200
    assert r2.json()["steps"]["1"]["state"] == "done"
```

- [ ] **Step 2: 运行测试确认失败**

运行：`python -m pytest tests/test_onboarding.py::test_wizard_api_roundtrip -q`

预期：404 或断言失败。

- [ ] **Step 3: 实现 API**

在 `app/main.py` 顶部 import：

```python
from . import onboarding
```

在 `@app.get("/api/health")` 后加：

```python
@app.get("/api/wizard")
def get_wizard():
    return onboarding.load_progress()


@app.put("/api/wizard")
def put_wizard(payload: dict):
    return onboarding.save_progress(payload)
```

- [ ] **Step 4: 运行测试确认通过**

运行：`python -m pytest tests/test_onboarding.py -q`

预期：5 passed。

- [ ] **Step 5: 提交**

```bash
git add app/main.py tests/test_onboarding.py
git commit -m "feat: 新增向导进度读取与保存 API"
```

## Task 3: 前端向导页

**Files:**
- Create: `web/wizard.js`
- Modify: `web/app.js:135` 附近
- Modify: `web/index.html`
- Modify: `web/style.css`

- [ ] **Step 1: 在 index.html 侧栏增加入口**

在 `#/guide` 导航项前加：

```html
<a href="#/wizard" data-route="wizard" id="nav-wizard"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3.5h12M2 8h12M2 12.5h7"/><circle cx="12.5" cy="12.5" r="1.7"/></svg><span>毕设向导</span></a>
```

在 `<script src="/app.js"></script>` 前加：

```html
<script src="/wizard.js"></script>
```

- [ ] **Step 2: 在 app.js 注册路由**

在 `route()` 的 hash 映射处加：

```js
if (name === "wizard") {
  renderWizard();
  return;
}
```

`renderWizard` 定义在 `wizard.js` 中；`app.js` 的 `route()` 需要在 `pageHead` 之前判断，避免进入普通页面渲染。

- [ ] **Step 3: 写向导 UI 核心逻辑**

创建 `web/wizard.js`。核心结构（完整代码在本步实现，测试用浏览器验证）：

```js
const WIZARD_STEPS = [
  { key: "1", name: "立项", tip: "先告诉软件你的毕设方向，后面每一步都会用。", fields: ["title", "direction", "author", "advisor", "goal"] },
  { key: "2", name: "数据", tip: "没有自己的数据就选内置三件套，有数据也可以直接上传。" },
  { key: "3", name: "数据预处理与划分", tip: "这些设置会应用到后面的每次训练。" },
  { key: "4", name: "任务与网络架构", tip: "选一个能完成你任务的神经网络。" },
  { key: "5", name: "优化与训练策略", tip: "不会选就保持推荐值。" },
  { key: "6", name: "训练与监控", tip: "给实验起个名字，然后开始训练。" },
  { key: "7", name: "评估分析", tip: "看看模型在测试集上到底怎么样。" },
  { key: "8", name: "消融实验", tip: "逐组件验证你的改进到底有没有用。" },
  { key: "9", name: "实验对比", tip: "把所有实验放到同一张表里对比。" },
  { key: "10", name: "报告工坊", tip: "把实验结果变成论文初稿。" }
];

let wizardState = null;

async function loadWizard() {
  const res = await api("/api/wizard");
  wizardState = res;
}

async function saveWizard(patch) {
  if (!wizardState) return;
  Object.assign(wizardState, patch);
  wizardState = await api("/api/wizard", { method: "PUT", body: JSON.stringify(wizardState) });
}

async function renderWizard() {
  await loadWizard();
  const page = document.getElementById("page");
  page.className = "page wizard-page";
  const stepNo = Math.max(1, Math.min(10, wizardState.active_step || 1));
  const step = WIZARD_STEPS[stepNo - 1];
  page.innerHTML = `
    ${wizardStepsBar(wizardState, stepNo)}
    <div class="wizard-body">
      <div class="wizard-main">
        <div class="wizard-kicker">第 ${stepNo} 步 / 共 10 步 · ${step.name}</div>
        <h1>${esc(step.name)}</h1>
        <p class="wizard-tip">${esc(step.tip)}</p>
        <div class="panel">${wizardStepBody(step, wizardState)}</div>
      </div>
      <div class="wizard-rail">
        <div class="panel">
          <div class="muted small">当前项目</div>
          <div class="mt8"><b>${esc(wizardState.project.title || "未填写题目")}</b></div>
          <div class="muted small mt4">${esc(wizardState.project.direction || "未选择方向")}</div>
          <hr class="mt16 mb16">
          <button class="btn" onclick="location.hash='#/guide'">查看毕设指南</button>
          <button class="btn ghost" onclick="location.hash='#/dashboard'">进入完整工作台</button>
        </div>
      </div>
    </div>
  `;
  bindWizardStep(step, stepNo);
}
```

还需实现 `wizardStepsBar`、`wizardStepBody`、`bindWizardStep`。步骤 1 渲染 project 输入并保存；步骤 2 调用 `/api/datasets` 展示内置/已导入数据集；步骤 6 调用 `/api/runs` 显示实验状态并可跳转 `#/train`；其余步骤先显示完成/跳过操作按钮，状态写入 `/api/wizard`。

- [ ] **Step 4: 添加向导样式**

在 `web/style.css` 末尾追加：

```css
.wizard-page { padding: 24px 32px; }
.wizard-steps { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }
.wizard-step-chip { padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; color: var(--muted); }
.wizard-step-chip.active { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.wizard-step-chip.done { color: #0d6e63; border-color: #0d6e63; }
.wizard-body { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 24px; }
.wizard-main { max-width: 880px; }
.wizard-kicker { color: var(--accent); font-size: 13px; font-weight: 600; }
.wizard-tip { color: var(--muted); margin: 8px 0 20px; }
.wizard-actions { display: flex; gap: 12px; justify-content: space-between; margin-top: 20px; }
@media (max-width: 900px) { .wizard-body { grid-template-columns: 1fr; } }
```

`style.css` 中的 CSS 变量若与现有主题不一致，以现有主题色板为准。

- [ ] **Step 5: 启动开发服务器做浏览器验证**

运行：`python -m app.main --browser` 或现有开发命令；打开 `http://127.0.0.1:8765/#/wizard`，验证：开始页显示、步骤条 10 项、步骤 1 输入保存后跳步骤 2 不丢、切回 `#/dashboard` 再回向导进度仍在、步骤 2 能看到内置数据集列表。

- [ ] **Step 6: 提交**

```bash
git add web/wizard.js web/app.js web/index.html web/style.css
git commit -m "feat: 毕设向导前端骨架与真实步骤 1/2/6 联动"
```

## Task 4: 全量回归

**Files:**
- Modify: 无（只验证）

- [ ] **Step 1: 跑全量测试**

运行：`python -m pytest tests -q`

预期：原 77 个测试 + 新增向导测试全部通过。

- [ ] **Step 2: 编译检查**

运行：`python -m compileall -q app packaging`

预期：退出码 0。

- [ ] **Step 3: 提交收尾**

```bash
git add docs/
git commit -m "docs: 引导式实验平台设计与 Sprint 1 计划"
```

## Sprint 1 完成后的下一个 Sprint

Sprint 2 计划文件在 Sprint 1 完成后创建，内容为：

- `app/networks.py` 网络注册表
- `app/trainer.py` 通用 Trainer
- `app/catalog.py` 参数扩展
- `tests/test_networks.py`、`tests/test_trainer.py`

