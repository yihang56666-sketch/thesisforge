# ThesisForge v0.3.0 设计规范：桌面窗口 + 完整毕设实验闭环

日期：2026-09-13
状态：已确认（用户选择 C 方案并授权开始实现）

## 1. 目标

在现有 ThesisForge 本地工作台基础上完成一次可交付升级：

1. EXE 默认以独立桌面窗口运行，同时保留浏览器打开方式；
2. 补齐完整毕设实验流程：实验分组、参数复用、多实验对比、结果导出、报告对比表；
3. 对现有全部功能做一次系统性检查，修复可能的问题与 bug；
4. 重新打包为独立 EXE 与 Windows 离线整合包，上传 GitHub Release v0.3.0。

## 2. 范围

保留现有全部功能：数据集中心、模型训练、实验记录、AI 分析、AIGC 自检、报告工坊、毕设指南。

本次新增：

- 桌面窗口外壳（pywebview + WebView2，浏览器降级）；
- 实验元数据（名称、分组、备注）与编辑；
- 复制参数创建新实验；
- 实验对比页（多选、指标表、图表、CSV/Markdown 导出）；
- 报告工坊按实验分组输出对比表；
- 毕设指南中的实验方案模板；
- v0.3.0 单元测试与端到端回归。

不包含：联网同步、账号系统、云端训练、自动提交至学校系统。

## 3. 软件形态与架构

### 3.1 现有架构

FastAPI 服务提供 `/api/*` 与静态页面 `web/`；训练任务由独立子进程执行；所有数据落在本地 `data/`；EXE 由 PyInstaller 打包。

### 3.2 v0.3.0 运行形态

1. 双击独立 EXE 或离线包 `ThesisForge.exe` 时，默认进入桌面窗口；
2. 桌面窗口使用 `pywebview` 内嵌 Edge WebView2；
3. WebView2 缺失或启动失败时自动降级为外部浏览器；
4. 页面侧边栏新增「浏览器打开」，窗口模式下调用原生桥接打开外部浏览器；
5. 关闭桌面窗口即停止本地服务，退出程序；
6. 保留启动参数：`--browser` 使用浏览器、`--no-window` 只启动服务不打开界面；
7. 源码开发模式默认仍使用浏览器，降低开发测试干扰。

### 3.3 新增后端模块

- `app/desktop.py`：本地服务线程、健康检查等待、pywebview 启动、浏览器降级、窗口关闭清理；
- `app/experiments.py`：实验对比数据构建、指标规范化、CSV/Markdown 文本生成。

### 3.4 新增/调整 API

- `POST /api/runs`：请求体新增 `name`、`group`、`note`，写入实验配置；
- `PATCH /api/runs/{run_id}/meta`：更新实验名称、分组、备注；
- `POST /api/experiments/compare`：传入 `run_ids`，返回对比表、排序后的指标行、CSV/Markdown 文本；
- 既有接口返回体扩展：`/api/runs` 与 `/api/runs/{run_id}` 增加 `name`、`group`、`note` 字段。

## 4. 桌面窗口详细设计

### 4.1 `app/desktop.py`

对外提供：

- `start_server_thread(host, port) -> tuple[Path|str, ServerShutdown]`：在线程中运行 `uvicorn.Server`，等待 `/api/health` 返回；
- `run_desktop(host, port)`：启动服务线程，创建 pywebview 窗口，窗口关闭后停止服务；
- `run_browser(host, port)`：保持现有 `webbrowser.open` 行为；
- `run_headless(host, port)`：只启动服务，供测试与后台运行。

模块内做以下处理：

- 服务线程使用 `uvicorn.Server(config).run()`，设置 `should_exit` 标志优雅退出；
- 健康检查最多等待 20 秒，等待期间不阻塞 UI；
- pywebview 导入或启动异常时，打印可读错误并回退浏览器；
- 窗口关闭时停止服务线程并 join，确保 EXE 进程退出；
- JS 桥接类 `DesktopBridge` 提供 `open_external_browser()` 与 `version()`；
- 不导出任何 API Key、不记录用户数据到桌面日志。

### 4.2 启动入口 `packaging/ThesisForge.py`

- 冻结独立 EXE 默认调用 `run_desktop`；
- `--browser` 强制浏览器；
- `--no-window` 等价 `--no-browser`，只启动服务；
- `--tf-worker` 仍用于训练子进程，不进入桌面逻辑；
- 离线整合包模式通过环境变量 `THESISFORGE_MODE=desktop` 通知内嵌 Python 以桌面窗口启动。

### 4.3 前端桥接

`web/app.js` 增加：

- 检测 `window.pywebview`，可用时侧边栏「浏览器打开」调用桥接；
- 不可用时调用 `window.open(location.origin, "_blank")`；
- 页面加载时向桥接登记版本号，不依赖桥接做任何核心功能。

## 5. 完整毕设实验闭环

### 5.1 实验元数据

每个实验在 `config.json` 中保存：

```json
{
  "name": "乳腺癌-逻辑回归基线",
  "group": "baseline",
  "note": "第一组基线实验"
}
```

分组取值：`baseline`、`improved`、`ablation`、`custom`。

- 训练页新增「实验名称」「实验分组」「备注」；
- 实验详情页可编辑分组、备注与名称；
- 列表页显示分组徽标。

### 5.2 参数复用创建新实验

实验详情页新增「复用参数」按钮：

- 将当前实验的 `dataset_id`、`task`、`model`、`params`、`target`、`text_column`、划分比例与随机种子写入前端训练表单；
- 跳转「模型训练」页并保持表单已填充；
- 由用户修改超参数后再次提交，实现消融与网格式调参。

### 5.3 实验对比页

新增 `#/compare` 路由与侧边栏入口。

功能：

- 多选已完成实验，默认全选同分组实验；
- 生成指标对比表：行 = 指标，列 = 实验；主指标固定第一列；
- 展示主指标条形图与关键指标热力格；
- 支持复制 Markdown 表格、下载 CSV；
- 未完成/失败实验不参与对比，界面给出提示；
- 对比数据由 `app/experiments.py` 统一构建，避免前端指标口径漂移。

### 5.4 报告工坊对比表

报告生成接口保留 `run_ids`，最终传给 `report.build_report` 的每个实验增加 `group`、`name`：

- 第四章对比表增加「实验分组」列；
- 表格行按：基线 → 改进 → 消融 → 自定义 排序；
- 正文文案由报告模块生成，不再依赖前端拼接。

### 5.5 毕设指南实验方案模板

指南页新增「实验方案模板」区，按当前任务类型生成建议：

- 基线实验建议模型；
- 改进实验建议模型与超参方向；
- 消融实验建议关闭/替换的组件；
- 超参实验建议扫描范围；
- 每组实验需产出 1-2 个已完成 run。

该模板为静态文案，不触发训练。

## 6. 详细检查范围

实现完成后按以下清单逐项复查：

1. 输入与安全：CSV/Excel/zip 导入、下载 SSRF、路径穿越、参数白名单、Host/Origin；
2. 训练稳定性：子进程启动、GPU/CPU 回退、取消与删除、日志轮转边界、模型文件写入；
3. 桌面形态：WebView2 缺失、窗口关闭、端口占用、连续双击、残留进程、EXE 启动参数；
4. 数据一致性：实验分组编辑、指标对比口径、报告选中状态、刷新后状态保持；
5. AI 与报告：AI 未配置、超长文本、并发请求、报告文件名、docx 生成失败；
6. 回归：49 个既有单元测试 + 新增测试 + `test_e2e.py` 全部通过。

## 7. 测试策略

- 单元测试：`test_experiments.py` 覆盖对比表、排序、CSV/Markdown、元数据编辑；
- 单元测试：`test_desktop.py` 覆盖启动参数解析与降级逻辑（mock pywebview）；
- 更新 `tests/test_main.py` 覆盖新字段与 PATCH 接口；
- 更新 `test_e2e.py`：完成一次训练后设置分组、复用参数、生成对比导出；
- 打包冒烟：`--no-window` 启动健康检查、表格训练、报告生成；
- 桌面窗口视觉验证：PyPI 包在用户机器上以可交互模式确认，自动化环境用 mock 覆盖。

## 8. 版本、打包与发布

- `app/config.py` 版本升为 `v0.3.0`；
- `requirements.txt` 增加 `pywebview>=5.0`；
- 独立 EXE 与离线包都包含 `pywebview`；
- 构建产物：
  - `dist/ThesisForge-v0.3.0-win-x64.exe`
  - `dist/ThesisForge-v0.3.0-win64-offline.zip`
- Release 标题与说明升级为 v0.3.0；
- 推送 main 并上传两个资产到 GitHub Release。

## 9. 风险与对策

| 风险 | 对策 |
| --- | --- |
| WebView2 未安装 | 启动失败自动回退浏览器，页面给出提示 |
| pywebview 打包不兼容 | 独立 EXE 冒烟测试 `--no-window`，桌面模式在离线包内嵌 Python 中运行，便于单独重装 |
| 训练子进程在桌面窗口关闭时未退出 | 关闭窗口前广播退出，训练子进程不强制终止，下次启动可查看状态 |
| 对比指标口径不一致 | 所有对比数据由 `app/experiments.py` 生成，前端只渲染 |
| 报告字段或 AI 依赖失败 | 报告生成失败返回可读中文错误，不写半成品 docx |

## 10. 明确不做

- 不做多用户、云同步、账户体系；
- 不保证任何学校 AIGC 检测系统的判定结果；
- 不把模型权重上传到任何服务器。
