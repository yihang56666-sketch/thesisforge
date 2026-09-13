# ThesisForge v0.3.0 执行计划

日期：2026-09-13
状态：执行中（用户已确认设计规范并开始）

## 1. 后端

- `app/config.py`：版本号升为 `0.3.0`。
- `app/runner.py`：新增实验分组常量、`update_meta()`；`list_runs()` 返回 `name/group/note`。
- 新增 `app/experiments.py`：分组排序、`build_comparison()`、CSV/Markdown 文本。
- `app/main.py`：`CreateRunReq` 增加元数据字段；新增 `PATCH /api/runs/{id}/meta` 与
  `POST /api/experiments/compare`；报告 runs 增加分组字段；新增桌面模式启动入口 `main()`。
- 新增 `app/desktop.py`：`start_server_thread()`、桌面窗口、浏览器/无头模式、
  `resolve_startup_mode()`、`DesktopBridge`。
- `app/report.py`：实验按 基线→改进→消融→自定义 排序，表 4-2 增加实验分组列。

## 2. 前端

- `web/index.html`：新增「实验对比」导航与「浏览器打开」入口。
- `web/app.js`：
  - 训练表单增加 实验名称/分组/备注；
  - 实验列表显示分组徽标；
  - 详情页支持编辑元数据、复用参数、加入报告；
  - 新增 `#/compare` 对比页（多选、指标表、主指标条形图、CSV/Markdown 导出）；
  - 报告勾选状态保持；指南页增加实验方案模板；接入 pywebview 桥接。
- `web/style.css`：分组徽标、对比表、热力格、条形图等样式。

## 3. 测试

- 新增 `tests/test_experiments.py`、`tests/test_runner_meta.py`、`tests/test_desktop.py`；
- 更新 `tests/test_main.py` 与 `test_e2e.py`；
- 复位：`compileall`、`unittest discover`、启动服务后 `test_e2e.py`。

## 4. 打包与发布

- `requirements.txt` 增加 `pywebview>=5.0`；
- `packaging/build_exe.py`：启动器排除 pywebview，独立版保留；
- `packaging/build_package.py`：v0.3.0、加入 pywebview、桌面窗口说明；
- `packaging/publish_github.py`、`upload_github.sh`、README、release notes 升到 v0.3.0；
- 构建 `ThesisForge-v0.3.0-win-x64.exe` 与 `ThesisForge-v0.3.0-win64-offline.zip`；
- 冒烟验证后推 `main` 并上传 Release v0.3.0。

## 5. 完成检查

- 单元测试 + E2E 全部通过；
- 输入安全、路径校验、参数白名单、训练子进程、端口冲突、桌面降级逐项复查；
- GitHub Release 两个资产可下载。
