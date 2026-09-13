# Sprint 2（神经网络注册表与通用训练引擎）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 PyTorch 训练从“只能跑图像 CNN/ResNet”升级为通用神经网络引擎，覆盖表格 MLP、图像 CNN/ResNet18、文本 LSTM/GRU/TextCNN/轻量 Transformer，并统一优化器、学习率调度、早停、权重衰减、梯度裁剪、评估与产物落盘。

**Architecture:** 新增 `app/networks.py`（网络注册表，懒加载 torch）与 `app/trainer.py`（通用 Trainer，可读 run_dir 配置并产出现有实验目录契约），扩展 `app/catalog.py` 的模型 schema，`app/main.py` 按 `engine` 派发训练脚本，`app/train_torch.py` 改为 Trainer 的薄入口。

**Tech Stack:** PyTorch 2.x、torchvision、pandas、numpy、pytest。torch 为可选依赖：`app/networks.py`/`app/trainer.py` 不参与普通模块导入链（main.py 不直接 import），相关测试在无 torch 环境自动跳过。

---

## 数据与训练契约

run_dir 产物保持现有读取方（`runner.list_runs` / `run_detail` / 前端实验详情页）兼容：

- `config.json`：完整参数（含 `engine`、`optimizer/scheduler/...` 训练策略键）。
- `metrics.jsonl`：`{"type":"epoch", ...}` 逐轮曲线。
- `summary.json`：`task/model/params/epochs/best_epoch/metrics/primary_metric/artifacts/...`。
- `best.pt` / `status.json`：最佳模型与运行状态。

训练策略统一键（同时兼容旧 `params` 内键）：

| 键 | 默认 | 说明 |
|---|---|---|
| optimizer | adam | sgd / sgd_momentum / adam / adamw / rmsprop |
| lr | 0.001 | 学习率 |
| momentum | 0.9 | SGD 动量（仅 sgd_momentum 使用） |
| batch_size | 32 | 批大小 |
| epochs | 15 | 训练轮数 |
| scheduler | cosine | cosine / step / plateau / none |
| step_size | 10 | StepLR 步长 |
| gamma | 0.5 | StepLR / ReduceLROnPlateau 系数 |
| weight_decay | 0.0 | L2 权重衰减（由优化器实现） |
| early_stop_patience | 0 | 早停耐心，0 表示关闭 |
| grad_clip | 0.0 | 梯度范数裁剪，0 表示关闭 |
| seed | 42 | 随机种子 |
| device | auto | auto / cpu / cuda |

## Task 1: 网络注册表

**Files:**
- Create: `app/networks.py`
- Create: `tests/test_networks.py`

- [x] **Step 1: 写失败测试**

`tests/test_networks.py` 覆盖：
- 无 torch 时跳过全部用例。
- MLP（分类/回归）前向形状与参数量。
- CNN 与 ResNet18（随机初始化）图像前向。
- LSTM、GRU、TextCNN、Transformer 文本前向。
- 未知网络/任务抛出 `ValueError`。
- `count_parameters` 返回正数。

- [x] **Step 2: 实现 `app/networks.py`**

`build_model(task, model, params, num_classes=None, num_features=None, num_tokens=None, max_seq_len=None, pretrained=None)`：按任务分发到内部构建函数；`_torch()` 懒加载 torch 模块；隐藏层/卷积核/词表等参数做安全解析（默认值兜底、逗号分隔转 int 列表）。

- [x] **Step 3: 运行测试并提交**

```bash
python -m pytest tests/test_networks.py -q
git add app/networks.py tests/test_networks.py
git commit -m "feat: 神经网络注册表（表格/图像/文本）"
```

## Task 2: 通用 Trainer

**Files:**
- Create: `app/trainer.py`
- Create: `tests/test_trainer.py`

- [x] **Step 1: 写失败测试**

`tests/test_trainer.py`（无 torch 自动跳过）覆盖：
- 小表格分类 CSV + MLP 完整跑 2 轮：`status.json` done、`metrics.jsonl` 有 stage/epoch/summary、`summary.json` 有 metrics/primary_metric、`best.pt` 存在。
- 5 种优化器各自可用（1 epoch，Adam 或 SGD 快速验证）。
- 调度器 cosine/step/plateau/none 可用。
- 早停：patience=1 且最优不再提升时提前结束，epochs 记录 < 配置轮数。
- 小文本 CSV + TextCNN 跑 1 轮；回归 CSV + MLP 跑 1 轮。
- 非法配置（未知优化器/未知模型）写 failed 状态并返回非 0。

- [x] **Step 2: 实现 `app/trainer.py`**

核心结构：

```text
Trainer
├─ read/config 清洗：训练策略键从 config["params"] 或 config 顶层合并，缺失用默认值
├─ 随机种子 / 设备选择
├─ 数据加载
│  ├─ tabular：pandas CSV，数值标准化+类别 one-hot（仅训练集拟合），train/val/test 划分
│  ├─ image：torchvision ImageFolder，Resize+增强+ImageNet 归一化，三方划分
│  └─ text：训练集建词表（中英混合 tokenizer），pad/truncate，三方划分
├─ 模型：app.networks.build_model
├─ 优化器/调度器/早停/梯度裁剪
├─ 逐轮评估 + metrics.jsonl
├─ 测试集终评 + summary.json / 图表（曲线/混淆矩阵/ROC/类别分布）
└─ status.json
```

`train_from_run_dir(run_dir)` 返回退出码并捕获异常写 failed 状态；`trainer.fit(config, run_dir)` 供单元测试直接调用。产物沿用 `plot_confusion_matrix` / `plot_training_curves` / `plot_roc` / `plot_class_balance`。

- [x] **Step 3: 运行测试并提交**

```bash
python -m pytest tests/test_trainer.py -q
git add app/trainer.py tests/test_trainer.py
git commit -m "feat: 通用 PyTorch 训练引擎（优化器/调度/早停/评估）"
```

## Task 3: catalog 扩展与引擎派发

**Files:**
- Modify: `app/catalog.py`
- Modify: `app/main.py`
- Modify: `tests/test_catalog.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: 扩展模型 schema**

- `tabular_classification.models.mlp` 改为 `engine: torch`，参数换为架构（hidden_sizes/activation/dropout）+ 训练策略（optimizer/lr/batch_size/epochs/...）。
- `tabular_regression.models` 新增 `mlp`（engine torch）。
- `text_classification.models` 新增 `lstm`、`gru`、`textcnn`、`transformer`（engine torch），保留 TF-IDF 基线。
- `image_classification.models.cnn/resnet18` 标记 `engine: torch` 并补充训练策略/架构参数。
- 保持现有测试不破坏：`all_models_flat`、`sanitize_params`、choice/bool 规则不变。

- [ ] **Step 2: create_run 按 engine 派发**

```text
engine = spec.get("engine") or "sklearn"
script = "train_torch.py" if engine == "torch" else "train_sklearn.py"
engine == "torch" 时检测 torch 可用，否则返回可操作的安装提示
```

新增测试：torch 模型中 `script` 为 `train_torch.py`；旧 sklearn 模型仍为 `train_sklearn.py`。

- [ ] **Step 3: 运行测试并提交**

```bash
python -m pytest tests/test_catalog.py tests/test_main.py -q
git add app/catalog.py app/main.py tests/test_catalog.py tests/test_main.py
git commit -m "feat: 目录覆盖全部神经网络并支持引擎自动派发"
```

## Task 4: 训练脚本升级与全量回归

**Files:**
- Modify: `app/train_torch.py`
- Run: 全量测试 + compileall + 浏览器冒烟（可选）

- [ ] **Step 1: 重写 `train_torch.py`**

薄入口：解析 `--run-dir`，调用 `trainer.train_from_run_dir`，保留退出码与 failed 日志契约。

- [ ] **Step 2: 冒烟验证**

用内置 Wine 数据集 + `tabular_classification/mlp` 建一个假 run_dir 并直接跑 `train_from_run_dir` 2 epochs，确认日志、summary、best.pt、曲线图齐全。

- [ ] **Step 3: 全量回归**

```bash
python -m pytest tests -q
python -m compileall -q app packaging
```

- [ ] **Step 4: 提交收尾**

```bash
git add app/train_torch.py docs/superpowers/plans/2026-09-13-sprint2-train-engine.md
git commit -m "feat: 神经网络注册表与通用训练引擎"
```

## Sprint 2 完成后的下一个 Sprint

Sprint 3：评估增强（重复运行 mean±std、混淆矩阵/ROC 详情）、消融实验自动创建与批量训练、实验对比增强。
