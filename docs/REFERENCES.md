# 开源借鉴清单

本项目在设计与实现上参考/借力了以下优秀开源项目与资源。

## 直接依赖

| 项目 | 用途 |
| --- | --- |
| [FastAPI](https://github.com/fastapi/fastapi) | 后端框架：自动 OpenAPI 文档、类型校验（Pydantic） |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | 表格/文本任务的 14 个模型、EDA 统计、内置数据集 |
| [PyTorch / torchvision](https://github.com/pytorch/pytorch) | 图像分类训练（CNN、ResNet18 迁移学习），自动 GPU |
| [pandas / numpy](https://github.com/pandas-dev/pandas) | 数据处理与统计 |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | 全部图表（中文字体适配） |
| [python-docx](https://github.com/python-openxml/python-docx) | 论文 Word 初稿生成 |
| [httpx](https://github.com/encode/httpx) | LLM 接口调用与数据集下载 |

## 设计借鉴

| 项目 | 借鉴点 |
| --- | --- |
| [MLflow](https://github.com/mlflow/mlflow) | 实验留档思想：每次 run 一个目录，存 config/指标/工件，本工具 `data/runs/` 即轻量版 Tracking |
| [TensorBoard / Weights & Biases](https://github.com/tensorflow/tensorboard) | 逐 epoch 写 metrics.jsonl 再绘制曲线的做法 |
| [Streamlit / Gradio](https://github.com/gradio-app/gradio) | 「把脚本变面板」的产品形态与表单即参数的交互 |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | 一行命令完成训练的封装思路：`runner.py` + `train_*.py` 子进程模式 |
| [Optuna / AutoGluon](https://github.com/optuna/optuna) | 超参数目录化（`catalog.py` 的参数 schema），后续可接入搜索 |
| [HuggingFace datasets](https://github.com/huggingface/datasets) | 数据集卡片（meta.json）与「一键载入」体验 |
| [Kaggle / UCI / Papers With Code](https://www.kaggle.com/datasets) | 数据来源与基线参照，指南页推荐 |
| [Label Studio](https://github.com/HumanSignal/label-studio) | 图像数据集 `类别文件夹/图片` 的组织约定（ImageFolder 布局） |
| [OpenAI API 规范](https://platform.openai.com/docs/api-reference) | LLM 调用采用 OpenAI 兼容 `/v1/chat/completions`，从而同时支持 DeepSeek/智谱/通义/Kimi |

## 论文格式与 AIGC 方向

| 项目/资源 | 借鉴点 |
| --- | --- |
| [GB/T 7714-2015](https://std.samr.gov.cn/gb/search/gbDetailed?id=71F7726730D6E1B0E21C0B95B6B0C11E) | 参考文献著录规则（报告工坊的文献格式） |
| 中文本科毕业论文通用排版规范 | 宋体小四/1.5 倍行距/三线表/页码域/TOC 域等报告默认规格的依据 |
| [Hello-SimpleAI/chatgpt-comparison-detection (HC3)](https://github.com/hello-simpleai/chatgpt-comparison-detection) | 中英双语 AI 文本检测的开源数据集与分类器。工具的「AIGC 自检」借鉴其结论：AI 文本在模板短语密度、句长均匀度上与人类文本系统性不同 |
| [GPTZero 的困惑度+突发度思路](https://gptzero.me) | 「句长波动（burstiness）」作为 AI 特征信号，是自检评分的三大信号之一 |
| [zejunwang1/GPTDetector](https://github.com/zejunwang1/GPTDetector) / DetectGPT | 检测技术路线调研参照；后续可把 HC3 微调的 RoBERTa 检测器接入自检（当前为启发式规则，不依赖 GPU） |
| PyPI: synonyms（中文近义词库） | 可选增强：替换词表可扩展为近义词轮换（当前用内置精简词表保证零依赖） |

> 说明：AIGC 自检与「降 AI 味」做的是**让文字更自然、更有个人风格**（去模板腔、打散句长、
> 减少列表体），并诚实地提示最可靠的方式是自己重写。工具不声称能保证任何检测系统的结果。

## 后续可集成

- **Optuna**：把「训练页」的表单升级为贝叶斯超参搜索；
- **Ultralytics**：加入目标检测任务（数据组织同为 YOLO 目录约定）；
- **transformers + PEFT**：加入 LLM 微调（LoRA）任务；
- **Grad-CAM**：为图像分类增加可解释性热力图；
- **LaTeX 导出**：`pandoc` 把报告同时导出 LaTeX/PDF。
