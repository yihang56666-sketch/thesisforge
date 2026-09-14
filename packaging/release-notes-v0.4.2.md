# 毕设工坊 ThesisForge v0.4.2

v0.4.2 补齐了「训练完成后怎么把模型用起来」的最后一环：每个实验目录现在会额外生成可直接运行的 `predict.py`，并把模型文件、预测脚本一起放进实验详情页下载区。

## 本次新增

- **导出推理脚本**：训练完成后自动生成 `predict.py`，表格、文本、图像三类任务都支持用新样本做单条预测。
- **模型权重可下载**：实验详情页现在可以直接下载 `best.pt` / `model.pkl`，与 `predict.py` 配套使用。
- **推理所需信息随模型保存**：PyTorch 模型检查点会保存词表、预处理流程、类别数与图像尺寸，sklearn 模型继续使用完整 Pipeline，避免换新数据后无法复现预测。
- **导出脚本回归测试**：新增自动测试，实际训练后调用导出的 `predict.py`，确认它能独立加载模型并输出预测标签。

## 使用方式

在实验详情页下载 `predict.py` 和模型文件，放到同一个目录后：

```bash
# 表格任务
python predict.py --input 新样本.csv

# 文本任务
python predict.py --text "一段新的文本"

# 图像任务
python predict.py --image 新图片.png
```

离线整合包内已带 Python 运行环境，可在包根目录使用 `runtime\python.exe` 运行；独立单文件版主要用于界面操作，不做图像训练。

## 资产

| 文件 | 说明 |
| --- | --- |
| `ThesisForge-v0.4.2-win-x64.exe` | 独立单文件版：自带 Python 与表格/文本/报告全部依赖，默认桌面窗口（不含 PyTorch） |
| `ThesisForge-v0.4.2-win64-offline.zip` | Windows 离线整合包：内嵌 Python，解压后双击 `ThesisForge.exe`，可自行安装 CPU/GPU 版 PyTorch 做图像训练 |

## 已知边界

- 独立单文件版为控制体积不包含 PyTorch，图像分类请使用离线整合包。
- 导出的 `predict.py` 需要 Python 运行环境；离线整合包已内置，源码运行方式也直接支持。
- 桌面窗口依赖 WebView2；未安装时程序会自动改用浏览器打开，不影响使用。
