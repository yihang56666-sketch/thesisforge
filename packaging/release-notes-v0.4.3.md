# 毕设工坊 ThesisForge v0.4.3

v0.4.3 把视觉检测路线补得更完整：新增实例分割与 RT-DETR 检测，修复分割数据集导入时类别与任务识别不生效的问题，并升级了检测/分割报告的方法学描述。

## 本次新增

- **实例分割**：支持 YOLOv8n-seg / YOLOv8s-seg；可直接导入多边形/掩码标注的 YOLO 分割数据集，实验报告自动使用 mask 口径描述。
- **RT-DETR 检测**：新增 Transformer 检测器路线，可与 YOLO 检测方法做架构对比；适合有一定显存预算的中小型数据集。
- **分割数据导入修复**：`data.yaml` 中的 `task: segment` 现在会正确识别为实例分割；类别、图像数量与 EDA 统计也会正常生成。
- **训练报告增强**：检测与分割实验在报告中补充数据划分、指标口径、随机种子与可复现说明，结果图与验证集可视化更完整。
- **打包与发布版本化**：EXE、离线包、Release 名称和说明文件统一由 `APP_VERSION` 派生，后续版本号不再需要散落多处手改。

## 资产

| 文件 | 说明 |
| --- | --- |
| `ThesisForge-v0.4.3-win-x64.exe` | 独立单文件版：自带 Python 与表格/文本/报告全部依赖，默认桌面窗口（不含 PyTorch） |
| `ThesisForge-v0.4.3-win64-offline.zip` | Windows 离线整合包：内嵌 Python，解压后双击 `ThesisForge.exe`，可自行安装 CPU/GPU 版 PyTorch 做视觉训练 |

## 已知边界

- 独立单文件版为控制体积不包含 PyTorch/Ultralytics，视觉任务请使用离线整合包并按提示安装训练依赖。
- RT-DETR 显存占用高于 YOLOv8n，小显卡建议从 YOLOv8n/s 或分割 nano 版开始。
- 导出的 `predict.py` 需要 Python 运行环境；离线整合包已内置，源码运行方式也直接支持。
