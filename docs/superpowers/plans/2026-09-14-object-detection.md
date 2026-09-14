# Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a complete YOLO object detection workflow: YOLO-format dataset import, Ultralytics training, mAP metrics, detection artifacts, and full UI/API coverage.

**Architecture:** Register a new `object_detection` task in the model catalog backed by Ultralytics YOLO. Extend dataset import to detect YOLO-format zips (images/ + labels/ or data.yaml). Add a dedicated detection trainer branch that runs Ultralytics training and exports mAP/precision/recall metrics plus detection artifacts. Wire API validation, training page, and wizard to the new task.

**Tech Stack:** FastAPI, Ultralytics YOLO, pytest, vanilla JavaScript.

---

### Task 1: YOLO Dataset Import and EDA

**Files:**
- Modify: `app/datasets_hub.py`
- Test: `tests/test_detection.py`

- [x] **Step 1: Detect YOLO-format zips on import**

When a zip contains `images/` + `labels/` (optionally with train/val/test subdirs) or a `data.yaml` at root, set `type: "image"`, `task: "object_detection"`, and keep the extracted structure as-is.

- [x] **Step 2: Detection EDA**

Count images, parse `data.yaml` class names, count boxes per class from label files, and plot class balance.

### Task 2: Catalog, API, and UI

**Files:**
- Modify: `app/catalog.py`
- Modify: `app/main.py`
- Modify: `web/app.js`
- Modify: `web/wizard.js`
- Test: `tests/test_detection.py`

- [x] **Step 1: Add object_detection task with YOLO models**

`yolov8n` and `yolov8s`, engine `ultralytics`, params: epochs, imgsz, batch_size, lr0, patience, seed, device.

- [x] **Step 2: API validation**

`object_detection` requires image datasets; torch/ultralytics presence check.

- [x] **Step 3: UI task lists**

Training page and wizard include object_detection for image datasets.

### Task 3: Detection Trainer

**Files:**
- Add: `app/train_detection.py` (standalone entry)
- Test: `tests/test_detection.py`

- [x] **Step 1: Detection training branch**

Use `ultralytics.YOLO` with the dataset `data.yaml`; train with catalog params; export metrics (mAP50, mAP50-95, precision, recall), artifacts (results.png, confusion_matrix.png, sample predictions), and summary with split scheme.

### Task 4: Docs and Regression Gate

**Files:**
- Create: `docs/目标检测指南.md`
- Modify: `README.md`

- [x] **Step 1: Document format, training, metrics, artifacts**

- [x] **Step 2: Run all checks**

`python -m pytest -q` and `python scripts/verify_wizard.py`.
