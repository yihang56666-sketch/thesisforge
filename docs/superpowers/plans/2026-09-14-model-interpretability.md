# Model Interpretability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic, task-aware interpretability artifacts for trained deep models.

**Architecture:** A new `app/explain.py` computes Grad-CAM for image models, token-gradient attribution for text models, and permutation importance for tabular MLPs. The existing PyTorch trainer calls these helpers after final evaluation and records outputs in `summary.json`.

**Tech Stack:** PyTorch, torchvision, matplotlib, existing trainer and artifact system.

---

### Task 1: Explanation Helpers

**Files:**
- Create: `app/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write failing helper tests**

```python
def test_normalize_cam():
    from app.explain import normalize_cam
    assert abs(normalize_cam([0.0, 1.0])[-1] - 1.0) < 1e-6

def test_token_attribution():
    from app.explain import token_attribution
    result = token_attribution(lambda x: x["logits"], x={"logits": None}, target_idx=0)
    assert isinstance(result, list)
```

- [ ] **Step 2: Implement helpers**

`normalize_cam()`, `grad_cam_image()`, `token_attribution()`, and `permutation_importance()` should be independent of the web layer.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_explain.py -q`
Expected: PASS.

### Task 2: Trainer Integration

**Files:**
- Modify: `app/trainer.py`

- [ ] **Step 1: Add task-aware calls**

After final evaluation:
- image classification saves `grad_cam.png`;
- text classification saves `token_attribution.png` and `token_attribution.json`;
- tabular MLP saves `permutation_importance.png`.

- [ ] **Step 2: Record artifacts**

Each generated file must be appended to `summary["artifacts"]`; failures must be logged and skipped without breaking training.

- [ ] **Step 3: Run training tests**

Run: `python -m pytest tests/test_trainer.py -q`
Expected: PASS.

### Task 3: UI and Docs

**Files:**
- Modify: `web/app.js`
- Create: `docs/模型解释性指南.md`
- Modify: `README.md`

- [ ] **Step 1: Surface explainability**

The run-detail artifact grid should display the new images automatically.

- [ ] **Step 2: Write guide**

Explain what each plot means, how to read it, and how to cite it in a thesis.

- [ ] **Step 3: Smoke check**

Run: `python scripts/verify_wizard.py`
Expected: PASS.
