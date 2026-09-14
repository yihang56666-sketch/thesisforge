"""表格/文本训练脚本（独立子进程运行）。

用法: python app/train_sklearn.py --run-dir data/runs/<run_id>
从 run_dir/config.json 读取配置；产出 metrics.jsonl / summary.json / 图表 / model.pkl / status.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.plots import (  # noqa: E402
    plot_class_balance, plot_confusion_matrix, plot_correlation_heatmap,
    plot_cv_scores, plot_feature_importance, plot_roc,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe(base: Path, name: str) -> Path:
    base = base.resolve()
    t = base / name
    if ".." in t.parts or not t.resolve().is_relative_to(base):
        raise ValueError("非法路径")
    return t.resolve()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()

    cfg = json.loads(safe(run_dir, "config.json").read_text(encoding="utf-8"))
    task = cfg["task"]
    model_key = cfg["model"]
    params = cfg.get("params") or {}
    seed = int(cfg.get("random_state", 42))
    test_size = float(cfg.get("test_size", 0.2))
    dataset_dir = Path(cfg["dataset_dir"]).resolve()
    target = cfg.get("target")
    text_column = cfg.get("text_column")

    metrics_events: list[dict] = []

    def emit(event: dict) -> None:
        metrics_events.append(event)
        safe(run_dir, "metrics.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in metrics_events), encoding="utf-8"
        )

    t0 = time.time()
    try:
        log(f"任务: {task} | 模型: {model_key} | 参数: {params}")
        csv_path = safe(dataset_dir, "dataset.csv")
        df = pd.read_csv(csv_path)
        log(f"数据载入完成: {df.shape[0]} 行 × {df.shape[1]} 列")
        if not target or target not in df.columns:
            target = df.columns[-1]
            log(f"标签列未指定，回退为最后一列: {target}")
        if task == "text_classification" and (not text_column or text_column not in df.columns):
            raise ValueError(f"文本列 {text_column!r} 不在数据集中")

        # ---------------- 特征/标签拆分
        if task == "text_classification":
            X_raw = df[text_column].astype(str).fillna("")
            y_raw = df[target].astype(str)
            feature_names = None
        else:
            X_raw = df.drop(columns=[target])
            y_raw = df[target]
            if task == "tabular_classification":
                y_raw = y_raw.astype(str)

        from sklearn.model_selection import train_test_split

        stratify = None
        if task != "tabular_regression" and y_raw.nunique() > 1:
            counts = y_raw.value_counts()
            if counts.min() >= 2:
                stratify = y_raw
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=seed, stratify=stratify
        )
        log(f"划分完成: 训练 {len(X_tr)} 条 / 测试 {len(X_te)} 条 (test_size={test_size})")
        emit({"type": "stage", "name": "split", "n_train": int(len(X_tr)), "n_test": int(len(X_te))})

        # ---------------- 构建 Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor,
            RandomForestClassifier, RandomForestRegressor,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.svm import SVC
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import f_classif  # noqa: F401

        def est_for(mkey: str, p: dict):
            if task == "tabular_classification":
                if mkey == "logistic_regression":
                    return LogisticRegression(C=float(p.get("C", 1.0)), max_iter=int(p.get("max_iter", 1000)), random_state=seed)
                if mkey == "random_forest":
                    md = int(p.get("max_depth", 0) or 0)
                    return RandomForestClassifier(n_estimators=int(p.get("n_estimators", 200)), max_depth=md or None, random_state=seed, n_jobs=-1)
                if mkey == "svm":
                    return SVC(C=float(p.get("C", 1.0)), kernel=p.get("kernel", "rbf"), probability=True, random_state=seed)
                if mkey == "gradient_boosting":
                    return GradientBoostingClassifier(n_estimators=int(p.get("n_estimators", 200)), learning_rate=float(p.get("learning_rate", 0.1)), max_depth=int(p.get("max_depth", 3)), random_state=seed)
                if mkey == "knn":
                    return KNeighborsClassifier(n_neighbors=int(p.get("n_neighbors", 5)))
                if mkey == "mlp":
                    hidden = tuple(int(x) for x in str(p.get("hidden_sizes", "128,64")).split(",") if x.strip())
                    return MLPClassifier(hidden_layer_sizes=hidden, learning_rate_init=float(p.get("learning_rate_init", 0.001)), max_iter=int(p.get("max_iter", 300)), random_state=seed)
            if task == "tabular_regression":
                if mkey == "linear_regression":
                    return LinearRegression()
                if mkey == "ridge":
                    return Ridge(alpha=float(p.get("alpha", 1.0)), random_state=seed)
                if mkey == "random_forest_regressor":
                    md = int(p.get("max_depth", 0) or 0)
                    return RandomForestRegressor(n_estimators=int(p.get("n_estimators", 200)), max_depth=md or None, random_state=seed, n_jobs=-1)
                if mkey == "gradient_boosting_regressor":
                    return GradientBoostingRegressor(n_estimators=int(p.get("n_estimators", 200)), learning_rate=float(p.get("learning_rate", 0.1)), max_depth=int(p.get("max_depth", 3)), random_state=seed)
            if task == "text_classification":
                vec = TfidfVectorizer(max_features=int(p.get("max_features", 20000)), ngram_range=(1, int(p.get("ngram_max", 2))))
                if mkey == "tfidf_logreg":
                    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(C=float(p.get("C", 1.0)), max_iter=1000, random_state=seed))])
                if mkey == "tfidf_svm":
                    return Pipeline([("tfidf", vec), ("clf", SVC(C=float(p.get("C", 1.0)), kernel="linear", probability=True, random_state=seed))])
            raise ValueError(f"未知模型: {mkey} (任务 {task})")

        if task == "text_classification":
            pipe = est_for(model_key, params)
        else:
            from app.prep import build_tabular_transformer

            num_cols = list(X_raw.select_dtypes(include=[np.number]).columns)
            cat_cols = [c for c in X_raw.columns if c not in num_cols]
            log(f"数值特征 {len(num_cols)} 个，类别特征 {len(cat_cols)} 个")
            pre = build_tabular_transformer(cfg.get("prep") or {}, num_cols, cat_cols)
            pipe = Pipeline([("pre", pre), ("clf", est_for(model_key, params))])

        # ---------------- 交叉验证
        cv_scores: list[float] = []
        if min(len(X_tr), 30) >= 30:
            from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

            n_splits = 5 if len(X_tr) >= 200 else 3
            if task == "tabular_regression":
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                scoring = "r2"
            else:
                min_class = y_tr.value_counts().min()
                n_splits = min(n_splits, int(min_class))
                cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
                scoring = "accuracy"
            log(f"开始 {n_splits} 折交叉验证 (scoring={scoring}) ...")
            scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
            cv_scores = [float(s) for s in scores]
            log(f"交叉验证得分: {[f'{s:.4f}' for s in cv_scores]} (均值 {np.mean(cv_scores):.4f})")
            emit({"type": "cv", "scores": cv_scores, "scoring": scoring})
        else:
            log("训练样本过少，跳过交叉验证")

        # ---------------- 训练与评估
        log("开始训练最终模型 ...")
        pipe.fit(X_tr, y_tr)
        log("训练完成，开始测试集评估 ...")
        y_pred = pipe.predict(X_te)

        summary: dict = {
            "task": task, "model": model_key, "model_label": cfg.get("model_label", model_key),
            "dataset_name": cfg.get("dataset_name"), "params": params, "test_size": test_size,
            "random_state": seed, "cv_scores": cv_scores,
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        }
        artifacts: list[str] = []

        if task == "tabular_regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mse = float(mean_squared_error(y_te, y_pred))
            metrics = {
                "r2": float(r2_score(y_te, y_pred)),
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "rmse": float(np.sqrt(mse)),
                "mse": mse,
            }
            summary["metrics"] = metrics
            summary["primary_metric"] = {"name": "r2", "value": metrics["r2"]}
            log(f"测试集指标: R²={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}")
            # 预测-真实散点
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            ax.scatter(y_te, y_pred, s=14, alpha=0.6, color="#4f7cff")
            lims = [min(np.min(y_te), np.min(y_pred)), max(np.max(y_te), np.max(y_pred))]
            ax.plot(lims, lims, "--", color="#e74c3c")
            ax.set_xlabel("真实值"), ax.set_ylabel("预测值"), ax.set_title("预测值 vs 真实值")
            ax.grid(alpha=0.25, linestyle="--")
            fig.tight_layout()
            fig.savefig(safe(run_dir, "pred_vs_true.png"), dpi=130, bbox_inches="tight")
            import matplotlib.pyplot as _plt

            _plt.close(fig)
            artifacts.append("pred_vs_true.png")
        else:
            from sklearn.metrics import (
                accuracy_score, classification_report, confusion_matrix,
                f1_score, precision_score, recall_score,
            )

            labels = sorted(pd.unique(pd.concat([y_te, pd.Series(y_pred)], ignore_index=True)))
            metrics = {
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "f1_macro": float(f1_score(y_te, y_pred, average="macro", labels=labels, zero_division=0)),
                "f1_weighted": float(f1_score(y_te, y_pred, average="weighted", labels=labels, zero_division=0)),
                "precision_macro": float(precision_score(y_te, y_pred, average="macro", labels=labels, zero_division=0)),
                "recall_macro": float(recall_score(y_te, y_pred, average="macro", labels=labels, zero_division=0)),
            }
            summary["metrics"] = metrics
            summary["primary_metric"] = {"name": "accuracy", "value": metrics["accuracy"]}
            summary["classes"] = [str(c) for c in labels]
            log(f"测试集指标: acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}")

            cm = confusion_matrix(y_te, y_pred, labels=labels)
            plot_confusion_matrix(cm, [str(c) for c in labels], safe(run_dir, "confusion_matrix.png"))
            artifacts.append("confusion_matrix.png")
            summary["confusion_matrix"] = cm.tolist()

            report = classification_report(y_te, y_pred, labels=labels, zero_division=0, output_dict=True)
            summary["per_class"] = report

            if y_te.nunique() == 2 and hasattr(pipe, "predict_proba"):
                try:
                    from sklearn.metrics import roc_auc_score, roc_curve

                    proba = pipe.predict_proba(X_te)[:, 1]
                    pos_label = sorted(y_te.unique())[-1]
                    fpr, tpr, _ = roc_curve((y_te == pos_label).astype(int), proba)
                    auc = float(roc_auc_score((y_te == pos_label).astype(int), proba))
                    metrics["roc_auc"] = auc
                    plot_roc(fpr, tpr, auc, safe(run_dir, "roc.png"))
                    artifacts.append("roc.png")
                except Exception as e:
                    log(f"ROC 计算跳过: {e}")

            counts = y_raw.value_counts().to_dict()
            plot_class_balance(counts, safe(run_dir, "class_balance.png"))
            artifacts.append("class_balance.png")

        # ---------------- 特征重要性
        try:
            clf = pipe.named_steps.get("clf") if task != "text_classification" else pipe
            pre = pipe.named_steps.get("pre") if task != "text_classification" else None
            names = None
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                names = list(pre.get_feature_names_out())
            imp = getattr(clf, "feature_importances_", None)
            if imp is None and hasattr(clf, "coef_"):
                imp = np.abs(np.asarray(clf.coef_)).sum(axis=0) if np.asarray(clf.coef_).ndim > 1 else np.abs(np.asarray(clf.coef_)).ravel()
            if imp is not None and names is None and task == "text_classification":
                names = list(pipe.named_steps["tfidf"].get_feature_names_out())
            if imp is not None and names is not None and len(imp) == len(names):
                p = plot_feature_importance(names, np.asarray(imp), safe(run_dir, "feature_importance.png"))
                if p:
                    artifacts.append("feature_importance.png")
                    summary["feature_importance_top"] = [
                        {"feature": str(names[i]), "value": float(imp[i])}
                        for i in np.argsort(imp)[::-1][:10]
                    ]
        except Exception as e:
            log(f"特征重要性提取跳过: {e}")

        if cv_scores:
            plot_cv_scores(cv_scores, safe(run_dir, "cv_scores.png"))
            artifacts.append("cv_scores.png")

        # ---------------- 保存模型
        joblib.dump(pipe, safe(run_dir, "model.pkl"))
        log("模型已保存: model.pkl")

        summary["artifacts"] = artifacts
        summary["train_time_sec"] = round(time.time() - t0, 1)
        summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            from app.export_predict import write_predict_script

            write_predict_script(run_dir, "sklearn", cfg)
            if "predict.py" not in artifacts:
                artifacts.append("predict.py")
            summary["artifacts"] = artifacts
            log("已生成推理脚本: predict.py")
        except Exception as e:
            log(f"推理脚本生成跳过: {e}")
        write_json(safe(run_dir, "summary.json"), summary)
        emit({"type": "summary", "primary_metric": summary["primary_metric"], "metrics": metrics})
        log(f"全部完成，用时 {summary['train_time_sec']}s")
        write_json(safe(run_dir, "status.json"), {"state": "done", "error": None, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 0
    except Exception:
        err = traceback.format_exc()
        log("训练失败:\n" + err)
        low = err.lower()
        if "out of memory" in low or "cuda oom" in low or "cuda out of memory" in low:
            kind, hint = "oom", "显存/内存不足。建议减小 batch_size、image_size 或 hidden_sizes，降低网络层数，或改用 CPU 训练。"
        elif "no space left" in low or "errno 28" in low or ("disk" in low and "space" in low):
            kind, hint = "disk", "磁盘空间不足。请清理实验输出目录所在磁盘，删除不再需要的实验或临时文件后重试。"
        else:
            kind, hint = "error", "训练异常，具体原因见下方错误信息与运行日志（data/logs/launch.log）。"
        (run_dir / "status.json").write_text(
            json.dumps({"state": "failed", "error": err[-1500:], "failure_kind": kind,
                        "failure_hint": hint, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
