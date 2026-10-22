"""
Evaluation module for cross-validation, test-set metrics, curves, and threshold analysis.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from src.utils import get_project_root, setup_logger

logger = setup_logger("evaluation")


def evaluate_cross_validation(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform 5-fold Stratified Cross-Validation across all models on training partition.

    Returns DataFrame with mean and std for Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    records = []
    for name, pipeline in models.items():
        logger.info(f"Running {cv_folds}-fold CV for [{name}]...")
        t0 = time.perf_counter()
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        duration = time.perf_counter() - t0

        records.append({
            "Model": name,
            "Accuracy_Mean": float(np.mean(scores["test_accuracy"])),
            "Accuracy_Std": float(np.std(scores["test_accuracy"])),
            "Precision_Mean": float(np.mean(scores["test_precision"])),
            "Precision_Std": float(np.std(scores["test_precision"])),
            "Recall_Mean": float(np.mean(scores["test_recall"])),
            "Recall_Std": float(np.std(scores["test_recall"])),
            "F1_Mean": float(np.mean(scores["test_f1"])),
            "F1_Std": float(np.std(scores["test_f1"])),
            "ROC_AUC_Mean": float(np.mean(scores["test_roc_auc"])),
            "ROC_AUC_Std": float(np.std(scores["test_roc_auc"])),
            "CV_Duration_Sec": round(duration, 2),
        })

    cv_df = pd.DataFrame(records)
    return cv_df


def evaluate_model_on_test(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Fit pipeline on training data and evaluate on untouched test set.

    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, and probability scores.
    """
    logger.info(f"Evaluating [{model_name}] on untouched test set (threshold={threshold})...")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_duration = time.perf_counter() - t0

    # Probability predictions
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-df_vals))
    else:
        y_prob = np.zeros(len(y_test))

    # Threshold-based predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Core evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_str = classification_report(y_test, y_pred, target_names=["Retained (No)", "Departed (Yes)"])

    return {
        "model_name": model_name,
        "fitted_model": model,
        "threshold": threshold,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": report_str,
        "fit_duration_seconds": round(fit_duration, 3),
    }


def find_optimal_f1_threshold(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Determine threshold that optimizes F1-score via cross-validated out-of-fold predictions.
    Avoids looking at the untouched test set.
    """
    from sklearn.model_selection import cross_val_predict
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_probs = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        score = f1_score(y_train, (oof_probs >= thresh).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    logger.info(f"Optimal threshold found from CV: {best_thresh:.2f} (Expected CV F1: {best_f1:.4f})")
    return best_thresh, best_f1


def plot_evaluation_charts(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    output_dir: Optional[str | Path] = None
) -> Dict[str, str]:
    """
    Generate ROC curves, Precision-Recall curves, Confusion Matrices, and Probability distributions.
    Saves high-resolution plots to output_dir.
    """
    if output_dir is None:
        save_path = get_project_root() / "outputs" / "figures"
    else:
        save_path = Path(output_dir)
        if not save_path.is_absolute():
            save_path = get_project_root() / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    fig_paths = {}

    # 1. Multi-Model ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in eval_results.items():
        if name == "Baseline":
            continue
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Benchmark (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", pad=15, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(loc="lower right")
    p_roc = save_path / "14_roc_curves_comparison.png"
    fig.savefig(p_roc, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig_paths["roc_curves"] = str(p_roc)

    # 2. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in eval_results.items():
        if name == "Baseline":
            continue
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ap = average_precision_score(y_test, res["y_prob"])
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.3f})", lw=2)
    ax.set_title("Precision-Recall (PR) Curves Under Class Imbalance", pad=15, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    p_pr = save_path / "15_precision_recall_curves.png"
    fig.savefig(p_pr, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig_paths["pr_curves"] = str(p_pr)

    # 3. 2x2 Grid of Confusion Matrices for the 4 classifiers
    models_to_plot = [m for m in eval_results.keys() if m != "Baseline"][:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for i, name in enumerate(models_to_plot):
        cm = eval_results[name]["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
            xticklabels=["Pred Retained", "Pred Departed"],
            yticklabels=["Actual Retained", "Actual Departed"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[i].set_title(f"{name}\nAcc: {eval_results[name]['accuracy']:.3f} | F1: {eval_results[name]['f1_score']:.3f} | Rec: {eval_results[name]['recall']:.3f}", pad=10)
    fig.suptitle("Confusion Matrices on Held-Out Test Set (N=294)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    p_cm = save_path / "16_confusion_matrices_grid.png"
    fig.savefig(p_cm, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig_paths["confusion_matrices"] = str(p_cm)

    # 4. Predicted Probability Distribution (for best model, e.g. Random Forest or XGBoost)
    lead_model = "Random Forest" if "Random Forest" in eval_results else list(eval_results.keys())[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    probs = eval_results[lead_model]["y_prob"]
    sns.kdeplot(probs[y_test == 0], fill=True, color="#2b5c8f", label="Actual Retained (0)", alpha=0.4, ax=ax)
    sns.kdeplot(probs[y_test == 1], fill=True, color="#d9534f", label="Actual Departed (1)", alpha=0.4, ax=ax)
    ax.axvline(0.5, color="gray", linestyle="--", label="Standard Threshold (0.50)")
    ax.set_title(f"Predicted Probability Distribution - {lead_model}", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Probability of Departure")
    ax.set_ylabel("Density")
    ax.legend()
    p_prob = save_path / "17_predicted_probability_distribution.png"
    fig.savefig(p_prob, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig_paths["probability_distribution"] = str(p_prob)

    return fig_paths


def export_evaluation_tables(
    cv_df: pd.DataFrame,
    test_eval_results: Dict[str, Dict[str, Any]],
    output_dir: Optional[str | Path] = None
) -> Tuple[Path, Path]:
    """Export comparison tables to CSV and Excel."""
    if output_dir is None:
        save_path = get_project_root() / "outputs" / "tables"
    else:
        save_path = Path(output_dir)
        if not save_path.is_absolute():
            save_path = get_project_root() / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    # Build Test comparison DataFrame
    test_records = []
    for name, res in test_eval_results.items():
        test_records.append({
            "Model": name,
            "Threshold": res["threshold"],
            "Accuracy": round(res["accuracy"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1_Score": round(res["f1_score"], 4),
            "ROC_AUC": round(res["roc_auc"], 4),
            "True_Positives": res["true_positives"],
            "False_Positives": res["false_positives"],
            "True_Negatives": res["true_negatives"],
            "False_Negatives": res["false_negatives"],
            "Fit_Duration_Sec": res["fit_duration_seconds"],
        })
    test_df = pd.DataFrame(test_records)

    # Save CSVs
    cv_csv = save_path / "cross_validation_results.csv"
    test_csv = save_path / "model_comparison_test_results.csv"
    cv_df.to_csv(cv_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save Combined Excel
    excel_path = save_path / "attrition_evaluation_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        test_df.to_excel(writer, sheet_name="Test_Comparison", index=False)
        cv_df.to_excel(writer, sheet_name="Cross_Validation", index=False)

    logger.info(f"Exported evaluation tables to {save_path}")
    return test_csv, excel_path
