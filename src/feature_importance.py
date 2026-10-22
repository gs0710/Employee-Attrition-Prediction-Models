"""
Feature importance extraction and cross-model explainability module.
Analyzes Logistic Regression coefficients, Tree MDI, and Permutation Importance.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from src.preprocessing import get_feature_names_after_preprocessing
from src.utils import get_project_root, setup_logger

logger = setup_logger("feature_importance")


def extract_logistic_regression_coefficients(
    fitted_pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """Extract and sort Logistic Regression coefficients and odds ratios."""
    clf = fitted_pipeline.named_steps["classifier"]
    coefs = clf.coef_[0]

    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Odds_Ratio": np.exp(coefs),
        "Absolute_Importance": np.abs(coefs),
        "Direction": ["Increases Attrition Risk" if c > 0 else "Protective (Decreases Risk)" for c in coefs]
    }).sort_values("Absolute_Importance", ascending=False).reset_index(drop=True)

    return df_coef


def extract_tree_feature_importance(
    fitted_pipeline: Pipeline,
    feature_names: List[str],
    model_name: str = "Tree Model"
) -> pd.DataFrame:
    """Extract impurity-based (MDI) feature importances from fitted tree ensemble."""
    clf = fitted_pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        raise ValueError(f"Classifier {model_name} does not expose feature_importances_.")

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    df_imp["Normalized_Percentage"] = (df_imp["Importance"] / df_imp["Importance"].sum()) * 100
    return df_imp


def compute_permutation_importance(
    fitted_pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation feature importance on held-out data to assess genuine predictive impact.
    Operates at raw input column level.
    """
    logger.info(f"Computing permutation importance ({n_repeats} repeats, metric: {scoring})...")
    perm = permutation_importance(
        fitted_pipeline, X_val, y_val,
        n_repeats=n_repeats, scoring=scoring,
        random_state=random_state, n_jobs=-1
    )

    df_perm = pd.DataFrame({
        "Feature": X_val.columns,
        "Mean_Importance": perm.importances_mean,
        "Std_Importance": perm.importances_std,
    }).sort_values("Mean_Importance", ascending=False).reset_index(drop=True)

    return df_perm


def map_encoded_feature_to_original(feature_name: str, original_columns: List[str]) -> str:
    """Map one-hot encoded dummy column (e.g., 'Department_Sales') back to its raw feature name."""
    for col in sorted(original_columns, key=len, reverse=True):
        if feature_name.startswith(col):
            return col
    return feature_name


def compare_feature_importances(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Extract, normalize, and compare top features across all fitted classifiers.
    Saves cross-model comparison plot and CSV summary table.
    """
    if output_dir is None:
        fig_dir = get_project_root() / "outputs" / "figures"
        tab_dir = get_project_root() / "outputs" / "tables"
    else:
        fig_dir = Path(output_dir) / "figures"
        tab_dir = Path(output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    comparison_dict = {}

    for name, pipeline in models.items():
        if name == "Baseline":
            continue
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = get_feature_names_after_preprocessing(preprocessor, X_test.columns.tolist())
            clf = pipeline.named_steps["classifier"]

            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
                norm_imp = imp / np.sum(imp)
                comparison_dict[name] = pd.Series(norm_imp, index=feature_names)
            elif hasattr(clf, "coef_"):
                imp = np.abs(clf.coef_[0])
                norm_imp = imp / np.sum(imp)
                comparison_dict[name] = pd.Series(norm_imp, index=feature_names)
        except Exception as exc:
            logger.warning(f"Could not extract feature importances for {name}: {exc}")

    if not comparison_dict:
        return pd.DataFrame(), ""

    df_comp = pd.DataFrame(comparison_dict).fillna(0)
    df_comp["Mean_Across_Models"] = df_comp.mean(axis=1)
    df_comp = df_comp.sort_values("Mean_Across_Models", ascending=False)

    # Save summary table
    csv_path = tab_dir / "feature_importance_summary.csv"
    df_comp.to_csv(csv_path)

    # Plot Top 15 features across models
    top15 = df_comp.head(15)
    fig, ax = plt.subplots(figsize=(11, 7))
    models_to_plot = [c for c in top15.columns if c != "Mean_Across_Models"]
    top15[models_to_plot].plot(kind="barh", ax=ax, width=0.8, colormap="viridis")
    ax.invert_yaxis()
    ax.set_title("Top 15 Predictive Features Across Classification Models", pad=15, fontweight="bold")
    ax.set_xlabel("Normalized Feature Importance / Relative Weight")
    ax.set_ylabel("Engineered Feature")
    ax.legend(title="Model", loc="lower right")

    fig_path = fig_dir / "18_feature_importance_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved feature importance comparison to {fig_path} and {csv_path}")
    return df_comp, str(fig_path)
