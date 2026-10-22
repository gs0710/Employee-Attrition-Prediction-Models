"""
Model definitions and pipeline construction for classification and optional regression.
Includes Baseline, Logistic Regression, Decision Tree, Random Forest, and XGBoost.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor
from src.utils import setup_logger

logger = setup_logger("models")


def get_baseline_pipeline() -> Pipeline:
    """
    Construct baseline classifier pipeline using DummyClassifier (predicts majority class).
    Establishes the minimum benchmark for model comparison.
    """
    preprocessor = build_preprocessor(scale_numeric=False)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DummyClassifier(strategy="most_frequent", random_state=42))
    ])
    return pipeline


def get_logistic_regression_pipeline(
    C: float = 1.0,
    class_weight: Optional[str | Dict[int, float]] = "balanced",
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Construct Logistic Regression pipeline with StandardScaler and OneHotEncoder.
    Uses lbfgs solver suitable for L2 regularization.
    """
    preprocessor = build_preprocessor(scale_numeric=True)
    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def get_decision_tree_pipeline(
    max_depth: Optional[int] = 5,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    criterion: str = "gini",
    class_weight: Optional[str | Dict[int, float]] = "balanced",
    random_state: int = 42
) -> Pipeline:
    """
    Construct Decision Tree Classifier pipeline.
    Leaves numerical features unscaled to preserve natural split boundaries.
    """
    preprocessor = build_preprocessor(scale_numeric=False)
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def get_random_forest_pipeline(
    n_estimators: int = 150,
    max_depth: Optional[int] = 8,
    min_samples_split: int = 6,
    min_samples_leaf: int = 3,
    max_features: str = "sqrt",
    class_weight: Optional[str | Dict[int, float]] = "balanced",
    random_state: int = 42
) -> Pipeline:
    """
    Construct Random Forest Classifier pipeline.
    Aggregates randomized de-correlated trees for non-linear interactions.
    """
    preprocessor = build_preprocessor(scale_numeric=False)
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def get_xgboost_pipeline(
    n_estimators: int = 120,
    max_depth: int = 4,
    learning_rate: float = 0.08,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    min_child_weight: int = 2,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 4.37,
    random_state: int = 42
) -> Pipeline:
    """
    Construct XGBoost Classifier pipeline.
    Gradient-boosted decision tree ensemble optimized for binary logloss.
    """
    try:
        from xgboost import XGBClassifier
    except (ImportError, Exception) as exc:
        raise ImportError(
            f"Failed to import XGBoost: {exc}. Ensure xgboost library and OpenMP runtimes are installed."
        )

    preprocessor = build_preprocessor(scale_numeric=False)
    classifier = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def get_all_classifiers(scale_pos_weight: float = 4.37) -> Dict[str, Pipeline]:
    """
    Instantiate default pipelines for all four required classifiers plus baseline.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative to positive samples in training partition.

    Returns
    -------
    Dict[str, Pipeline]
        Mapping from model identifier to un-fitted scikit-learn Pipeline.
    """
    models: Dict[str, Pipeline] = {
        "Baseline": get_baseline_pipeline(),
        "Logistic Regression": get_logistic_regression_pipeline(),
        "Decision Tree": get_decision_tree_pipeline(),
        "Random Forest": get_random_forest_pipeline(),
    }
    try:
        models["XGBoost"] = get_xgboost_pipeline(scale_pos_weight=scale_pos_weight)
    except Exception as exc:
        logger.warning(f"XGBoost pipeline deferred: {exc}")
    return models


def get_optional_regression_pipeline() -> Pipeline:
    """
    Construct optional regression pipeline for continuous workforce target (MonthlyIncome).
    Uses RandomForestRegressor on operational career attributes.
    """
    num_cols = [
        "Age", "JobLevel", "TotalWorkingYears", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    cat_cols = ["Department", "JobRole", "EducationLevel"]
    preprocessor = build_preprocessor(numerical_cols=num_cols, categorical_cols=cat_cols, scale_numeric=False)
    regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
