"""
Hyperparameter optimization module using Stratified Cross-Validation.
Supports systematic tuning for Logistic Regression, Decision Tree, Random Forest, and XGBoost.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from src.models import (
    get_logistic_regression_pipeline,
    get_decision_tree_pipeline,
    get_random_forest_pipeline,
    get_xgboost_pipeline,
)
from src.utils import setup_logger

logger = setup_logger("tuning")


def get_param_grids() -> Dict[str, Dict[str, Any]]:
    """Return configured parameter search spaces for each classifier."""
    return {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "classifier__class_weight": ["balanced", None],
        },
        "Decision Tree": {
            "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
            "classifier__min_samples_split": [5, 10, 15, 20],
            "classifier__min_samples_leaf": [2, 4, 6, 10],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__class_weight": ["balanced", None],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [5, 8, 12, 16, None],
            "classifier__min_samples_split": [4, 8, 12],
            "classifier__min_samples_leaf": [2, 4, 6],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": ["balanced", "balanced_subsample", None],
        },
        "XGBoost": {
            "classifier__n_estimators": [80, 120, 180],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.03, 0.08, 0.15],
            "classifier__subsample": [0.75, 0.85, 1.0],
            "classifier__colsample_bytree": [0.75, 0.85, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__scale_pos_weight": [1.0, 4.37],
        },
    }


def tune_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_type: str = "randomized",
    n_iter: int = 15,
    cv_folds: int = 5,
    scoring: str = "f1",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Tune a classification pipeline using Stratified K-Fold Cross-Validation.

    Parameters
    ----------
    model_name : str
        Name identifier of the classifier.
    pipeline : Pipeline
        Base pipeline before tuning.
    param_grid : Dict[str, Any]
        Hyperparameter space.
    X_train : pd.DataFrame
        Training predictor features.
    y_train : pd.Series
        Training binary target vector.
    search_type : str
        'grid' for exhaustive grid search, 'randomized' for budget-bounded random search.
    n_iter : int
        Number of parameter settings sampled if using RandomizedSearchCV.
    cv_folds : int
        Number of stratified CV splits.
    scoring : str
        Primary metric for parameter selection (default: 'f1').
    random_state : int
        Seed for deterministic reproducibility.

    Returns
    -------
    Dict[str, Any]
        Tuning outcomes including best_estimator, best_params, best_score, cv_results, duration.
    """
    logger.info(f"Starting {search_type} tuning for [{model_name}] on {cv_folds} folds (metric: {scoring})...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    start_time = time.perf_counter()
    if search_type == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )
    else:
        # Check if total combinations is smaller than n_iter
        from itertools import product
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        actual_iter = min(n_iter, total_combinations)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=actual_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
            return_train_score=True
        )

    search.fit(X_train, y_train)
    duration = time.perf_counter() - start_time

    clean_best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    logger.info(f"[{model_name}] Tuning completed in {duration:.2f}s | Best {scoring}: {search.best_score_:.4f}")
    logger.info(f"[{model_name}] Optimal parameters: {clean_best_params}")

    return {
        "model_name": model_name,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "clean_best_params": clean_best_params,
        "best_cv_score": float(search.best_score_),
        "scoring_metric": scoring,
        "cv_results": pd.DataFrame(search.cv_results_),
        "tuning_duration_seconds": round(duration, 3),
        "cv_folds": cv_folds,
        "search_type": search_type,
    }


def tune_all_models(
    base_pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_type: str = "randomized",
    n_iter: int = 15,
    cv_folds: int = 5,
    scoring: str = "f1",
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """Tune all available classification models sequentially."""
    param_grids = get_param_grids()
    tuning_results: Dict[str, Dict[str, Any]] = {}

    for name, pipeline in base_pipelines.items():
        if name == "Baseline":
            continue
        if name not in param_grids:
            logger.warning(f"No param grid configured for {name}. Skipping tuning.")
            continue
        tuning_results[name] = tune_model(
            model_name=name,
            pipeline=pipeline,
            param_grid=param_grids[name],
            X_train=X_train,
            y_train=y_train,
            search_type=search_type,
            n_iter=n_iter,
            cv_folds=cv_folds,
            scoring=scoring,
            random_state=random_state
        )
    return tuning_results
