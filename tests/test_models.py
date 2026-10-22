"""
Unit tests for model pipeline instantiation, fitting, and probability outputs.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import split_and_save_data
from src.models import (
    get_baseline_pipeline,
    get_logistic_regression_pipeline,
    get_decision_tree_pipeline,
    get_random_forest_pipeline,
    get_xgboost_pipeline,
    get_all_classifiers,
    get_optional_regression_pipeline,
)


@pytest.fixture(scope="module")
def data_splits():
    df = load_raw_data()
    return split_and_save_data(df, test_size=0.2, random_state=42, save_processed=False)


def test_baseline_classifier(data_splits):
    X_train, X_test, y_train, y_test = data_splits
    pipeline = get_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds) == {0}  # Predicts majority class


def test_all_four_classifiers_fitting_and_probabilities(data_splits):
    X_train, X_test, y_train, y_test = data_splits
    classifiers = get_all_classifiers()

    expected_models = {"Baseline", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"}
    assert set(classifiers.keys()) == expected_models

    for name, pipeline in classifiers.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        assert len(preds) == len(y_test), f"Predictions length mismatch for {name}"

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)
            assert probs.shape == (len(y_test), 2), f"Probability shape mismatch for {name}"
            # Probabilities must sum to 1
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_optional_regression_pipeline(data_splits):
    X_train, X_test, y_train, y_test = data_splits
    reg_pipeline = get_optional_regression_pipeline()

    # Target is continuous MonthlyIncome
    y_reg_train = X_train["MonthlyIncome"]
    X_reg_train = X_train.drop(columns=["MonthlyIncome"])
    y_reg_test = X_test["MonthlyIncome"]
    X_reg_test = X_test.drop(columns=["MonthlyIncome"])

    reg_pipeline.fit(X_reg_train, y_reg_train)
    preds = reg_pipeline.predict(X_reg_test)
    assert len(preds) == len(y_reg_test)
    assert not np.isnan(preds).any()
