"""
Unit tests for evaluation metrics, cross-validation, threshold tuning, and artifact serialization.
"""

import pytest
import os
import tempfile
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import split_and_save_data
from src.models import get_logistic_regression_pipeline
from src.evaluation import evaluate_model_on_test, find_optimal_f1_threshold
from src.utils import save_joblib_artifact, load_joblib_artifact


@pytest.fixture(scope="module")
def sample_data():
    df = load_raw_data()
    return split_and_save_data(df, test_size=0.2, random_state=42, save_processed=False)


def test_evaluate_model_metrics(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    pipeline = get_logistic_regression_pipeline()
    res = evaluate_model_on_test(pipeline, X_train, y_train, X_test, y_test, "Logistic Regression")

    assert "accuracy" in res
    assert "precision" in res
    assert "recall" in res
    assert "f1_score" in res
    assert "roc_auc" in res
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert res["true_positives"] + res["false_positives"] + res["true_negatives"] + res["false_negatives"] == len(y_test)


def test_threshold_tuning(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    pipeline = get_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)

    opt_thresh, opt_f1 = find_optimal_f1_threshold(pipeline, X_train, y_train, cv_folds=3)
    assert 0.1 <= opt_thresh <= 0.9
    assert 0.0 < opt_f1 <= 1.0


def test_model_serialization_and_loading(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    pipeline = get_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.joblib")
        save_joblib_artifact(pipeline, model_path)
        assert os.path.exists(model_path)

        loaded_pipeline = load_joblib_artifact(model_path)
        orig_preds = pipeline.predict(X_test)
        loaded_preds = loaded_pipeline.predict(X_test)
        assert (orig_preds == loaded_preds).all()
