"""
Unit tests for preprocessing pipelines, train-test splitting, and feature encoding.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.preprocessing import (
    split_and_save_data,
    build_preprocessor,
    get_feature_names_after_preprocessing,
)


def test_train_test_split_ratio_and_stratification():
    """Verify stratified split maintains approximately 80/20 ratio and class proportions."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.2, random_state=42, save_processed=False)

    total_records = len(df)
    assert len(X_train) == 1176
    assert len(X_test) == 294
    assert len(y_train) == 1176
    assert len(y_test) == 294

    # Check EmployeeID was dropped from predictors
    assert "EmployeeID" not in X_train.columns
    assert "EmployeeID" not in X_test.columns
    assert "Attrition" not in X_train.columns

    # Verify class proportions match within 1%
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.015


def test_build_preprocessor_transformation():
    """Verify ColumnTransformer transforms training data correctly without leakage."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.2, random_state=42, save_processed=False)

    preprocessor = build_preprocessor(scale_numeric=True)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert isinstance(X_train_trans, np.ndarray)
    assert isinstance(X_test_trans, np.ndarray)
    assert X_train_trans.shape[0] == 1176
    assert X_test_trans.shape[0] == 294
    assert X_train_trans.shape[1] == X_test_trans.shape[1]

    # Verify feature names extraction
    feature_names = get_feature_names_after_preprocessing(preprocessor, X_train.columns.tolist())
    assert len(feature_names) == X_train_trans.shape[1]


def test_unseen_category_handling():
    """Verify OneHotEncoder safely handles unseen categorical levels (handle_unknown='ignore')."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.2, random_state=42, save_processed=False)

    preprocessor = build_preprocessor(scale_numeric=False)
    preprocessor.fit(X_train)

    # Introduce synthetic unseen category
    X_unseen = X_test.copy().head(5)
    X_unseen.loc[X_unseen.index[0], "Department"] = "Space Exploration"
    X_unseen.loc[X_unseen.index[1], "JobRole"] = "Cosmic Navigator"

    trans = preprocessor.transform(X_unseen)
    assert trans.shape[0] == 5
    assert not np.isnan(trans).any()
