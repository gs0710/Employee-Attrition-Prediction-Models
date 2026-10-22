"""
Unit tests for data loading, data validation, and leakage detection.
"""

import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.data_validation import validate_dataset, get_column_summary
from src.preprocessing import map_target_to_binary


def test_load_raw_data():
    """Verify raw dataset can be loaded, is non-empty, and has expected shape."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == 1470
    assert df.shape[1] == 22
    assert "Attrition" in df.columns
    assert "EmployeeID" in df.columns


def test_validate_dataset():
    """Verify data validation detects zero nulls, zero duplicates, and valid target."""
    df = load_raw_data()
    report = validate_dataset(df)
    assert report["is_valid"] is True
    assert report["duplicate_rows"] == 0
    assert report["has_missing"] is False
    assert len(report["unexpected_target_labels"]) == 0
    assert len(report["leakage_risks"]) == 1
    assert report["leakage_risks"][0]["column"] == "EmployeeID"


def test_map_target_to_binary():
    """Verify binary target mapping converts 'Yes' to 1 and 'No' to 0."""
    sample_target = pd.Series(["Yes", "No", "No", "Yes", "No"])
    binary = map_target_to_binary(sample_target)
    assert list(binary) == [1, 0, 0, 1, 0]
    assert binary.dtype == "int32" or binary.dtype == "int64"


def test_map_target_unexpected_raises():
    """Verify invalid target values raise ValueError."""
    sample_target = pd.Series(["Yes", "Unknown", "No"])
    with pytest.raises(ValueError):
        map_target_to_binary(sample_target)


def test_column_summary():
    """Verify column summary table is generated correctly."""
    df = load_raw_data()
    summary_df = get_column_summary(df)
    assert len(summary_df) == 22
    assert set(summary_df.columns) == {"Column", "Dtype", "Non-Null Count", "Null Count", "Unique Values", "Sample Values"}
