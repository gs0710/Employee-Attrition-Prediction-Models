"""
Data validation and quality checks for employee attrition dataset.
"""

from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger("data_validation")


def validate_dataset(
    df: pd.DataFrame,
    target_col: str = "Attrition",
    id_col: str = "EmployeeID"
) -> Dict[str, Any]:
    """
    Perform comprehensive validation checks on employee dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.
    target_col : str
        Name of target column. Defaults to 'Attrition'.
    id_col : str
        Name of identifier column. Defaults to 'EmployeeID'.

    Returns
    -------
    Dict[str, Any]
        Dictionary with validation results and diagnostic metadata.
    """
    logger.info("Running dataset validation checks...")
    n_rows, n_cols = df.shape

    # 1. Missing values
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()

    # 2. Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # 3. Target column validation
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")

    target_counts = df[target_col].value_counts().to_dict()
    target_pcts = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
    valid_target_labels = {"Yes", "No", 1, 0, "1", "0"}
    actual_labels = set(df[target_col].unique())
    unexpected_target_labels = list(actual_labels - valid_target_labels)

    # 4. Constant or near-constant features (variance == 0 or single unique value)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    quasi_constant_cols = []
    for col in df.columns:
        if col not in constant_cols and col != id_col:
            top_freq_ratio = df[col].value_counts(normalize=True).iloc[0]
            if top_freq_ratio >= 0.99:
                quasi_constant_cols.append((col, round(float(top_freq_ratio), 4)))

    # 5. Identifier and leakage verification
    leakage_risks = []
    if id_col in df.columns:
        leakage_risks.append({
            "column": id_col,
            "type": "Administrative Identifier",
            "reason": "Direct memorization hazard. Must be dropped before model training."
        })

    # Summary report
    report: Dict[str, Any] = {
        "shape": (n_rows, n_cols),
        "total_records": n_rows,
        "total_columns": n_cols,
        "missing_values": cols_with_missing,
        "has_missing": len(cols_with_missing) > 0,
        "duplicate_rows": duplicate_rows,
        "target_column": target_col,
        "target_distribution_counts": target_counts,
        "target_distribution_pct": target_pcts,
        "unexpected_target_labels": unexpected_target_labels,
        "constant_columns": constant_cols,
        "quasi_constant_columns": quasi_constant_cols,
        "leakage_risks": leakage_risks,
        "is_valid": (
            len(unexpected_target_labels) == 0
            and duplicate_rows == 0
            and len(constant_cols) == 0
        ),
    }

    logger.info(
        f"Validation complete: {n_rows} rows, {n_cols} cols. "
        f"Duplicates: {duplicate_rows}, Missing columns: {len(cols_with_missing)}, "
        f"Target distribution: {target_counts}"
    )
    return report


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a clean summary table of data types, unique counts, and missingness."""
    summary_data = []
    for col in df.columns:
        summary_data.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Non-Null Count": int(df[col].notnull().sum()),
            "Null Count": int(df[col].isnull().sum()),
            "Unique Values": int(df[col].nunique()),
            "Sample Values": str(df[col].dropna().unique()[:3].tolist())
        })
    return pd.DataFrame(summary_data)
