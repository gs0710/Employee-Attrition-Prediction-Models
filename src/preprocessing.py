"""
Feature engineering, train-test splitting, and leakage-safe preprocessing pipelines.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import get_project_root, setup_logger

logger = setup_logger("preprocessing")

# Feature classification
DEFAULT_NUMERICAL_COLS: List[str] = [
    "Age",
    "JobLevel",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "PerformanceRating",
    "AbsenteeismDays",
]

DEFAULT_CATEGORICAL_COLS: List[str] = [
    "Department",
    "JobRole",
    "Gender",
    "EducationLevel",
    "Overtime",
    "BusinessTravel",
]

DEFAULT_EXCLUDED_COLS: List[str] = ["EmployeeID"]


def map_target_to_binary(target_series: pd.Series) -> pd.Series:
    """
    Convert target series to binary numeric format: 'Yes' -> 1, 'No' -> 0.

    Parameters
    ----------
    target_series : pd.Series
        Original target values.

    Returns
    -------
    pd.Series
        Integer binary series (1 for Yes, 0 for No).
    """
    if target_series.dtype == "int64" or target_series.dtype == "int32":
        return target_series

    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
    mapped = target_series.map(mapping)
    if mapped.isnull().any():
        unknown_vals = target_series[mapped.isnull()].unique()
        raise ValueError(f"Encountered unexpected target values during binary mapping: {unknown_vals}")
    return mapped.astype(int)


def split_and_save_data(
    df: pd.DataFrame,
    target_col: str = "Attrition",
    excluded_cols: Optional[List[str]] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train-test split and optionally export processed sets.

    Parameters
    ----------
    df : pd.DataFrame
        Complete raw dataframe.
    target_col : str
        Target column name.
    excluded_cols : Optional[List[str]]
        Columns to drop prior to modeling (e.g. EmployeeID).
    test_size : float
        Fraction of data reserved for held-out testing (default 0.20).
    random_state : int
        Seed for deterministic split.
    save_processed : bool
        Whether to save CSV files to data/processed/.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if excluded_cols is None:
        excluded_cols = DEFAULT_EXCLUDED_COLS

    cols_to_drop = [c for c in excluded_cols if c in df.columns] + [target_col]
    X = df.drop(columns=cols_to_drop)
    y = map_target_to_binary(df[target_col])

    logger.info(f"Splitting data: {len(df)} records into train ({(1-test_size)*100:.0f}%) and test ({test_size*100:.0f}%)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info(f"Train set: {X_train.shape[0]} samples (Attrition rate: {y_train.mean():.4f})")
    logger.info(f"Test set:  {X_test.shape[0]} samples (Attrition rate: {y_test.mean():.4f})")

    if save_processed:
        processed_dir = get_project_root() / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(processed_dir / "train_features.csv", index=False)
        X_test.to_csv(processed_dir / "test_features.csv", index=False)
        y_train.to_frame(name=target_col).to_csv(processed_dir / "train_target.csv", index=False)
        y_test.to_frame(name=target_col).to_csv(processed_dir / "test_target.csv", index=False)
        logger.info(f"Saved processed partitions to {processed_dir}")

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for preprocessing.

    Parameters
    ----------
    numerical_cols : Optional[List[str]]
        List of numeric column names.
    categorical_cols : Optional[List[str]]
        List of categorical column names.
    scale_numeric : bool
        If True, apply StandardScaler (essential for Logistic Regression).
        If False, passthrough numericals (ideal for Decision Tree, Random Forest, XGBoost).

    Returns
    -------
    ColumnTransformer
        Configured preprocessor with handle_unknown='ignore' on OneHotEncoder.
    """
    if numerical_cols is None:
        numerical_cols = DEFAULT_NUMERICAL_COLS
    if categorical_cols is None:
        categorical_cols = DEFAULT_CATEGORICAL_COLS

    if scale_numeric:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    feature_names_in: List[str]
) -> List[str]:
    """Retrieve output feature names following ColumnTransformer fitting."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_cols = preprocessor.transformers[1][2]
        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        num_cols = list(preprocessor.transformers[0][2])
        return num_cols + cat_names
