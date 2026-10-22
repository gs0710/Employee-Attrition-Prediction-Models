"""
Data loading utilities for employee attrition dataset.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from src.utils import get_project_root, setup_logger

logger = setup_logger("data_loader")


def load_raw_data(data_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load raw employee dataset from CSV file.

    Parameters
    ----------
    data_path : Optional[str | Path]
        Path to raw CSV file. Defaults to 'data/raw/employee_data.csv'.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame containing the raw employee records.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or corrupted.
    """
    if data_path is None:
        resolved_path = get_project_root() / "data" / "raw" / "employee_data.csv"
    else:
        resolved_path = Path(data_path)
        if not resolved_path.is_absolute():
            resolved_path = get_project_root() / resolved_path

    if not resolved_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {resolved_path}")

    logger.info(f"Loading raw dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path)

    if df.empty:
        raise ValueError(f"Loaded dataset from {resolved_path} is empty.")

    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    return df
