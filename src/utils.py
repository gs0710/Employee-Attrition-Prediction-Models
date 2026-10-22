"""
Utility functions for reproducibility, logging, timing, and artifact persistence.
"""

from __future__ import annotations
import os
import random
import logging
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np
import joblib


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42) -> None:
    """Set random seed across standard libraries for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str = "attrition_ml", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a standardized console and file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


class ExecutionTimer:
    """Context manager and timer to record execution duration in seconds."""

    def __init__(self, description: str = "Operation") -> None:
        self.description = description
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> "ExecutionTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


def save_joblib_artifact(artifact: Any, relative_or_abs_path: str | Path) -> Path:
    """Save a Python object via joblib and return the resolved Path."""
    path = Path(relative_or_abs_path)
    if not path.is_absolute():
        path = get_project_root() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_joblib_artifact(relative_or_abs_path: str | Path) -> Any:
    """Load a Python object from a joblib file."""
    path = Path(relative_or_abs_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at path: {path}")
    return joblib.load(path)
