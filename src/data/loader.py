"""
Data ingestion and train/test splitting utilities.

Handles CSV loading, data validation, optional stratified sub-sampling for
development speed, and reproducible train/test partitioning.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Optional

from src.data.validator import validate_and_clean_data

logger = logging.getLogger(__name__)


def load_and_split_data(
    filepath: str,
    text_col: str = "Consumer complaint narrative",
    target_col: str = "Product",
    sample_size: Optional[int] = 400_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, int]:
    """
    Load a CSV dataset, clean it, optionally sample, and split.

    Args:
        filepath: Path to the raw CSV file.
        text_col: Column name containing complaint narratives.
        target_col: Column name containing classification labels.
        sample_size: If set and smaller than the dataset, a stratified
            sub-sample of this size is drawn for faster iteration.
        test_size: Proportion of data reserved for the test set.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple of ``(X_train, X_test, y_train, y_test, y_full, original_size)`` where
        ``y_full`` is the target column of the (possibly sampled) dataset
        before splitting, useful for class-distribution plotting, and ``original_size``
        is the row count before any sampling.
    """
    logger.info("Loading data from %s ...", filepath)
    df: pd.DataFrame = pd.read_csv(filepath, low_memory=False)

    df = validate_and_clean_data(df, text_col, target_col)

    original_size = len(df)

    # Stratified sub-sampling for development speed
    if sample_size and sample_size < len(df):
        logger.info("Sampling %d rows from %d total ...", sample_size, len(df))
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[target_col],
            random_state=random_state,
        )

    X: pd.Series = df[text_col].astype(str)
    y: pd.Series = df[target_col].astype(str)

    logger.info("Splitting data with test_size=%.2f ...", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, df[target_col], original_size

