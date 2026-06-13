"""
Data validation and cleaning utilities for the CFPB complaint dataset.

This module provides robust data quality functions that handle missing values,
empty text fields, and rare class filtering to produce a clean training corpus.
"""

import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


def validate_and_clean_data(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    min_class_samples: int = 100
) -> pd.DataFrame:
    """
    Validate the dataset and remove invalid or unusable rows.

    Steps performed:
        1. Drop rows where target or text column is missing (NaN).
        2. Drop rows where text is empty after whitespace stripping.
        3. Drop rows belonging to classes with fewer than ``min_class_samples``.

    Args:
        df: Raw input DataFrame.
        text_col: Name of the column containing complaint text.
        target_col: Name of the column containing the classification label.
        min_class_samples: Minimum number of samples required for a class
            to be retained.  Classes below this threshold are dropped.

    Returns:
        A cleaned copy of the DataFrame.
    """
    initial_rows: int = len(df)

    # 1. Drop rows with missing text or target
    df = df.dropna(subset=[text_col, target_col]).copy()
    logger.info(
        "Dropped %d rows with missing '%s' or '%s'.",
        initial_rows - len(df), text_col, target_col,
    )

    # 2. Remove rows where text is effectively empty
    initial_rows = len(df)
    df["_text_len"] = df[text_col].astype(str).str.strip().str.len()
    df = df[df["_text_len"] > 0].copy()
    df = df.drop(columns=["_text_len"])
    logger.info("Dropped %d rows with empty text.", initial_rows - len(df))

    # 3. Filter out rare classes
    class_counts: pd.Series = df[target_col].value_counts()
    valid_classes: List[str] = class_counts[class_counts >= min_class_samples].index.tolist()
    initial_rows = len(df)
    df = df[df[target_col].isin(valid_classes)]
    logger.info(
        "Dropped %d rows belonging to rare classes (<%d samples).",
        initial_rows - len(df), min_class_samples,
    )

    logger.info("Cleaned dataset shape: %s", df.shape)
    return df
