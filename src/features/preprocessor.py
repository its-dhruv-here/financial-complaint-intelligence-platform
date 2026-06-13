"""
Custom scikit-learn text preprocessing transformer.

Provides a ``TextPreprocessor`` class that integrates into scikit-learn
``Pipeline`` objects and safely handles ``str``, ``list``, ``np.ndarray``,
and ``pd.Series`` inputs during both training and inference.
"""

import re
from typing import Any, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import logging

logger = logging.getLogger(__name__)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Configurable NLP preprocessing transformer compatible with scikit-learn.

    Supports:
        - Lowercasing
        - Special-character removal
        - Whitespace normalisation
        - Optional English stop-word removal

    Parameters:
        remove_stopwords: If ``True``, remove English stop-words after
            tokenisation.
        lowercase: If ``True``, convert all text to lowercase.
        remove_special_chars: If ``True``, strip non-alphanumeric characters.
    """

    def __init__(
        self,
        remove_stopwords: bool = False,
        lowercase: bool = True,
        remove_special_chars: bool = True,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars

    def fit(self, X: Any, y: Any = None) -> "TextPreprocessor":
        """No-op fit — preprocessing is stateless."""
        return self

    def _clean_text(self, text: Any) -> str:
        """Apply cleaning rules to a single text string."""
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_special_chars:
            text = re.sub(r"[^\w\s]", " ", text)

        tokens: List[str] = text.split()

        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]

        return " ".join(tokens)

    def transform(
        self, X: Union[str, list, np.ndarray, pd.Series], y: Any = None
    ) -> Union[List[str], np.ndarray, pd.Series]:
        """
        Clean input text(s).

        Accepts ``str``, ``list[str]``, ``np.ndarray``, or ``pd.Series``
        and returns the same container type (except a bare ``str`` is
        returned as a single-element ``list``).
        """
        if isinstance(X, str):
            return [self._clean_text(X)]
        elif isinstance(X, list):
            return [self._clean_text(t) for t in X]
        elif isinstance(X, np.ndarray):
            return np.array([self._clean_text(t) for t in X])
        elif isinstance(X, pd.Series):
            return X.apply(self._clean_text)
        else:
            return [self._clean_text(str(t)) for t in X]
