"""
Model training utilities.

Provides TF-IDF feature-engineering experiments and the model registry
used for benchmarking (Dummy Classifier, Multinomial NB, Logistic Regression,
Linear SVM).
"""

import time
import logging
from typing import Any, Dict, List

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def run_feature_engineering_experiments(
    X_train: Any, y_train: Any
) -> List[Dict[str, Any]]:
    """
    Compare various TF-IDF vectorisation strategies.

    For each strategy the function records vocabulary size, matrix sparsity,
    fit-transform wall-time, and inference wall-time on a 1 000-sample slice.

    Returns:
        A list of result dictionaries, one per experiment.
    """
    logger.info("Running feature-engineering experiments ...")

    experiments: Dict[str, TfidfVectorizer] = {
        "A. TF-IDF Unigrams": TfidfVectorizer(ngram_range=(1, 1)),
        "B. TF-IDF Uni+Bigrams": TfidfVectorizer(ngram_range=(1, 2)),
        "C. TF-IDF max-10k features": TfidfVectorizer(max_features=10_000),
        "D. TF-IDF + stopword removal": TfidfVectorizer(stop_words="english"),
        "E. TF-IDF + sublinear TF": TfidfVectorizer(sublinear_tf=True),
    }

    results: List[Dict[str, Any]] = []

    for name, vec in experiments.items():
        t0 = time.time()
        X_vec = vec.fit_transform(X_train)
        t_fit = time.time() - t0

        t0 = time.time()
        _ = vec.transform(X_train[:1_000])
        t_inf = time.time() - t0

        vocab_size: int = X_vec.shape[1]
        sparsity: float = 1.0 - (X_vec.nnz / (X_vec.shape[0] * X_vec.shape[1]))

        results.append(
            {
                "Experiment": name,
                "Vocabulary Size": vocab_size,
                "Sparsity": sparsity,
                "Fit Time (s)": round(t_fit, 2),
                "Inference Time (s)": round(t_inf, 3),
            }
        )
        logger.info(
            "  %s — vocab=%d  sparsity=%.4f  fit=%.2fs",
            name, vocab_size, sparsity, t_fit,
        )

    return results


def get_model_benchmarks() -> Dict[str, Any]:
    """
    Return an ordered dictionary of models for benchmarking.

    Includes a *Dummy Classifier* (most-frequent strategy) as the random
    baseline, followed by Multinomial NB, Logistic Regression, and Linear SVM.
    """
    return {
        "Dummy (Most-Frequent)": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1_000, random_state=42, n_jobs=1,
            class_weight="balanced",
        ),
        "Linear SVM": LinearSVC(
            random_state=42, dual=False,
            class_weight="balanced",
        ),
    }
