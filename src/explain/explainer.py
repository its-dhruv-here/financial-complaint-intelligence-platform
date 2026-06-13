"""
Model explainability utilities.

Extracts per-class feature importances from linear models by inspecting the
learned coefficient matrix, producing human-readable keyword lists.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def generate_explainability(
    pipeline: Pipeline,
    top_n: int = 10,
) -> Dict[str, List[str]]:
    """
    Extract the top-``top_n`` positive keywords per class from a trained
    scikit-learn pipeline containing ``tfidf`` and ``clf`` steps.

    The classifier must expose a ``coef_`` attribute (e.g. Logistic Regression,
    LinearSVC).  If it does not, an empty dictionary is returned.

    Args:
        pipeline: A fitted ``Pipeline`` with named steps ``"tfidf"`` and
            ``"clf"``.
        top_n: Number of top keywords to extract per class.

    Returns:
        A mapping from class name to a list of keyword strings.
    """
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
    except KeyError:
        logger.warning("Pipeline missing 'tfidf' or 'clf' step — cannot generate explainability.")
        return {}

    if not hasattr(clf, "coef_"):
        logger.warning(
            "Model %s has no coef_ attribute — explainability unavailable.",
            clf.__class__.__name__,
        )
        return {}

    feature_names: np.ndarray = np.array(tfidf.get_feature_names_out())
    coefs: np.ndarray = clf.coef_
    explanations: Dict[str, List[str]] = {}

    if len(clf.classes_) == 2 and coefs.shape[0] == 1:
        # Binary classification edge-case
        top_pos = np.argsort(coefs[0])[-top_n:][::-1]
        top_neg = np.argsort(coefs[0])[:top_n]
        explanations[str(clf.classes_[1])] = [feature_names[j] for j in top_pos]
        explanations[str(clf.classes_[0])] = [feature_names[j] for j in top_neg]
    else:
        for i, class_name in enumerate(clf.classes_):
            top_indices = np.argsort(coefs[i])[-top_n:][::-1]
            explanations[str(class_name)] = [feature_names[j] for j in top_indices]

    logger.info("Explainability generation complete — %d classes.", len(explanations))
    return explanations
