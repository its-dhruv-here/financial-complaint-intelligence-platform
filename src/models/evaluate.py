"""
Model evaluation utilities.

Computes classification metrics, generates per-model confusion-matrix plots,
and returns structured results for downstream reporting.
"""

import os
import logging
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    labels: List[str],
    model_name: str,
    figures_dir: str,
) -> Tuple[Dict[str, Any], str, str]:
    """
    Evaluate model predictions and persist a confusion-matrix figure.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of unique label strings.
        model_name: Human-readable model name (used in plot titles and
            file names).
        figures_dir: Directory in which to save the confusion-matrix PNG.

    Returns:
        A tuple of ``(metrics_dict, classification_report_str, cm_png_path)``.
    """
    metrics: Dict[str, Any] = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    clf_report: str = classification_report(y_true, y_pred, zero_division=0)

    # Confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()

    safe_name: str = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    cm_path: str = os.path.join(figures_dir, f"cm_{safe_name}.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    logger.info(
        "Evaluation complete for %s — Accuracy=%.4f  Weighted-F1=%.4f",
        model_name, metrics["Accuracy"], metrics["Weighted F1"],
    )
    return metrics, clf_report, cm_path
