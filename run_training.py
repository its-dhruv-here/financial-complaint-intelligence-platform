"""
run_training.py — End-to-end ML training orchestration script.

Executes the full pipeline:
    1. Data loading, validation, and stratified splitting.
    2. Text preprocessing.
    3. TF-IDF feature-engineering experiments.
    4. Model benchmarking (Dummy, NB, LogReg, SVM).
    5. Best-model selection by Weighted F1.
    6. Pipeline serialisation.
    7. Explainability report generation.
    8. Export of all reporting assets (metrics.json, benchmark_results.csv,
       figures/).

Usage:
    python run_training.py
"""

import json
import os
import time
import logging
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.loader import load_and_split_data
from src.features.preprocessor import TextPreprocessor
from src.models.train import run_feature_engineering_experiments, get_model_benchmarks
from src.models.evaluate import evaluate_predictions
from src.explain.explainer import generate_explainability

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = "rows.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_class_distribution(y: pd.Series) -> None:
    """Save a horizontal bar chart of class frequencies."""
    fig, ax = plt.subplots(figsize=(10, 8))
    counts = y.value_counts()
    ax.barh(counts.index, counts.values, color="#3574a7")
    ax.set_xlabel("Number of Complaints")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)


def _plot_benchmark_comparison(df_eval: pd.DataFrame) -> None:
    """Save a grouped bar chart comparing Accuracy and Weighted F1 across models."""
    df_melted = df_eval.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Weighted F1"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Benchmarking Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "benchmark_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_benchmark_all_metrics(df_eval: pd.DataFrame) -> None:
    """Save a grouped bar chart comparing all metrics across models."""
    metric_cols = ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]
    df_melted = df_eval.melt(
        id_vars="Model",
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Full Metric Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "benchmark_all_metrics.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_dirs()
    logger.info("=" * 60)
    logger.info("Starting ML Training Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & split data
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, y_full, original_size = load_and_split_data(
        filepath=DATA_PATH, sample_size=400_000
    )
    _plot_class_distribution(y_full)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    logger.info("Applying TextPreprocessor ...")
    preprocessor = TextPreprocessor(
        remove_stopwords=False, lowercase=True, remove_special_chars=True
    )
    X_train_clean = preprocessor.transform(X_train)
    X_test_clean = preprocessor.transform(X_test)

    # ------------------------------------------------------------------
    # 3. Feature engineering experiments
    # ------------------------------------------------------------------
    feat_results: List[Dict[str, Any]] = run_feature_engineering_experiments(
        X_train_clean, y_train
    )
    with open(os.path.join(REPORTS_DIR, "feature_engineering_report.md"), "w") as f:
        f.write("# Feature Engineering Experiments\n\n")
        f.write("| Experiment | Vocab Size | Sparsity | Fit Time (s) | Inference Time (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in feat_results:
            f.write(
                f"| {r['Experiment']} | {r['Vocabulary Size']:,} "
                f"| {r['Sparsity']:.4%} | {r['Fit Time (s)']} "
                f"| {r['Inference Time (s)']} |\n"
            )

    # ------------------------------------------------------------------
    # 4. Model benchmarking
    # ------------------------------------------------------------------
    logger.info("Running model benchmarking ...")
    tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_clean)
    X_test_vec = tfidf.transform(X_test_clean)

    models = get_model_benchmarks()
    eval_results: List[Dict[str, Any]] = []

    best_f1: float = 0.0
    best_model_name: str = ""
    best_clf: Any = None

    labels: List[str] = sorted(y_train.unique().tolist())

    for name, clf in models.items():
        logger.info("Training %s ...", name)
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test_vec)
        t_inf = time.time() - t0

        metrics, clf_report, _ = evaluate_predictions(
            y_test, y_pred, labels, name, FIGURES_DIR
        )
        metrics["Train Time (s)"] = round(t_train, 2)
        metrics["Inference Time (s)"] = round(t_inf, 3)
        eval_results.append(metrics)

        if metrics["Weighted F1"] > best_f1:
            best_f1 = metrics["Weighted F1"]
            best_model_name = name
            best_clf = clf

    logger.info(
        "Best model: %s  (Weighted F1 = %.4f)", best_model_name, best_f1
    )

    # ------------------------------------------------------------------
    # 5. Export benchmark assets
    # ------------------------------------------------------------------
    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(os.path.join(REPORTS_DIR, "benchmark_results.csv"), index=False)

    _plot_benchmark_comparison(df_eval)
    _plot_benchmark_all_metrics(df_eval)

    # Export production metrics JSON
    best_metrics = next(m for m in eval_results if m["Model"] == best_model_name)
    best_metrics["Train Samples"] = len(X_train)
    best_metrics["Test Samples"] = len(X_test)
    best_metrics["Total Rows"] = len(y_full)
    best_metrics["Classes"] = len(labels)
    best_metrics["Vocabulary Size"] = tfidf.max_features or X_train_vec.shape[1]
    best_metrics["Avg Complaint Length"] = int(
        pd.Series(X_train_clean).str.len().mean()
    )

    top_categories = y_train.value_counts().head(3).index.tolist()

    # Store the full benchmark table in metrics.json too
    all_benchmarks = []
    for m in eval_results:
        all_benchmarks.append({
            "Model": m["Model"],
            "Accuracy": round(m["Accuracy"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "Macro F1": round(m["Macro F1"], 4),
            "Weighted F1": round(m["Weighted F1"], 4),
            "Train Time (s)": m["Train Time (s)"],
            "Inference Time (s)": m["Inference Time (s)"],
        })

    metrics_export = {
        "best_model": {
            "Model": best_metrics["Model"],
            "Accuracy": round(best_metrics["Accuracy"], 4),
            "Precision": round(best_metrics["Precision"], 4),
            "Recall": round(best_metrics["Recall"], 4),
            "Macro F1": round(best_metrics["Macro F1"], 4),
            "Weighted F1": round(best_metrics["Weighted F1"], 4),
        },
        "dataset": {
            "Original Rows": original_size,
            "Total Rows": best_metrics["Total Rows"],
            "Train Samples": best_metrics["Train Samples"],
            "Test Samples": best_metrics["Test Samples"],
            "Classes": best_metrics["Classes"],
            "Vocabulary Size": best_metrics["Vocabulary Size"],
            "Avg Complaint Length": best_metrics["Avg Complaint Length"],
            "Top Categories": top_categories,
        },
        "benchmarks": all_benchmarks,
    }

    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_export, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Hyperparameter tuning on the best model
    # ------------------------------------------------------------------
    from sklearn.model_selection import GridSearchCV

    logger.info("Running hyperparameter tuning on %s ...", best_model_name)

    param_grid = {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
    grid_search = GridSearchCV(
        LogisticRegression(
            max_iter=1_000, random_state=42, n_jobs=1,
            class_weight="balanced",
        ),
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=1,
        verbose=0,
    )
    grid_search.fit(X_train_vec, y_train)
    logger.info(
        "Best C=%.2f  (CV Weighted-F1=%.4f)",
        grid_search.best_params_["C"],
        grid_search.best_score_,
    )

    # Re-evaluate the tuned model on the held-out test set
    tuned_clf = grid_search.best_estimator_
    y_pred_tuned = tuned_clf.predict(X_test_vec)
    tuned_metrics, tuned_report, _ = evaluate_predictions(
        y_test, y_pred_tuned, labels, f"{best_model_name} (Tuned)", FIGURES_DIR
    )
    logger.info(
        "Tuned test-set Weighted-F1=%.4f  (was %.4f)",
        tuned_metrics["Weighted F1"], best_f1,
    )

    # Use tuned model if it improved
    if tuned_metrics["Weighted F1"] >= best_f1:
        best_clf = tuned_clf
        best_f1 = tuned_metrics["Weighted F1"]
        best_model_name_display = f"{best_model_name} (Tuned C={grid_search.best_params_['C']})"
        logger.info("Tuned model adopted.")
    else:
        best_model_name_display = best_model_name
        logger.info("Tuned model did not improve — keeping original.")

    # Update metrics export with tuning info
    metrics_export["tuning"] = {
        "param_grid": param_grid,
        "best_C": grid_search.best_params_["C"],
        "cv_weighted_f1": round(grid_search.best_score_, 4),
        "test_weighted_f1_before": round(best_metrics["Weighted F1"], 4),
        "test_weighted_f1_after": round(tuned_metrics["Weighted F1"], 4),
    }
    # Update best_model section with tuned values
    metrics_export["best_model"]["Accuracy"] = round(tuned_metrics["Accuracy"], 4)
    metrics_export["best_model"]["Precision"] = round(tuned_metrics["Precision"], 4)
    metrics_export["best_model"]["Recall"] = round(tuned_metrics["Recall"], 4)
    metrics_export["best_model"]["Macro F1"] = round(tuned_metrics["Macro F1"], 4)
    metrics_export["best_model"]["Weighted F1"] = round(tuned_metrics["Weighted F1"], 4)
    metrics_export["best_model"]["Model"] = best_model_name_display

    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_export, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Serialize final pipeline
    # ------------------------------------------------------------------
    logger.info("Serializing final pipeline ...")
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("tfidf", tfidf),
        ("clf", best_clf),
    ])
    artifact_path = os.path.join(MODELS_DIR, "best_model_pipeline.pkl")
    joblib.dump(final_pipeline, artifact_path)
    logger.info("Pipeline saved to %s", artifact_path)

    # ------------------------------------------------------------------
    # 7. Explainability
    # ------------------------------------------------------------------
    explanations = generate_explainability(final_pipeline, top_n=10)
    with open(os.path.join(REPORTS_DIR, "explainability_report.md"), "w") as f:
        f.write("# Model Explainability Report\n\n")
        f.write(f"**Production Model:** {best_model_name}\n\n")
        f.write("Top 10 positive keywords driving classification per class:\n\n")
        for class_name, words in explanations.items():
            f.write(f"- **{class_name}**: {', '.join(words)}\n")

    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
