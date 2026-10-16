"""
Evaluation module for model performance diagnostics.

Calculates Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix.
Ensures strict reproducibility by utilizing persisted split indices and provenance hashes.
Saves metrics JSON and visualization plots to outputs/.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    DEFAULT_CONFUSION_MATRIX_FILE,
    DEFAULT_METRICS_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_RAW_DATA_FILE,
    DEFAULT_SPLIT_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    ensure_directories,
)
from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.predictor import get_positive_class_index
from src.preprocessing import (
    clean_dataset,
    split_features_and_target,
    validate_split_metadata,
)
from src.utils import get_file_hash, get_timestamp, load_json, save_json, setup_logger

logger = setup_logger("evaluate_model")


def evaluate_model(
    model_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    split_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    plot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate the serialized model pipeline on the reproducible held-out test set.

    Evaluation Methodology:
    - Held-out Partition: 20% test partition (2,000 samples) verified against persisted indices.
    - Dataset Provenance: Validates SHA-256 hash of dataset against training split hash.
    - Positive Class: 1 (Machine Failure).
    - Averaging Strategy: 'binary' (strict focus on positive minority class detection).
    - Class Imbalance Consideration: Accuracy is uninformative (naive majority achieves 96.6%).
      Primary decision metrics are Recall, Precision, and PR-AUC.

    Args:
        model_path: Path to serialized model pipeline.
        dataset_path: Path to dataset CSV.
        split_path: Path to persisted train_test_split.json indices.
        metrics_path: Target path for output metrics JSON.
        plot_path: Target path for confusion matrix PNG.

    Returns:
        Dictionary containing all evaluation metrics and classification report.
    """
    ensure_directories()
    m_path = model_path or DEFAULT_MODEL_FILE
    d_path = dataset_path or DEFAULT_RAW_DATA_FILE
    s_path = split_path or DEFAULT_SPLIT_FILE
    out_metrics = metrics_path or DEFAULT_METRICS_FILE
    out_plot = plot_path or DEFAULT_CONFUSION_MATRIX_FILE

    if not m_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {m_path}. Please run 'python main.py train' first."
        )

    logger.info(f"Loading model pipeline from: {m_path}")
    pipeline = joblib.load(m_path)

    df_raw = load_dataset(d_path)
    df_clean = clean_dataset(df_raw)
    df_feat = engineer_features(df_clean, validate_columns=True)
    X, y = split_features_and_target(df_feat)

    # Provenance check: verify dataset hash
    current_hash = get_file_hash(d_path)

    if not s_path.exists():
        raise FileNotFoundError(
            f"Persisted train/test split file not found at: {s_path}.\n"
            f"Evaluation requires the exact split indices used during model training to guarantee valid held-out assessment.\n"
            f"Please run 'python main.py train' to regenerate consistent model and split artifacts."
        )

    logger.info(f"Loading reproducible test split indices from: {s_path}")
    split_info = load_json(s_path)

    # Strictly validate split metadata, coverage, overlap, bounds, and dataset hash
    validate_split_metadata(split_info, total_rows=len(X), expected_hash=current_hash)

    test_indices = split_info["test_indices"]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    # Compute model artifact hash for cross-artifact provenance
    model_sha256 = get_file_hash(m_path)

    logger.info(f"Evaluating on {len(X_test)} held-out test observations...")
    pos_idx = get_positive_class_index(pipeline)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, pos_idx]

    # Calculate metrics with explicit positive class (1) and binary averaging
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    tn, fp = cm[0]
    fn, tp = cm[1]
    total_test = tn + fp + fn + tp
    assert total_test == len(y_test), "Confusion matrix sum must equal total test samples!"

    metrics = {
        "evaluated_at": get_timestamp(),
        "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
        "dataset_source": "UCI Machine Learning Repository / Matan Atzmon et al.",
        "dataset_file": str(d_path.name),
        "dataset_sha256": current_hash,
        "model_file": str(m_path.name),
        "model_sha256": model_sha256,
        "split_file": str(s_path.name),
        "split_method": "Stratified Train/Test Split (80/20) with random_state=42",
        "total_test_samples": int(len(y_test)),
        "class_support": {
            "total": int(len(y_test)),
            "positive_failure": int(y_test.sum()),
            "negative_normal": int((y_test == 0).sum()),
        },
        "target_positive_class": 1,
        "metric_averaging_strategy": "binary",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm,
        },
        "classification_report": report_dict,
        "note": (
            "Accuracy is reported for completeness but is uninformative due to the 96.6% "
            "majority non-failure class. Prioritize Recall, Precision, and PR-AUC."
        ),
        "split_limitation_disclaimer": (
            "A random stratified split evaluates held-out classification under identically distributed conditions. "
            "However, because the observations derive from a continuous simulation stream, a random split does not "
            "demonstrate temporal forecasting of future machine degradation or generalization to completely unseen machines."
        ),
    }

    # Save metrics JSON
    save_json(metrics, out_metrics)
    logger.info(f"Evaluation metrics saved to: {out_metrics}")

    # Plot Confusion Matrix
    save_confusion_matrix_plot(cm, out_plot)
    logger.info(f"Confusion matrix plot saved to: {out_plot}")

    return metrics


def save_confusion_matrix_plot(cm: list, filepath: Path) -> None:
    """Generate and save a Seaborn heatmap confusion matrix."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal (0)", "Predicted Failure (1)"],
        yticklabels=["Actual Normal (0)", "Actual Failure (1)"],
        cbar=False,
        ax=ax,
    )
    ax.set_title("AI4I Predictive Maintenance: Confusion Matrix", fontsize=12, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"Evaluation halted: {e}")
