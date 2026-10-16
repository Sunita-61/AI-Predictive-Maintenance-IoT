"""
Utility functions for logging, JSON persistence, and system metadata.
"""

from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import json
import logging
import os


def setup_logger(name: str = "predictive_maintenance", level: str = "INFO") -> logging.Logger:
    """
    Configure and return a standardized logger with console output formatting.

    Args:
        name: Logger identifier name.
        level: Logging verbosity level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        logging.Logger instance.
    """
    env_level = os.getenv("LOG_LEVEL", level).upper()
    log_level = getattr(logging, env_level, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save a dictionary as a formatted JSON file.

    Args:
        data: Dictionary data to persist.
        filepath: Target Path object.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dictionary containing file contents.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    """Return an ISO 8601 formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_file_hash(filepath: Path) -> str:
    """
    Compute and return the SHA-256 hash of a file for reproducibility and provenance tracking.

    Args:
        filepath: Path to the target file.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    import hashlib

    if not filepath.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {filepath}")

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


compute_file_sha256 = get_file_hash


def verify_artifact_provenance(
    model_path: Any = None,
    dataset_path: Any = None,
    metadata_path: Any = None,
    split_path: Any = None,
    metrics_path: Any = None,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Verify cross-artifact provenance, dataset hash matching, and model artifact alignment.

    Args:
        model_path: Path to serialized model pipeline.
        dataset_path: Path to raw dataset CSV.
        metadata_path: Path to model_metadata.json.
        split_path: Path to train_test_split.json.
        metrics_path: Path to evaluation_metrics.json.
        raise_on_error: If True, raises ValueError upon any provenance mismatch.

    Returns:
        Dictionary with 'is_consistent', 'model_sha256', 'dataset_sha256', and 'mismatches'.

    Raises:
        ValueError: If raise_on_error is True and any artifact is missing or mismatched.
    """
    from src.config import (
        DEFAULT_METRICS_FILE,
        DEFAULT_MODEL_FILE,
        DEFAULT_RAW_DATA_FILE,
        DEFAULT_SPLIT_FILE,
        MODELS_DIR,
    )

    m_path = model_path or DEFAULT_MODEL_FILE
    d_path = dataset_path or DEFAULT_RAW_DATA_FILE
    meta_path = metadata_path or (MODELS_DIR / "model_metadata.json")
    s_path = split_path or DEFAULT_SPLIT_FILE
    met_path = metrics_path or DEFAULT_METRICS_FILE

    mismatches = []
    dataset_hash = None
    model_hash = None

    if not d_path.exists():
        mismatches.append(f"Raw dataset file missing at: {d_path}")
    else:
        dataset_hash = get_file_hash(d_path)

    if not m_path.exists():
        mismatches.append(f"Model artifact file missing at: {m_path}")
    else:
        model_hash = get_file_hash(m_path)

    if not meta_path.exists():
        mismatches.append(f"Model metadata file missing at: {meta_path}")
    else:
        try:
            meta = load_json(meta_path)
            if dataset_hash and meta.get("dataset_sha256") != dataset_hash:
                mismatches.append(
                    f"Metadata dataset hash mismatch: metadata has '{meta.get('dataset_sha256')}', "
                    f"active dataset has '{dataset_hash}'"
                )
            if model_hash and meta.get("model_sha256") and meta.get("model_sha256") != model_hash:
                mismatches.append(
                    f"Metadata model hash mismatch: metadata has '{meta.get('model_sha256')}', "
                    f"active model has '{model_hash}'"
                )
        except Exception as e:
            mismatches.append(f"Failed to read model metadata: {e}")

    if not s_path.exists():
        mismatches.append(f"Train/test split file missing at: {s_path}")
    else:
        try:
            split_info = load_json(s_path)
            if dataset_hash and split_info.get("dataset_sha256") != dataset_hash:
                mismatches.append(
                    f"Split dataset hash mismatch: split has '{split_info.get('dataset_sha256')}', "
                    f"active dataset has '{dataset_hash}'"
                )
        except Exception as e:
            mismatches.append(f"Failed to read split file: {e}")

    if met_path.exists():
        try:
            metrics_data = load_json(met_path)
            if dataset_hash and metrics_data.get("dataset_sha256") != dataset_hash:
                mismatches.append(
                    f"Metrics dataset hash mismatch: metrics has '{metrics_data.get('dataset_sha256')}', "
                    f"active dataset has '{dataset_hash}'"
                )
            if model_hash and metrics_data.get("model_sha256") and metrics_data.get("model_sha256") != model_hash:
                mismatches.append(
                    f"Metrics model hash mismatch: metrics has '{metrics_data.get('model_sha256')}', "
                    f"active model has '{model_hash}'"
                )
        except Exception as e:
            mismatches.append(f"Failed to read metrics file: {e}")

    is_consistent = len(mismatches) == 0

    if raise_on_error and not is_consistent:
        raise ValueError(f"Artifact provenance verification failed: {'; '.join(mismatches)}")

    return {
        "is_consistent": is_consistent,
        "dataset_sha256": dataset_hash,
        "model_sha256": model_hash,
        "mismatches": mismatches,
    }
