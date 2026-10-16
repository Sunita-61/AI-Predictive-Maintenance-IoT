"""
Model training pipeline for AI Predictive Maintenance.

Trains a scikit-learn RandomForestClassifier with class weighting,
stratified train-test splitting, provenance tracking, and pipeline serialization via Joblib.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_FILE,
    DEFAULT_RAW_DATA_FILE,
    DEFAULT_SPLIT_FILE,
    ENGINEERED_FEATURES,
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    MAX_DEPTH,
    MODELS_DIR,
    N_ESTIMATORS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_directories,
)
from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import (
    build_preprocessor,
    clean_dataset,
    generate_stratified_split,
    split_features_and_target,
)
from src.utils import get_file_hash, get_timestamp, save_json, setup_logger

logger = setup_logger("train_model")


def train(
    dataset_path: Optional[Path] = None,
    model_save_path: Optional[Path] = None,
    split_save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute the complete reproducible model training workflow:
    1. Load raw dataset and validate schema.
    2. Compute dataset SHA-256 hash for provenance and reproducibility.
    3. Apply cleaning and target leakage safeguards.
    4. Generate engineered domain features.
    5. Stratify split into train and test partitions using shared split generator.
    6. Persist exact split indices to models/train_test_split.json.
    7. Fit a Scikit-Learn Pipeline (ColumnTransformer + RandomForestClassifier).
    8. Serialize the pipeline to models/ and record metadata.

    Args:
        dataset_path: Path to raw dataset CSV.
        model_save_path: Target path to save the serialized joblib artifact.
        split_save_path: Target path to save the split index metadata.

    Returns:
        Dictionary containing training metadata, data shapes, and parameters.
    """
    ensure_directories()
    raw_path = dataset_path or DEFAULT_RAW_DATA_FILE
    save_path = model_save_path or DEFAULT_MODEL_FILE
    split_path = split_save_path or DEFAULT_SPLIT_FILE

    logger.info("Starting model training pipeline...")
    df_raw = load_dataset(raw_path)

    # Compute provenance dataset hash
    dataset_sha256 = get_file_hash(raw_path)
    logger.info(f"Dataset SHA-256 hash verified: {dataset_sha256}")

    # Clean and remove leakage/identifiers
    df_clean = clean_dataset(df_raw)

    # Feature engineering
    df_features = engineer_features(df_clean, validate_columns=True)

    # Split into X and y
    X, y = split_features_and_target(df_features)

    # Stratified train/test split to preserve 3.39% minority class distribution
    X_train, X_test, y_train, y_test, split_meta = generate_stratified_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Enrich and persist split metadata
    split_meta["dataset_file"] = str(raw_path.name)
    split_meta["dataset_sha256"] = dataset_sha256
    split_meta["created_at"] = get_timestamp()
    save_json(split_meta, split_path)
    logger.info(f"Stratified split indices saved to: {split_path}")

    logger.info(
        f"Data split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples "
        f"(Train Failure Rate: {y_train.mean():.2%}, Test Failure Rate: {y_test.mean():.2%})"
    )

    # All numerical columns including engineered ones
    all_numerical = NUMERICAL_FEATURES + ENGINEERED_FEATURES

    # Build preprocessing column transformer
    preprocessor = build_preprocessor(
        categorical_cols=CATEGORICAL_FEATURES,
        numerical_cols=all_numerical,
    )

    # RandomForest with balanced class weights to address minority class rarity
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=None,
    )

    # Construct unified Scikit-Learn pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    logger.info("Fitting RandomForest pipeline...")
    pipeline.fit(X_train, y_train)

    # Persist model bundle
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Trained model pipeline saved to: {save_path}")

    model_sha256 = get_file_hash(save_path)

    metadata = {
        "created_at": get_timestamp(),
        "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
        "dataset_source": "UCI Machine Learning Repository (Dua and Graff, 2019 / Matan Atzmon et al.)",
        "dataset_file": str(raw_path.name),
        "dataset_sha256": dataset_sha256,
        "dataset_total_rows": int(len(df_raw)),
        "target_column": TARGET_COLUMN,
        "positive_class_definition": "Machine failure == 1 (Failure detected across any failure mode)",
        "negative_class_definition": "Machine failure == 0 (Normal nominal operation)",
        "split_method": "stratified",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": list(X.columns),
        "feature_order": list(X.columns),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": all_numerical,
        "engineered_feature_definitions": {
            "Temp_Difference_K": "Process temperature [K] - Air temperature [K] (thermal dissipation efficacy)",
            "Power_Proxy_W": "Torque [Nm] * Rotational speed [rpm] * (2*pi/60) (mechanical power proxy)",
            "Overstrain_Proxy": "Torque [Nm] * Tool wear [min] (cumulative stress proxy)",
        },
        "excluded_columns": {
            "identifiers": IDENTIFIER_COLUMNS,
            "leakage_safeguards": LEAKAGE_COLUMNS,
        },
        "model_class": "sklearn.ensemble.RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": None,
        },
        "decision_threshold": 0.5,
        "environment_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "class_balance_train": {
            "normal": int((y_train == 0).sum()),
            "failure": int((y_train == 1).sum()),
        },
        "class_balance_test": {
            "normal": int((y_test == 0).sum()),
            "failure": int((y_test == 1).sum()),
        },
        "model_file": str(save_path.name),
        "model_sha256": model_sha256,
        "split_file": str(split_path.name),
    }

    meta_path = MODELS_DIR / "model_metadata.json"
    save_json(metadata, meta_path)
    logger.info(f"Model metadata persisted to: {meta_path}")

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "split_metadata": split_meta,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    try:
        train()
    except FileNotFoundError as e:
        print(f"Training stopped: {e}")
