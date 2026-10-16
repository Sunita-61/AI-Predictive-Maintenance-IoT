"""
Preprocessing module implementing target leakage guards, identifier exclusion,
and feature-target separation for the AI4I 2020 dataset.
"""

from typing import List, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.utils import setup_logger

logger = setup_logger("preprocessing")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preliminary data cleaning and drop identifiers and leakage columns.

    Args:
        df: Input raw DataFrame.

    Returns:
        Cleaned DataFrame without identifier or leakage features.
    """
    df_cleaned = df.copy()

    # Drop non-predictive identifiers if present
    cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df_cleaned.columns]
    if cols_to_drop:
        logger.info(f"Excluding non-predictive identifier columns: {cols_to_drop}")
        df_cleaned.drop(columns=cols_to_drop, inplace=True)

    # Strictly exclude failure-mode columns to prevent catastrophic target leakage
    leakage_present = [c for c in LEAKAGE_COLUMNS if c in df_cleaned.columns]
    if leakage_present:
        logger.info(
            f"Applying target leakage safeguard. Dropping failure-type columns: {leakage_present}"
        )
        df_cleaned.drop(columns=leakage_present, inplace=True)

    # Handle missing values if any exist in the dataset
    missing_count = df_cleaned.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values. Dropping incomplete rows.")
        df_cleaned.dropna(inplace=True)

    return df_cleaned


def split_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split cleaned DataFrame into feature matrix X and target vector y.

    Args:
        df: Cleaned DataFrame containing both features and the target label.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series)

    Raises:
        ValueError: If TARGET_COLUMN is missing.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in DataFrame columns: {list(df.columns)}"
        )

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    logger.info(f"Features and target split. Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def build_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for categorical one-hot encoding
    and numerical standard scaling.

    Args:
        categorical_cols: List of categorical feature names.
        numerical_cols: List of numerical feature names.

    Returns:
        Configured scikit-learn ColumnTransformer.
    """
    cats = categorical_cols if categorical_cols is not None else CATEGORICAL_FEATURES
    nums = numerical_cols if numerical_cols is not None else NUMERICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cats,
            ),
            (
                "num",
                StandardScaler(),
                nums,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def generate_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Generate reproducible stratified train and test partitions along with exact row index lists.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of dataset for test partition.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, split_metadata_dict)
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    split_meta = {
        "random_state": random_state,
        "test_size": test_size,
        "train_indices": [int(idx) for idx in X_train.index],
        "test_indices": [int(idx) for idx in X_test.index],
        "train_count": len(X_train),
        "test_count": len(X_test),
        "train_failure_count": int((y_train == 1).sum()),
        "test_failure_count": int((y_test == 1).sum()),
        "train_normal_count": int((y_train == 0).sum()),
        "test_normal_count": int((y_test == 0).sum()),
    }

    return X_train, X_test, y_train, y_test, split_meta


def validate_split_metadata(
    split_info: dict,
    total_rows: int,
    expected_hash: Optional[str] = None,
) -> None:
    """
    Validate that train/test split metadata is sound, non-overlapping, and compatible.

    Checks:
    1. Required metadata keys are present.
    2. Dataset SHA-256 hash matches the expected raw data file (if provided).
    3. Training and test indices are non-empty integer lists.
    4. Neither partition contains duplicate indices.
    5. Training and test partitions are completely disjoint (zero overlap).
    6. Partitions completely cover all dataset rows: train + test == total_rows.
    7. All row indices fall within the valid bound [0, total_rows - 1].
    8. Partition counts agree with recorded summary figures.

    Args:
        split_info: Loaded dictionary from train_test_split.json.
        total_rows: Total observation count of the active cleaned dataset.
        expected_hash: Optional SHA-256 hash string of the active dataset.

    Raises:
        ValueError: If any integrity or consistency rule is violated.
    """
    required_keys = ["train_indices", "test_indices", "train_count", "test_count"]
    for key in required_keys:
        if key not in split_info:
            raise ValueError(f"Malformed split metadata: Missing required key '{key}'.")

    # Hash verification
    if expected_hash and "dataset_sha256" in split_info:
        saved_hash = split_info["dataset_sha256"]
        if saved_hash != expected_hash:
            raise ValueError(
                f"Dataset provenance mismatch: Split was generated for dataset with SHA-256 "
                f"'{saved_hash}', but active dataset hash is '{expected_hash}'. "
                f"Please run 'python main.py train' to regenerate consistent artifacts."
            )

    train_idx = split_info["train_indices"]
    test_idx = split_info["test_indices"]

    if not isinstance(train_idx, list) or not isinstance(test_idx, list):
        raise ValueError("Train and test indices in split metadata must be lists of integers.")

    # Duplicate index check
    if len(train_idx) != len(set(train_idx)):
        raise ValueError("Invalid split: Training partition contains duplicate indices.")
    if len(test_idx) != len(set(test_idx)):
        raise ValueError("Invalid split: Test partition contains duplicate indices.")

    train_set = set(train_idx)
    test_set = set(test_idx)

    # Disjoint check (zero overlap)
    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(
            f"Invalid split: Training and test partitions overlap! Found {len(overlap)} shared indices."
        )

    # Coverage and bounds check
    all_indices = train_set.union(test_set)
    if len(all_indices) != total_rows:
        raise ValueError(
            f"Invalid split: Partition union size ({len(all_indices)}) does not equal "
            f"total dataset rows ({total_rows}). Every row must belong to exactly one partition."
        )

    min_idx = min(all_indices) if all_indices else -1
    max_idx = max(all_indices) if all_indices else -1
    if min_idx < 0 or max_idx >= total_rows:
        raise ValueError(
            f"Invalid split: Indices out of bounds! Range is [{min_idx}, {max_idx}], "
            f"expected [0, {total_rows - 1}]."
        )

    # Count agreement
    if len(train_idx) != split_info["train_count"]:
        raise ValueError(
            f"Split count mismatch: train_indices length ({len(train_idx)}) != "
            f"recorded train_count ({split_info['train_count']})."
        )
    if len(test_idx) != split_info["test_count"]:
        raise ValueError(
            f"Split count mismatch: test_indices length ({len(test_idx)}) != "
            f"recorded test_count ({split_info['test_count']})."
        )


if __name__ == "__main__":
    print("Preprocessing module ready. ColumnTransformer and Split Validator configured.")
