"""
Dataset loading and schema verification module for the AI4I 2020 dataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.config import (
    DEFAULT_RAW_DATA_FILE,
    EXPECTED_RAW_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.utils import setup_logger

logger = setup_logger("data_loader")


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verify whether the DataFrame contains the required AI4I 2020 columns.

    Args:
        df: Pandas DataFrame to validate.

    Returns:
        Tuple of (is_valid: bool, missing_columns: List[str])
    """
    cleaned_cols = [col.strip() for col in df.columns]
    missing = [col for col in EXPECTED_RAW_COLUMNS if col not in cleaned_cols]
    is_valid = len(missing) == 0
    return is_valid, missing


def audit_dataset_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive audit of the dataset according to the AI4I benchmark requirements:
    1. Row and column counts.
    2. Column presence and missing column check.
    3. Missing values analysis.
    4. Duplicate rows detection.
    5. Data types inspection.
    6. Target value validation (must be binary 0 or 1).
    7. Class balance analysis (proportions and counts).
    8. Numeric physical sanity checks (temperatures > 0 K, non-negative rpm, torque, tool wear).

    Args:
        df: Loaded raw DataFrame.

    Returns:
        Structured audit report dictionary.
    """
    # Clean whitespace in column headers
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    rows, cols = df_clean.shape
    is_valid_schema, missing_cols = validate_dataset_schema(df_clean)

    # Missing values
    null_counts = df_clean.isnull().sum()
    total_nulls = int(null_counts.sum())
    cols_with_nulls = {k: int(v) for k, v in null_counts.items() if v > 0}

    # Duplicate rows
    duplicates_count = int(df_clean.duplicated().sum())

    # Target column analysis
    target_present = TARGET_COLUMN in df_clean.columns
    target_stats: Dict[str, Any] = {}
    if target_present:
        target_series = df_clean[TARGET_COLUMN]
        unique_vals = set(target_series.dropna().unique())
        is_binary = unique_vals.issubset({0, 1})
        val_counts = target_series.value_counts().to_dict()
        fail_count = int(val_counts.get(1, 0))
        norm_count = int(val_counts.get(0, 0))
        fail_rate = float(fail_count / rows) if rows > 0 else 0.0

        target_stats = {
            "present": True,
            "is_binary": is_binary,
            "unique_values": list(unique_vals),
            "normal_count": norm_count,
            "failure_count": fail_count,
            "failure_rate_percent": round(fail_rate * 100.0, 2),
            "class_imbalance_ratio": f"{norm_count}:{fail_count}",
        }
    else:
        target_stats = {"present": False}

    # Physical numerical sanity checks
    invalid_physical_values = {}
    if "Air temperature [K]" in df_clean.columns:
        invalid_air = int((df_clean["Air temperature [K]"] <= 0).sum())
        if invalid_air > 0:
            invalid_physical_values["Air temperature [K] <= 0 K"] = invalid_air

    if "Process temperature [K]" in df_clean.columns:
        invalid_proc = int((df_clean["Process temperature [K]"] <= 0).sum())
        if invalid_proc > 0:
            invalid_physical_values["Process temperature [K] <= 0 K"] = invalid_proc

    if "Rotational speed [rpm]" in df_clean.columns:
        invalid_rpm = int((df_clean["Rotational speed [rpm]"] < 0).sum())
        if invalid_rpm > 0:
            invalid_physical_values["Rotational speed [rpm] < 0"] = invalid_rpm

    if "Torque [Nm]" in df_clean.columns:
        invalid_torque = int((df_clean["Torque [Nm]"] < 0).sum())
        if invalid_torque > 0:
            invalid_physical_values["Torque [Nm] < 0"] = invalid_torque

    if "Tool wear [min]" in df_clean.columns:
        invalid_wear = int((df_clean["Tool wear [min]"] < 0).sum())
        if invalid_wear > 0:
            invalid_physical_values["Tool wear [min] < 0"] = invalid_wear

    # Leakage columns present
    leakage_found = [c for c in LEAKAGE_COLUMNS if c in df_clean.columns]

    return {
        "rows": rows,
        "columns": cols,
        "is_valid_schema": is_valid_schema,
        "missing_columns": missing_cols,
        "total_null_values": total_nulls,
        "columns_with_nulls": cols_with_nulls,
        "duplicate_rows": duplicates_count,
        "target_analysis": target_stats,
        "invalid_physical_values": invalid_physical_values,
        "leakage_columns_present_in_raw": leakage_found,
        "dtypes": {str(k): str(v) for k, v in df_clean.dtypes.items()},
    }


def format_audit_report(audit: Dict[str, Any]) -> str:
    """Format the audit dictionary into a human-readable text report."""
    lines = [
        "=" * 60,
        "AI4I 2020 DATASET QUALITY & SCHEMA AUDIT REPORT",
        "=" * 60,
        f"Observation Count (Rows):    {audit['rows']}",
        f"Feature Count (Columns):       {audit['columns']}",
        f"Schema Validation Passed:     {'YES' if audit['is_valid_schema'] else 'NO'}",
    ]

    if audit["missing_columns"]:
        lines.append(f"Missing Standard Columns:     {audit['missing_columns']}")

    lines.append(f"Missing Values:               {audit['total_null_values']}")
    if audit["columns_with_nulls"]:
        lines.append(f"  Columns with Nulls:         {audit['columns_with_nulls']}")

    lines.append(f"Duplicate Rows:               {audit['duplicate_rows']}")

    target = audit.get("target_analysis", {})
    if target.get("present"):
        lines.append("Target Analysis ('Machine failure'):")
        lines.append(f"  Binary Valid (0/1):         {'YES' if target['is_binary'] else 'NO'}")
        lines.append(f"  Normal Class (0):           {target['normal_count']}")
        lines.append(f"  Failure Class (1):          {target['failure_count']}")
        lines.append(f"  Failure Prevalence:         {target['failure_rate_percent']}%")
        lines.append(f"  Imbalance Ratio:            {target['class_imbalance_ratio']}")
    else:
        lines.append("Target Column ('Machine failure'): NOT FOUND")

    if audit["invalid_physical_values"]:
        lines.append(f"Physical Sanity Violations:   {audit['invalid_physical_values']}")
    else:
        lines.append("Physical Sanity Violations:   NONE (All numeric ranges within physical bounds)")

    lines.append(f"Target Leakage Columns in Raw: {audit['leakage_columns_present_in_raw']}")
    lines.append("(Note: Leakage columns are strictly dropped during preprocessing prior to model fitting)")
    lines.append("=" * 60)
    return "\n".join(lines)


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the AI4I 2020 Predictive Maintenance dataset from CSV.

    Args:
        filepath: Optional path to the CSV file. Defaults to DEFAULT_RAW_DATA_FILE.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file is missing.
        ValueError: If essential columns or the target column are not present.
    """
    path = filepath or DEFAULT_RAW_DATA_FILE

    if not path.exists():
        msg = (
            f"Dataset not found at expected location: {path}\n"
            f"Please download 'ai4i2020.csv' from the official UCI repository:\n"
            f"https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            f"and place it at: {DEFAULT_RAW_DATA_FILE}"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Normalize column names in case of whitespace
    df.columns = [c.strip() for c in df.columns]

    is_valid, missing = validate_dataset_schema(df)
    if not is_valid:
        critical_missing = [c for c in [TARGET_COLUMN] if c not in df.columns]
        if critical_missing:
            raise ValueError(
                f"Malformed dataset: Critical target column '{TARGET_COLUMN}' is missing.\n"
                f"Found columns: {list(df.columns)}"
            )
        logger.warning(
            f"Dataset loaded with non-standard column set. Missing standard columns: {missing}"
        )

    logger.info(f"Dataset successfully loaded with shape: {df.shape}")
    return df


if __name__ == "__main__":
    try:
        data = load_dataset()
        audit_rep = audit_dataset_quality(data)
        print(format_audit_report(audit_rep))
    except FileNotFoundError as err:
        print(f"Data loader check: {err}")
