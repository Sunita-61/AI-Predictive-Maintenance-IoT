"""
Tests for data loader schema validation and dataset auditing.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.data_loader import (
    audit_dataset_quality,
    format_audit_report,
    load_dataset,
    validate_dataset_schema,
)


def test_validate_dataset_schema():
    # Valid mock dataframe containing all 14 AI4I columns
    valid_cols = [
        "UDI",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    df_valid = pd.DataFrame(columns=valid_cols)
    is_valid, missing = validate_dataset_schema(df_valid)
    assert is_valid is True
    assert len(missing) == 0

    # Incomplete schema
    df_incomplete = pd.DataFrame(columns=["Type", "Air temperature [K]"])
    is_valid, missing = validate_dataset_schema(df_incomplete)
    assert is_valid is False
    assert "Machine failure" in missing
    assert "Torque [Nm]" in missing


def test_audit_dataset_quality():
    data = {
        "UDI": [1, 2, 3],
        "Product ID": ["L14860", "M14861", "H14862"],
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [298.1, 298.2, 300.0],
        "Process temperature [K]": [308.6, 308.7, 310.0],
        "Rotational speed [rpm]": [1500, 1400, 1600],
        "Torque [Nm]": [40.0, 45.0, 38.0],
        "Tool wear [min]": [0, 5, 10],
        "Machine failure": [0, 0, 1],
        "TWF": [0, 0, 0],
        "HDF": [0, 0, 0],
        "PWF": [0, 0, 0],
        "OSF": [0, 0, 1],
        "RNF": [0, 0, 0],
    }
    df = pd.DataFrame(data)
    audit = audit_dataset_quality(df)

    assert audit["rows"] == 3
    assert audit["columns"] == 14
    assert audit["is_valid_schema"] is True
    assert audit["total_null_values"] == 0
    assert audit["target_analysis"]["failure_count"] == 1
    assert audit["target_analysis"]["normal_count"] == 2
    assert audit["target_analysis"]["is_binary"] is True

    # Test report formatting
    report_text = format_audit_report(audit)
    assert "AI4I 2020 DATASET QUALITY" in report_text
    assert "Failure Prevalence" in report_text


def test_load_dataset_missing_file_raises_error():
    non_existent = Path("data/raw/does_not_exist.csv")
    with pytest.raises(FileNotFoundError, match="https://archive.ics.uci.edu"):
        load_dataset(non_existent)
