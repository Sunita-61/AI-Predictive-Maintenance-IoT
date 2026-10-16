"""
Tests for preprocessing safeguards, identifier exclusion, and leakage prevention.
"""

import pandas as pd
import pytest
from src.config import LEAKAGE_COLUMNS
from src.preprocessing import clean_dataset, split_features_and_target


def test_leakage_and_identifier_exclusion():
    raw_mock = pd.DataFrame(
        {
            "UDI": [1, 2],
            "Product ID": ["L14860", "M14861"],
            "Type": ["L", "M"],
            "Air temperature [K]": [298.1, 298.2],
            "Process temperature [K]": [308.6, 308.7],
            "Rotational speed [rpm]": [1500, 1400],
            "Torque [Nm]": [40.0, 45.0],
            "Tool wear [min]": [0, 5],
            "Machine failure": [0, 1],
            # Leakage failure subtype columns
            "TWF": [0, 0],
            "HDF": [0, 0],
            "PWF": [0, 1],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )

    cleaned = clean_dataset(raw_mock)

    # Identifiers must be removed
    assert "UDI" not in cleaned.columns
    assert "Product ID" not in cleaned.columns

    # Leakage failure subtype columns must be strictly excluded
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in cleaned.columns

    # Features and target split
    X, y = split_features_and_target(cleaned)
    assert "Machine failure" not in X.columns
    assert len(y) == 2
    assert list(y) == [0, 1]

    # Verify no leakage column is in X
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in X.columns


def test_split_features_and_target_missing_target_raises_error():
    df_no_target = pd.DataFrame(
        {
            "Type": ["L"],
            "Air temperature [K]": [300.0],
        }
    )
    with pytest.raises(ValueError, match="Target column 'Machine failure' not found"):
        split_features_and_target(df_no_target)
