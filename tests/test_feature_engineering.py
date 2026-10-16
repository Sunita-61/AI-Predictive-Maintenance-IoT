"""
Tests for feature engineering transformations, consistency, and column validation.
"""

import math
import pandas as pd
import pytest
from src.feature_engineering import engineer_features, get_all_feature_names


def test_engineer_features_calculation():
    """Verify deterministic calculations of thermodynamic and mechanical features."""
    sample = pd.DataFrame(
        {
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [120],
        }
    )

    df_out = engineer_features(sample, validate_columns=True)

    # 1. Temp difference: 310 - 300 = 10.0 K
    assert "Temp_Difference_K" in df_out.columns
    assert pytest.approx(df_out["Temp_Difference_K"].iloc[0], rel=1e-4) == 10.0

    # 2. Power proxy: 40 * 1500 * (2 * pi / 60) = 40 * 157.0796 = 6283.185 W
    expected_power = 40.0 * 1500.0 * (2.0 * math.pi / 60.0)
    assert "Power_Proxy_W" in df_out.columns
    assert pytest.approx(df_out["Power_Proxy_W"].iloc[0], rel=1e-4) == expected_power

    # 3. Overstrain proxy: 40 * 120 = 4800
    assert "Overstrain_Proxy" in df_out.columns
    assert pytest.approx(df_out["Overstrain_Proxy"].iloc[0], rel=1e-4) == 4800.0


def test_engineer_features_missing_column_raises_error():
    """Verify that missing required base columns raise a clear ValueError."""
    incomplete_sample = pd.DataFrame(
        {
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            # Process temperature missing!
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [120],
        }
    )

    with pytest.raises(ValueError, match="Missing required base sensor column"):
        engineer_features(incomplete_sample, validate_columns=True)


def test_get_all_feature_names():
    """Verify helper returns complete combined feature names."""
    cats = ["Type"]
    nums = ["Air temperature [K]", "Torque [Nm]"]
    names = get_all_feature_names(cats, nums)
    assert "Temp_Difference_K" in names
    assert "Power_Proxy_W" in names
    assert "Overstrain_Proxy" in names
