"""
Feature engineering module for domain-specific industrial IoT transformations.

Calculates thermodynamic, mechanical, and tool strain indicators.
Ensures strict consistency between offline training and real-time inference.
"""

import math
from typing import List
import pandas as pd

from src.config import NUMERICAL_FEATURES
from src.utils import setup_logger

logger = setup_logger("feature_engineering")

# Base sensor columns required to compute domain features
REQUIRED_BASE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def engineer_features(df: pd.DataFrame, validate_columns: bool = True) -> pd.DataFrame:
    """
    Apply domain-grounded feature engineering transformations.

    Formulas:
    1. Temp_Difference_K:
       delta_T = Process temperature [K] - Air temperature [K]
       (Physical meaning: Measures heat dissipation efficacy. Smaller delta T indicates
       inadequate cooling and potential thermal distress).

    2. Power_Proxy_W:
       P = Torque [Nm] * omega = Torque [Nm] * (Rotational speed [rpm] * 2 * pi / 60)
       (Physical meaning: Approximate mechanical power transmitted through spindle).

    3. Overstrain_Proxy:
       OS = Torque [Nm] * Tool wear [min]
       (Physical meaning: Product of mechanical load and tool contact degradation).

    Args:
        df: Input DataFrame containing AI4I sensor columns.
        validate_columns: If True, validates that all required base sensor columns exist.

    Returns:
        DataFrame containing original features and the 3 appended engineered features.

    Raises:
        ValueError: If validate_columns is True and any required base column is missing.
    """
    if validate_columns:
        missing = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Feature engineering failed: Missing required base sensor column(s): {missing}.\n"
                f"Available columns: {list(df.columns)}"
            )

    df_feat = df.copy()

    # 1. Thermal dissipation delta
    df_feat["Temp_Difference_K"] = (
        df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
    )

    # 2. Rotational mechanical power proxy: Power = Torque * omega (rad/s)
    omega_factor = 2.0 * math.pi / 60.0
    df_feat["Power_Proxy_W"] = (
        df_feat["Torque [Nm]"] * df_feat["Rotational speed [rpm]"] * omega_factor
    )

    # 3. Mechanical overstrain indicator: Torque * Tool Wear
    df_feat["Overstrain_Proxy"] = (
        df_feat["Torque [Nm]"] * df_feat["Tool wear [min]"]
    )

    return df_feat


def get_all_feature_names(
    categorical_cols: List[str], numerical_cols: List[str]
) -> List[str]:
    """Return the full list of expected feature names after engineering."""
    return categorical_cols + numerical_cols + [
        "Temp_Difference_K",
        "Power_Proxy_W",
        "Overstrain_Proxy",
    ]


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "Type": ["M"],
            "Air temperature [K]": [298.1],
            "Process temperature [K]": [308.6],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [120],
        }
    )
    result = engineer_features(sample)
    print("Engineered features sample:\n", result)
