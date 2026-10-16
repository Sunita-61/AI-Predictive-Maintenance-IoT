"""
Inference module for real-time and batch equipment failure prediction.

Loads serialized model pipelines, validates observation schemas,
applies identical feature engineering, and calculates risk scores.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import pandas as pd

from src.config import ALL_RAW_FEATURES, DEFAULT_MODEL_FILE
from src.feature_engineering import engineer_features
from src.utils import setup_logger

logger = setup_logger("predictor")

_MODEL_CACHE = {}


def load_model_artifact(model_path: Optional[Path] = None):
    """
    Load serialized model pipeline with memory caching.

    Args:
        model_path: Optional path to the serialized model file.

    Returns:
        Loaded Scikit-Learn Pipeline.
    """
    path = model_path or DEFAULT_MODEL_FILE
    path_key = str(path)

    if path_key in _MODEL_CACHE:
        return _MODEL_CACHE[path_key]

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {path}\n"
            f"Please train the model first by running: python main.py train"
        )

    logger.info(f"Loading model pipeline into memory from: {path}")
    pipeline = joblib.load(path)
    _MODEL_CACHE[path_key] = pipeline
    return pipeline


def extract_model_features(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate only the feature subset required for model inference,
    safely filtering out metadata keys (such as 'is_simulated', 'scenario', etc.).

    Enforces:
    - Presence of all required raw sensor keys.
    - Valid machine quality Type ('L', 'M', or 'H').
    - Parsing to finite floating-point values (rejecting NaN, Inf, non-numeric).
    - Physical domain boundaries:
      * Absolute temperatures > 0 K (strictly positive Kelvin scale).
      * Rotational speed >= 0 rpm.
      * Torque >= 0 Nm.
      * Tool wear >= 0 min.

    Args:
        observation: Dictionary containing sensor readings and optional metadata.

    Returns:
        Clean dictionary containing only the valid AI4I model features.

    Raises:
        ValueError: If required features are missing, non-numeric, non-finite, or physically invalid.
    """
    missing = [feat for feat in ALL_RAW_FEATURES if feat not in observation]
    if missing:
        raise ValueError(
            f"Input observation is missing required sensor features: {missing}\n"
            f"Expected features: {ALL_RAW_FEATURES}"
        )

    device_type = str(observation["Type"]).strip().upper()
    if device_type not in ["L", "M", "H"]:
        raise ValueError(f"Invalid device Type '{device_type}'. Must be 'L', 'M', or 'H'.")

    # Define physical boundary rules
    physical_bounds = {
        "Air temperature [K]": {"min_exclusive": 0.0, "max_inclusive": 1000.0},
        "Process temperature [K]": {"min_exclusive": 0.0, "max_inclusive": 1000.0},
        "Rotational speed [rpm]": {"min_inclusive": 0.0, "max_inclusive": 50000.0},
        "Torque [Nm]": {"min_inclusive": 0.0, "max_inclusive": 1000.0},
        "Tool wear [min]": {"min_inclusive": 0.0, "max_inclusive": 2000.0},
    }

    clean_dict: Dict[str, Any] = {"Type": device_type}
    for field, bounds in physical_bounds.items():
        val = observation[field]
        try:
            val_float = float(val)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Invalid non-numeric sensor value for '{field}': '{val}'."
            ) from err

        if not math.isfinite(val_float):
            raise ValueError(
                f"Invalid non-finite sensor value for '{field}': {val_float}."
            )

        if "min_exclusive" in bounds and val_float <= bounds["min_exclusive"]:
            raise ValueError(
                f"Physically impossible value for '{field}': {val_float}. "
                f"Must be strictly positive (> {bounds['min_exclusive']} K absolute temperature)."
            )
        if "min_inclusive" in bounds and val_float < bounds["min_inclusive"]:
            raise ValueError(
                f"Physically impossible value for '{field}': {val_float}. "
                f"Cannot be negative (must be >= {bounds['min_inclusive']})."
            )
        if "max_inclusive" in bounds and val_float > bounds["max_inclusive"]:
            raise ValueError(
                f"Sensor reading exceeds realistic physical bound for '{field}': {val_float} "
                f"(max {bounds['max_inclusive']})."
            )

        clean_dict[field] = val_float

    return clean_dict


def validate_observation(observation: Dict[str, Any]) -> None:
    """Validate that observation contains valid schema."""
    _ = extract_model_features(observation)


def get_positive_class_index(pipeline) -> int:
    """
    Dynamically locate the column index corresponding to positive failure class (1).
    Avoids assuming a fixed column position without checking classes_.

    Args:
        pipeline: Scikit-Learn Pipeline or Classifier.

    Returns:
        Integer column index for positive class label 1.
    """
    classes = getattr(pipeline, "classes_", None)
    if classes is None and hasattr(pipeline, "named_steps"):
        clf = pipeline.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "classes_"):
            classes = clf.classes_

    if classes is not None:
        try:
            return list(classes).index(1)
        except ValueError:
            pass
    return 1


def predict_single(
    observation: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate failure prediction and risk score for a single sensor telemetry reading.

    Args:
        observation: Dictionary with raw sensor inputs.
        model_path: Optional path to model file.

    Returns:
        Dictionary containing prediction results:
        - predicted_class: 0 (Normal) or 1 (Failure)
        - risk_score: Floating point score between 0.0 and 1.0 (uncalibrated)
        - is_failure: bool
        - is_simulated: bool (transparency indicator)
        - score_interpretation: str
        - disclaimer: str (if simulated)
    """
    is_sim = bool(observation.get("is_simulated", False))
    clean_obs = extract_model_features(observation)
    pipeline = load_model_artifact(model_path)

    df_single = pd.DataFrame([clean_obs])
    # Apply identical feature engineering transformations
    df_engineered = engineer_features(df_single, validate_columns=True)

    pos_idx = get_positive_class_index(pipeline)
    pred_class = int(pipeline.predict(df_engineered)[0])
    pred_score = float(pipeline.predict_proba(df_engineered)[0, pos_idx])

    if pred_class not in (0, 1):
        raise ValueError(
            f"Classifier returned unexpected prediction class: {pred_class}. Expected 0 or 1."
        )

    if not math.isfinite(pred_score) or not (0.0 <= pred_score <= 1.0):
        raise ValueError(
            f"Classifier returned invalid score: {pred_score}. Expected finite in [0.0, 1.0]."
        )

    result = {
        "predicted_class": pred_class,
        "is_failure": bool(pred_class == 1),
        "risk_score": round(pred_score, 4),
        "score_interpretation": (
            "Uncalibrated Random Forest tree voting fraction, not an empirical physical failure probability."
        ),
        "observation": clean_obs,
        "is_simulated": is_sim,
    }

    if is_sim:
        result["disclaimer"] = (
            "Demonstration prediction on synthetic telemetry; not validated on physical hardware."
        )

    return result


def predict_batch(
    observations: Union[List[Dict[str, Any]], pd.DataFrame],
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate predictions for a batch of sensor readings.

    Args:
        observations: List of observation dictionaries or a pandas DataFrame.
        model_path: Optional path to model file.

    Returns:
        DataFrame with appended 'Predicted_Class' and 'Risk_Score' columns.
    """
    pipeline = load_model_artifact(model_path)

    if isinstance(observations, list):
        clean_list = [extract_model_features(obs) for obs in observations]
        df = pd.DataFrame(clean_list)
    else:
        df = observations.copy()

    df_engineered = engineer_features(df, validate_columns=True)
    pos_idx = get_positive_class_index(pipeline)
    classes = pipeline.predict(df_engineered)
    scores = pipeline.predict_proba(df_engineered)[:, pos_idx]

    df_out = df.copy()
    df_out["Predicted_Class"] = classes
    df_out["Risk_Score"] = [round(float(s), 4) for s in scores]
    return df_out


if __name__ == "__main__":
    test_obs = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 120,
    }
    try:
        res = predict_single(test_obs)
        print("Prediction result:", res)
    except FileNotFoundError as err:
        print("Predictor check:", err)
