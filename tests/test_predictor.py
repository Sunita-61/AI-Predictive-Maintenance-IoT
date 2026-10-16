"""
Tests for inference predictor schema extraction, input validation,
physical domain bounds, and prediction score constraints.
"""

from pathlib import Path
import pytest
from src.predictor import (
    extract_model_features,
    load_model_artifact,
    predict_single,
    validate_observation,
)


def test_extract_model_features_valid():
    raw_obs = {
        "Type": "m",  # should normalize to 'M'
        "Air temperature [K]": 300.2,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 115,
        "is_simulated": True,  # metadata to filter
        "scenario": "normal",   # metadata to filter
    }

    clean = extract_model_features(raw_obs)
    assert clean["Type"] == "M"
    assert clean["Air temperature [K]"] == 300.2
    assert "is_simulated" not in clean
    assert "scenario" not in clean


def test_extract_model_features_missing_key_raises_error():
    incomplete_obs = {
        "Type": "M",
        "Air temperature [K]": 300.2,
    }
    with pytest.raises(ValueError, match="missing required sensor features"):
        extract_model_features(incomplete_obs)


def test_extract_model_features_invalid_type_raises_error():
    invalid_type_obs = {
        "Type": "X",  # Invalid type
        "Air temperature [K]": 300.2,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 115,
    }
    with pytest.raises(ValueError, match="Invalid device Type 'X'"):
        extract_model_features(invalid_type_obs)


def test_extract_model_features_non_numeric_raises_error():
    obs = {
        "Type": "M",
        "Air temperature [K]": "three_hundred",
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 115,
    }
    with pytest.raises(ValueError, match="Invalid non-numeric sensor value"):
        extract_model_features(obs)


def test_extract_model_features_nan_and_inf_raises_error():
    obs_nan = {
        "Type": "M",
        "Air temperature [K]": float("nan"),
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 115,
    }
    with pytest.raises(ValueError, match="Invalid non-finite sensor value"):
        extract_model_features(obs_nan)

    obs_inf = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": float("inf"),
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 115,
    }
    with pytest.raises(ValueError, match="Invalid non-finite sensor value"):
        extract_model_features(obs_inf)


def test_extract_model_features_physical_limits():
    # Negative Kelvin
    obs_neg_k = {
        "Type": "L",
        "Air temperature [K]": -5.0,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 115,
    }
    with pytest.raises(ValueError, match="Physically impossible value.*strictly positive"):
        extract_model_features(obs_neg_k)

    # Negative Speed
    obs_neg_speed = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": -100,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 115,
    }
    with pytest.raises(ValueError, match="Physically impossible value.*Cannot be negative"):
        extract_model_features(obs_neg_speed)

    # Negative Torque
    obs_neg_torque = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": -10.0,
        "Tool wear [min]": 115,
    }
    with pytest.raises(ValueError, match="Physically impossible value.*Cannot be negative"):
        extract_model_features(obs_neg_torque)

    # Negative Tool Wear
    obs_neg_wear = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": -5,
    }
    with pytest.raises(ValueError, match="Physically impossible value.*Cannot be negative"):
        extract_model_features(obs_neg_wear)


def test_predict_single_with_trained_model():
    obs = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
        "is_simulated": True,
    }
    result = predict_single(obs)
    assert result["predicted_class"] in (0, 1)
    assert isinstance(result["is_failure"], bool)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["is_simulated"] is True
    assert "disclaimer" in result
    assert "score_interpretation" in result


def test_load_model_artifact_missing_raises_error():
    missing_path = Path("models/non_existent_model.joblib")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_artifact(missing_path)


def test_positive_class_score_extraction_dynamic():
    import numpy as np
    from src.predictor import get_positive_class_index

    class MockClassifier:
        def __init__(self, classes):
            self.classes_ = classes

    class MockPipeline:
        def __init__(self, classes):
            self.named_steps = {"classifier": MockClassifier(classes)}

    # Standard binary ordering [0, 1]
    pipe_std = MockPipeline(np.array([0, 1]))
    assert get_positive_class_index(pipe_std) == 1

    # Inverted binary ordering [1, 0]
    pipe_inv = MockPipeline(np.array([1, 0]))
    assert get_positive_class_index(pipe_inv) == 0

    # Direct attribute fallback
    clf_direct = MockClassifier(np.array([1, 0]))
    assert get_positive_class_index(clf_direct) == 0

