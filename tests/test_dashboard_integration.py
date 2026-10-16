"""
Tests for dashboard integration, graceful fallbacks, and input validation.
"""

from pathlib import Path
import tempfile
import pytest
from src.database import get_recent_alerts, get_recent_predictions, init_db
from src.predictor import extract_model_features, load_model_artifact, predict_single


def test_dashboard_handles_empty_database():
    """Verify that empty database tables return clean empty lists without exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "empty_pm.db"
        init_db(test_db)
        preds = get_recent_predictions(limit=10, db_path=test_db)
        alerts = get_recent_alerts(limit=10, db_path=test_db)
        assert isinstance(preds, list)
        assert len(preds) == 0
        assert isinstance(alerts, list)
        assert len(alerts) == 0


def test_dashboard_missing_model_handling():
    """Verify that attempting to load a missing model raises a clean FileNotFoundError."""
    missing_path = Path("models/missing_model_test.joblib")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_artifact(missing_path)


def test_dashboard_input_validation_errors():
    """Verify that invalid manual inputs produce immediate schema errors."""
    # Missing required key
    bad_payload_missing = {
        "Type": "M",
        "Air temperature [K]": 300.0,
    }
    with pytest.raises(ValueError, match="missing required sensor features"):
        extract_model_features(bad_payload_missing)

    # Invalid category
    bad_payload_type = {
        "Type": "Z",  # Not in L, M, H
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 120,
    }
    with pytest.raises(ValueError, match="Invalid device Type 'Z'"):
        extract_model_features(bad_payload_type)


def test_dashboard_audit_distinguishes_simulated_from_manual():
    """Verify that database records distinguish simulated telemetry from manual tests."""
    from src.database import log_prediction

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "audit_test.db"
        init_db(test_db)
        obs = {
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 120,
        }
        log_prediction(obs, 0, 0.05, "NORMAL", is_simulated=False, db_path=test_db)
        log_prediction(obs, 1, 0.85, "CRITICAL", is_simulated=True, db_path=test_db)

        records = get_recent_predictions(limit=10, db_path=test_db)
        assert len(records) == 2
        sim_flags = [r["is_simulated"] for r in records]
        assert 0 in sim_flags
        assert 1 in sim_flags

