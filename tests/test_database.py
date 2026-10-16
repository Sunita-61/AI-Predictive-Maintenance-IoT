"""
Tests for SQLite database initialization, query safety, persistence,
and empty-table handling.
"""

import tempfile
from pathlib import Path
from src.database import (
    get_recent_alerts,
    get_recent_predictions,
    init_db,
    log_alert,
    log_prediction,
)


def test_database_initialization_and_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_pm.db"
        init_db(test_db)
        assert test_db.exists()

        # Log prediction
        features = {
            "Type": "L",
            "Air temperature [K]": 300.2,
            "Process temperature [K]": 310.4,
            "Rotational speed [rpm]": 1490,
            "Torque [Nm]": 42.5,
            "Tool wear [min]": 105,
        }
        pred_id = log_prediction(
            features=features,
            predicted_failure=0,
            prediction_score=0.12,
            alert_level="NORMAL",
            is_simulated=True,
            db_path=test_db,
        )
        assert pred_id > 0

        # Log alert
        alert_id = log_alert(
            alert_level="CRITICAL",
            risk_score=0.88,
            message="Test high risk alert",
            is_simulated=True,
            db_path=test_db,
        )
        assert alert_id > 0

        # Fetch recent records
        preds = get_recent_predictions(limit=10, db_path=test_db)
        assert len(preds) == 1
        assert preds[0]["device_type"] == "L"
        assert preds[0]["is_simulated"] == 1

        alerts = get_recent_alerts(limit=10, db_path=test_db)
        assert len(alerts) == 1
        assert alerts[0]["alert_level"] == "CRITICAL"


def test_database_empty_tables_return_empty_lists():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_empty.db"
        init_db(test_db)

        preds = get_recent_predictions(limit=10, db_path=test_db)
        alerts = get_recent_alerts(limit=10, db_path=test_db)

        assert preds == []
        assert alerts == []


def test_database_parameterized_query_safety():
    """Verify parameterized insertion handles malicious SQL injection strings safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_inject.db"
        init_db(test_db)

        # Injection payload in string field
        malicious_message = "Normal'; DROP TABLE alerts; --"
        alert_id = log_alert(
            alert_level="WARNING",
            risk_score=0.45,
            message=malicious_message,
            is_simulated=False,
            db_path=test_db,
        )
        assert alert_id > 0

        # Table must still exist and record stored literally
        alerts = get_recent_alerts(limit=5, db_path=test_db)
        assert len(alerts) == 1
        assert alerts[0]["message"] == malicious_message
