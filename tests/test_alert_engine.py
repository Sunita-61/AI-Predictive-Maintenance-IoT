"""
Tests for alert engine thresholds, tier assignments, boundary conditions,
non-finite score rejection, and simulated telemetry labeling.
"""

import pytest
from src.alert_engine import (
    ALERT_LEVEL_CRITICAL,
    ALERT_LEVEL_NORMAL,
    ALERT_LEVEL_WARNING,
    evaluate_alert,
)


def test_alert_tiers():
    # Normal tier test (< 0.30)
    res_normal = evaluate_alert(0.15)
    assert res_normal["alert_level"] == ALERT_LEVEL_NORMAL
    assert res_normal["color"] == "#28a745"

    # Warning tier test (0.30 <= score < 0.70)
    res_warning = evaluate_alert(0.45)
    assert res_warning["alert_level"] == ALERT_LEVEL_WARNING
    assert res_warning["color"] == "#ffc107"

    # Critical tier test (>= 0.70)
    res_critical = evaluate_alert(0.85)
    assert res_critical["alert_level"] == ALERT_LEVEL_CRITICAL
    assert res_critical["color"] == "#dc3545"


def test_alert_boundary_conditions():
    # Exact 0.0 boundary
    res_zero = evaluate_alert(0.0)
    assert res_zero["alert_level"] == ALERT_LEVEL_NORMAL

    # Exact warning threshold boundary (0.30)
    res_warn_bound = evaluate_alert(0.30)
    assert res_warn_bound["alert_level"] == ALERT_LEVEL_WARNING

    # Just below warning threshold (0.2999)
    res_just_below_warn = evaluate_alert(0.2999)
    assert res_just_below_warn["alert_level"] == ALERT_LEVEL_NORMAL

    # Exact critical threshold boundary (0.70)
    res_crit_bound = evaluate_alert(0.70)
    assert res_crit_bound["alert_level"] == ALERT_LEVEL_CRITICAL

    # Just below critical threshold (0.6999)
    res_just_below_crit = evaluate_alert(0.6999)
    assert res_just_below_crit["alert_level"] == ALERT_LEVEL_WARNING

    # Exact 1.0 boundary
    res_one = evaluate_alert(1.0)
    assert res_one["alert_level"] == ALERT_LEVEL_CRITICAL


def test_alert_invalid_scores_raise_error():
    # Negative score
    with pytest.raises(ValueError, match="Must be a finite number within"):
        evaluate_alert(-0.01)

    # Above 1.0
    with pytest.raises(ValueError, match="Must be a finite number within"):
        evaluate_alert(1.05)

    # NaN
    with pytest.raises(ValueError, match="Must be a finite number within"):
        evaluate_alert(float("nan"))

    # Inf
    with pytest.raises(ValueError, match="Must be a finite number within"):
        evaluate_alert(float("inf"))

    # Non-numeric
    with pytest.raises(ValueError, match="Must be numeric"):
        evaluate_alert("not_a_number")  # type: ignore


def test_alert_invalid_threshold_configuration_raises_error():
    # warning threshold higher than critical threshold
    with pytest.raises(ValueError, match="Invalid alert thresholds"):
        evaluate_alert(0.5, warning_threshold=0.8, critical_threshold=0.4)


def test_alert_with_telemetry_diagnostics_and_simulation():
    obs_sim = {
        "Tool wear [min]": 230,
        "Torque [Nm]": 68.0,
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 304.0,  # Low delta T
        "is_simulated": True,
    }
    res = evaluate_alert(0.78, observation=obs_sim)
    assert res["alert_level"] == ALERT_LEVEL_CRITICAL
    assert res["is_simulated"] is True
    assert "[SIMULATED TELEMETRY]" in res["message"]
    assert "maintenance_disclaimer" in res
    assert "tool wear" in res["message"].lower()
    assert "torque" in res["message"].lower()
    assert "heat dissipation" in res["message"].lower()


def test_alert_semantic_separation_from_classifier():
    """
    Verify that operational alert levels are semantically distinct from binary classification.
    A score of 0.45 is below the default binary threshold of 0.50, yet triggers WARNING tier
    for preventative operational inspection.
    """
    res_sub_threshold = evaluate_alert(0.45)
    assert res_sub_threshold["alert_level"] == ALERT_LEVEL_WARNING
    assert res_sub_threshold["is_calibrated"] is False
    assert "maintenance_disclaimer" in res_sub_threshold

