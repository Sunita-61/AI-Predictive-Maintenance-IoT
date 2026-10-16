import math
from typing import Any, Dict, Optional

from src.config import ALERT_CRITICAL_THRESHOLD, ALERT_WARNING_THRESHOLD
from src.utils import setup_logger

logger = setup_logger("alert_engine")

# Defined Alert Levels
ALERT_LEVEL_NORMAL = "NORMAL"
ALERT_LEVEL_WARNING = "WARNING"
ALERT_LEVEL_CRITICAL = "CRITICAL"


def evaluate_alert(
    risk_score: float,
    observation: Optional[Dict[str, Any]] = None,
    warning_threshold: float = ALERT_WARNING_THRESHOLD,
    critical_threshold: float = ALERT_CRITICAL_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluate equipment failure risk score and operational telemetry
    to assign an alert tier and actionable recommendations.

    IMPORTANT SCIENTIFIC DISCLAIMER:
    The risk_score produced by the baseline tree ensemble is an uncalibrated
    algorithmic output score (tree voting fraction), not a calibrated empirical
    probability of hardware failure. Thresholds are illustrative and configurable.
    Maintenance actions are student demonstration guidance, not certified industrial safety protocols.

    Args:
        risk_score: Model output score between 0.0 and 1.0.
        observation: Optional raw sensor dictionary for telemetry-driven explanations.
        warning_threshold: Score cut-off for WARNING tier (default 0.30).
        critical_threshold: Score cut-off for CRITICAL tier (default 0.70).

    Returns:
        Dictionary containing:
        - alert_level: 'NORMAL', 'WARNING', or 'CRITICAL'
        - color: Hex color for UI presentation ('#28a745', '#ffc107', '#dc3545')
        - message: Human-readable diagnostic rationale
        - recommended_action: Next step for maintenance personnel
        - is_calibrated: False (always uncalibrated tree votes)
        - is_simulated: bool (reflects observation metadata)
        - maintenance_disclaimer: Educational/portfolio guidance notice
    """
    try:
        score = float(risk_score)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid risk score '{risk_score}': Must be numeric.") from e

    if not math.isfinite(score) or not (0.0 <= score <= 1.0):
        raise ValueError(
            f"Invalid risk score: {risk_score}. Must be a finite number within [0.0, 1.0]."
        )

    if not (0.0 <= warning_threshold <= critical_threshold <= 1.0):
        raise ValueError(
            f"Invalid alert thresholds: Expected 0.0 <= warning ({warning_threshold}) <= "
            f"critical ({critical_threshold}) <= 1.0."
        )

    is_sim = False
    if observation and bool(observation.get("is_simulated", False)):
        is_sim = True

    # Telemetry insights
    notes = []
    if observation:
        tool_wear = observation.get("Tool wear [min]", 0)
        torque = observation.get("Torque [Nm]", 0)
        p_temp = observation.get("Process temperature [K]", 0)
        a_temp = observation.get("Air temperature [K]", 0)

        if tool_wear > 200:
            notes.append(f"Elevated tool wear detected ({tool_wear} min).")
        if torque > 60:
            notes.append(f"High torque load detected ({torque} Nm).")
        if p_temp > 0 and a_temp > 0 and (p_temp - a_temp) < 8.0:
            notes.append(f"Suboptimal heat dissipation delta ({(p_temp - a_temp):.1f} K).")

    notes_str = " " + " ".join(notes) if notes else ""
    sim_prefix = "[SIMULATED TELEMETRY] " if is_sim else ""

    safety_disclaimer = (
        "Demonstration recommendation for educational portfolio; not a certified industrial safety instruction."
    )

    if score >= critical_threshold:
        return {
            "alert_level": ALERT_LEVEL_CRITICAL,
            "color": "#dc3545",  # Danger Red
            "message": f"{sim_prefix}CRITICAL: High algorithmic failure risk score ({score:.2f}).{notes_str}",
            "recommended_action": (
                "Halt machine cycle immediately for diagnostic inspection. "
                "Check cutting insert wear, spindle bearings, and thermal ventilation."
            ),
            "score": score,
            "is_calibrated": False,
            "is_simulated": is_sim,
            "maintenance_disclaimer": safety_disclaimer,
        }
    elif score >= warning_threshold:
        return {
            "alert_level": ALERT_LEVEL_WARNING,
            "color": "#ffc107",  # Warning Amber
            "message": f"{sim_prefix}WARNING: Elevated failure risk score ({score:.2f}).{notes_str}",
            "recommended_action": (
                "Schedule preventative maintenance check within upcoming shift. "
                "Monitor vibration and inspect tool contact surfaces."
            ),
            "score": score,
            "is_calibrated": False,
            "is_simulated": is_sim,
            "maintenance_disclaimer": safety_disclaimer,
        }
    else:
        return {
            "alert_level": ALERT_LEVEL_NORMAL,
            "color": "#28a745",  # Success Green
            "message": f"{sim_prefix}NORMAL: Equipment operating within healthy parameters (Risk score: {score:.2f}).",
            "recommended_action": "Continue regular production monitoring.",
            "score": score,
            "is_calibrated": False,
            "is_simulated": is_sim,
            "maintenance_disclaimer": safety_disclaimer,
        }


if __name__ == "__main__":
    res = evaluate_alert(0.85, {"Tool wear [min]": 220, "Torque [Nm]": 65})
    print("Alert Evaluation Output:\n", res)
