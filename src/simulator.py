"""
IoT Telemetry Simulator module.

Generates realistic, physically plausible synthetic sensor readings for software
demonstrations and testing. All generated records are clearly flagged as simulated.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("simulator")


def generate_simulated_reading(
    device_type: Optional[str] = None,
    scenario: str = "normal",
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a single synthetic IoT telemetry reading for software demonstration.

    Args:
        device_type: Machine quality variant ('L', 'M', or 'H'). Random if None.
        scenario: Operational scenario ('normal', 'wear_stress', 'thermal_stress', 'overstrain').
        random_seed: Optional integer seed for reproducibility.

    Returns:
        Dictionary of raw sensor features plus an explicit 'is_simulated' indicator.
    """
    rng = np.random.default_rng(random_seed)

    if device_type not in ["L", "M", "H"]:
        device_type = rng.choice(["L", "M", "H"], p=[0.50, 0.30, 0.20])

    # Baseline typical operating conditions
    air_temp = float(rng.normal(loc=300.0, scale=2.0))
    process_temp = float(air_temp + rng.normal(loc=10.0, scale=1.0))
    rotational_speed = int(rng.normal(loc=1530.0, scale=150.0))
    torque = float(rng.normal(loc=40.0, scale=8.0))
    tool_wear = int(rng.uniform(low=10, high=120))

    # Apply scenario perturbations
    if scenario == "wear_stress":
        tool_wear = int(rng.uniform(low=190, high=245))
        torque += float(rng.uniform(low=8.0, high=18.0))
    elif scenario == "thermal_stress":
        process_temp = float(air_temp + rng.uniform(low=1.0, high=6.0))  # Low delta T
        rotational_speed = int(rng.uniform(low=1200, high=1380))
    elif scenario == "overstrain":
        torque = float(rng.uniform(low=58.0, high=74.0))
        tool_wear = int(rng.uniform(low=160, high=230))
        rotational_speed = int(rng.uniform(low=1250, high=1450))

    # Clamp to realistic physical bounds
    air_temp = round(max(293.0, min(306.0, air_temp)), 1)
    process_temp = round(max(air_temp + 1.0, min(316.0, process_temp)), 1)
    rotational_speed = int(max(1100, min(2900, rotational_speed)))
    torque = round(max(5.0, min(80.0, torque)), 1)
    tool_wear = int(max(0, min(260, tool_wear)))

    return {
        "Type": device_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        # Mandatory transparency flag: Never represent synthetic data as real telemetry
        "is_simulated": True,
        "scenario": scenario,
    }


def generate_simulated_stream(
    n_samples: int = 10,
    scenario: str = "mixed",
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a series of synthetic telemetry readings for streaming demonstration.

    Args:
        n_samples: Number of synthetic observations to generate.
        scenario: 'normal', 'wear_stress', 'overstrain', or 'mixed'.
        random_seed: Optional seed for reproducible streaming sequences.

    Returns:
        Pandas DataFrame containing labeled synthetic sensor readings.
    """
    rng = np.random.default_rng(random_seed)
    scenarios = ["normal", "wear_stress", "thermal_stress", "overstrain"]
    weights = [0.70, 0.10, 0.10, 0.10]

    records = []
    for _ in range(n_samples):
        current_scenario = scenario if scenario != "mixed" else rng.choice(scenarios, p=weights)
        seed_i = int(rng.integers(1, 1_000_000))
        rec = generate_simulated_reading(scenario=current_scenario, random_seed=seed_i)
        records.append(rec)

    return pd.DataFrame(records)


if __name__ == "__main__":
    sample = generate_simulated_reading(scenario="overstrain", random_seed=42)
    print("Synthetic IoT telemetry sample:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
