"""
Tests for IoT telemetry simulator reproducibility and output formatting.
"""

from src.simulator import generate_simulated_reading, generate_simulated_stream


def test_simulation_reproducibility():
    r1 = generate_simulated_reading(device_type="M", scenario="wear_stress", random_seed=123)
    r2 = generate_simulated_reading(device_type="M", scenario="wear_stress", random_seed=123)
    assert r1 == r2
    assert r1["is_simulated"] is True
    assert r1["Type"] == "M"


def test_simulation_stream_generation():
    df_stream = generate_simulated_stream(n_samples=5, random_seed=99)
    assert len(df_stream) == 5
    assert "is_simulated" in df_stream.columns
    assert all(df_stream["is_simulated"])
    assert "Tool wear [min]" in df_stream.columns
    assert "Torque [Nm]" in df_stream.columns
