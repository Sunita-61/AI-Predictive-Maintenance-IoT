"""
Tests for model explainability and feature importance analysis.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest
from src.config import DEFAULT_MODEL_FILE
from src.explainability import (
    analyze_feature_importance,
    extract_tree_feature_importance,
    get_clean_feature_names,
    plot_feature_importance,
)


def test_explainability_on_trained_model():
    """Verify that explainability extraction runs on the serialized pipeline."""
    if not DEFAULT_MODEL_FILE.exists():
        pytest.skip("Trained model not available for explainability test.")

    import joblib

    pipeline = joblib.load(DEFAULT_MODEL_FILE)
    preprocessor = pipeline.named_steps["preprocessor"]

    clean_names = get_clean_feature_names(preprocessor)
    assert len(clean_names) == 11
    assert any("Type" in name for name in clean_names)
    assert any("Torque" in name for name in clean_names)

    df_imp = extract_tree_feature_importance(pipeline)
    assert len(df_imp) == 11
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert "relative_importance_pct" in df_imp.columns

    # Importance values should be non-negative and sum to ~1.0
    assert (df_imp["importance"] >= 0).all()
    assert pytest.approx(df_imp["importance"].sum(), rel=1e-2) == 1.0

    # Ensure sorted in descending order
    assert df_imp["importance"].is_monotonic_decreasing


def test_plot_feature_importance_saves_file():
    """Verify plotting function writes an image to target path."""
    df_mock = pd.DataFrame(
        {
            "feature": ["Torque [Nm]", "Tool wear [min]"],
            "importance": [0.60, 0.40],
            "relative_importance_pct": [60.0, 40.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / "test_importance.png"
        plot_feature_importance(df_mock, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0
