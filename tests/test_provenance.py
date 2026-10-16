"""
Tests for artifact version consistency, dataset and model SHA-256 hashing,
and stale or incompatible metadata detection.
"""

import tempfile
from pathlib import Path
import pytest
from src.utils import get_file_hash, save_json, verify_artifact_provenance


def test_artifact_provenance_active_system_consistent():
    """Verify that current repository artifacts pass provenance verification."""
    result = verify_artifact_provenance()
    assert result["is_consistent"] is True
    assert result["mismatches"] == []
    assert isinstance(result["dataset_sha256"], str)
    assert isinstance(result["model_sha256"], str)


def test_artifact_provenance_detects_dataset_hash_mismatch():
    """Verify that a modified dataset triggers provenance mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        d_file = tmp_path / "raw.csv"
        m_file = tmp_path / "model.joblib"
        meta_file = tmp_path / "meta.json"
        s_file = tmp_path / "split.json"

        d_file.write_text("dummy dataset", encoding="utf-8")
        m_file.write_text("dummy model", encoding="utf-8")

        d_hash = get_file_hash(d_file)
        m_hash = get_file_hash(m_file)

        # Write metadata with mismatched dataset hash
        save_json({"dataset_sha256": "wrong_hash_123", "model_sha256": m_hash}, meta_file)
        save_json({"dataset_sha256": d_hash}, s_file)

        res = verify_artifact_provenance(
            model_path=m_file,
            dataset_path=d_file,
            metadata_path=meta_file,
            split_path=s_file,
        )
        assert res["is_consistent"] is False
        assert any("Metadata dataset hash mismatch" in m for m in res["mismatches"])

        # Check raise_on_error
        with pytest.raises(ValueError, match="Artifact provenance verification failed"):
            verify_artifact_provenance(
                model_path=m_file,
                dataset_path=d_file,
                metadata_path=meta_file,
                split_path=s_file,
                raise_on_error=True,
            )


def test_artifact_provenance_detects_model_hash_mismatch():
    """Verify that replacing the model binary without updating metadata triggers provenance error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        d_file = tmp_path / "raw.csv"
        m_file = tmp_path / "model.joblib"
        meta_file = tmp_path / "meta.json"
        s_file = tmp_path / "split.json"

        d_file.write_text("dataset", encoding="utf-8")
        m_file.write_text("model_v2", encoding="utf-8")

        d_hash = get_file_hash(d_file)

        # Meta contains hash of model_v1
        save_json({"dataset_sha256": d_hash, "model_sha256": "hash_of_model_v1"}, meta_file)
        save_json({"dataset_sha256": d_hash}, s_file)

        res = verify_artifact_provenance(
            model_path=m_file,
            dataset_path=d_file,
            metadata_path=meta_file,
            split_path=s_file,
        )
        assert res["is_consistent"] is False
        assert any("Metadata model hash mismatch" in m for m in res["mismatches"])


def test_artifact_provenance_missing_files():
    """Verify missing files are reported in mismatches list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        res = verify_artifact_provenance(
            model_path=tmp_path / "missing_model.joblib",
            dataset_path=tmp_path / "missing_data.csv",
            metadata_path=tmp_path / "missing_meta.json",
            split_path=tmp_path / "missing_split.json",
        )
        assert res["is_consistent"] is False
        assert len(res["mismatches"]) >= 3
