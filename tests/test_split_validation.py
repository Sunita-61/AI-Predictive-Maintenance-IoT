"""
Regression tests for train/test split metadata integrity, partition disjointness,
bounds checking, and dataset-hash mismatch detection.
"""

import pytest
from src.preprocessing import validate_split_metadata


def test_split_validation_valid():
    """Verify that a valid split metadata dictionary passes without errors."""
    valid_split = {
        "train_indices": [0, 1, 2, 3],
        "test_indices": [4, 5],
        "train_count": 4,
        "test_count": 2,
        "dataset_sha256": "fakehash123",
    }
    # Should not raise any error
    validate_split_metadata(valid_split, total_rows=6, expected_hash="fakehash123")


def test_split_validation_missing_key_raises_error():
    """Verify that missing required metadata keys raise ValueError."""
    incomplete_split = {
        "train_indices": [0, 1, 2],
        "test_indices": [3],
        # missing train_count, test_count
    }
    with pytest.raises(ValueError, match="Missing required key"):
        validate_split_metadata(incomplete_split, total_rows=4)


def test_split_validation_duplicate_indices_raises_error():
    """Verify that duplicate indices within either partition raise ValueError."""
    split_dup_train = {
        "train_indices": [0, 1, 1],
        "test_indices": [2],
        "train_count": 3,
        "test_count": 1,
    }
    with pytest.raises(ValueError, match="Training partition contains duplicate indices"):
        validate_split_metadata(split_dup_train, total_rows=3)

    split_dup_test = {
        "train_indices": [0, 1],
        "test_indices": [2, 2],
        "train_count": 2,
        "test_count": 2,
    }
    with pytest.raises(ValueError, match="Test partition contains duplicate indices"):
        validate_split_metadata(split_dup_test, total_rows=3)


def test_split_validation_overlapping_partitions_raises_error():
    """Verify that any overlapping index between train and test partitions raises ValueError."""
    overlapping_split = {
        "train_indices": [0, 1, 2],
        "test_indices": [2, 3],
        "train_count": 3,
        "test_count": 2,
    }
    with pytest.raises(ValueError, match="Training and test partitions overlap"):
        validate_split_metadata(overlapping_split, total_rows=4)


def test_split_validation_incomplete_coverage_raises_error():
    """Verify that partitions failing to cover the entire dataset raise ValueError."""
    incomplete_coverage_split = {
        "train_indices": [0, 1],
        "test_indices": [2],
        "train_count": 2,
        "test_count": 1,
    }
    with pytest.raises(ValueError, match="Partition union size .* does not equal total dataset rows"):
        validate_split_metadata(incomplete_coverage_split, total_rows=4)


def test_split_validation_index_out_of_bounds_raises_error():
    """Verify that out-of-bounds indices (negative or >= total_rows) raise ValueError."""
    out_of_bounds_split = {
        "train_indices": [0, 1],
        "test_indices": [2, 10],  # 10 is >= total_rows 4
        "train_count": 2,
        "test_count": 2,
    }
    with pytest.raises(ValueError, match="Indices out of bounds"):
        validate_split_metadata(out_of_bounds_split, total_rows=4)


def test_split_validation_count_mismatch_raises_error():
    """Verify that discrepancy between index length and recorded count raises ValueError."""
    count_mismatch_split = {
        "train_indices": [0, 1],
        "test_indices": [2, 3],
        "train_count": 5,  # Mismatch with len 2
        "test_count": 2,
    }
    with pytest.raises(ValueError, match="Split count mismatch"):
        validate_split_metadata(count_mismatch_split, total_rows=4)


def test_split_validation_hash_mismatch_raises_error():
    """Verify that mismatch between saved split hash and active dataset hash raises ValueError."""
    hash_mismatch_split = {
        "train_indices": [0, 1],
        "test_indices": [2, 3],
        "train_count": 2,
        "test_count": 2,
        "dataset_sha256": "expected_old_hash_abc",
    }
    with pytest.raises(ValueError, match="Dataset provenance mismatch"):
        validate_split_metadata(
            hash_mismatch_split,
            total_rows=4,
            expected_hash="current_new_hash_xyz",
        )
