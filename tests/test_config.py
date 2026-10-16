"""
Tests for configuration paths and directory resolution.
"""

from pathlib import Path
from src.config import (
    DATA_DIR,
    DEFAULT_MODEL_FILE,
    DEFAULT_RAW_DATA_FILE,
    DOCS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SRC_DIR,
    ensure_directories,
)


def test_paths_resolve_relative_to_root():
    """Verify that all paths are Path instances and rooted under PROJECT_ROOT."""
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists()
    assert SRC_DIR.is_relative_to(PROJECT_ROOT)
    assert DATA_DIR.is_relative_to(PROJECT_ROOT)
    assert MODELS_DIR.is_relative_to(PROJECT_ROOT)
    assert OUTPUTS_DIR.is_relative_to(PROJECT_ROOT)
    assert DOCS_DIR.is_relative_to(PROJECT_ROOT)


def test_ensure_directories():
    """Verify ensure_directories creates essential directories."""
    ensure_directories()
    assert RAW_DATA_DIR.exists()
    assert MODELS_DIR.exists()
    assert OUTPUTS_DIR.exists()
