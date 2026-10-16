"""
Configuration module for the AI Predictive Maintenance System.

Centralizes paths, model hyperparameters, column definitions, and alert thresholds.
All paths are resolved dynamically relative to the project root directory.
"""

from pathlib import Path
import os

# ==========================================
# Root and Directory Paths
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

DOCS_DIR = PROJECT_ROOT / "docs"
IMAGES_DIR = PROJECT_ROOT / "images"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# ==========================================
# Default File Locations
# ==========================================
DEFAULT_RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"
DEFAULT_PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_ai4i2020.csv"
DEFAULT_MODEL_FILE = MODELS_DIR / "predictive_maintenance_rf.joblib"
DEFAULT_DB_FILE = DATA_DIR / "predictive_maintenance.db"
DEFAULT_METRICS_FILE = METRICS_DIR / "evaluation_metrics.json"
DEFAULT_CONFUSION_MATRIX_FILE = PLOTS_DIR / "confusion_matrix.png"
DEFAULT_SPLIT_FILE = MODELS_DIR / "train_test_split.json"
DEFAULT_FEATURE_IMPORTANCE_FILE = METRICS_DIR / "feature_importance.csv"
DEFAULT_FEATURE_IMPORTANCE_PLOT = PLOTS_DIR / "feature_importance.png"

# ==========================================
# Dataset Column Specifications
# ==========================================
# Primary binary classification target
TARGET_COLUMN = "Machine failure"

# Columns that represent direct failure modes resulting in target leakage
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Non-predictive unique identifiers to be excluded
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]

# Expected raw input features used for modeling
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ALL_RAW_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Engineered feature names created in feature_engineering.py
ENGINEERED_FEATURES = [
    "Temp_Difference_K",
    "Power_Proxy_W",
    "Overstrain_Proxy",
]

# Complete set of expected columns in the raw UCI dataset
EXPECTED_RAW_COLUMNS = (
    IDENTIFIER_COLUMNS
    + CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [TARGET_COLUMN]
    + LEAKAGE_COLUMNS
)

# ==========================================
# Machine Learning Configuration
# ==========================================
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = 0.20
N_ESTIMATORS = 150
MAX_DEPTH = 12

# ==========================================
# Alert Engine Configuration
# ==========================================
# Note: Scores produced by default tree ensembles are algorithmic risk scores,
# not calibrated real-world probabilities. These thresholds are illustrative defaults.
ALERT_WARNING_THRESHOLD = float(os.getenv("ALERT_WARNING_THRESHOLD", "0.30"))
ALERT_CRITICAL_THRESHOLD = float(os.getenv("ALERT_CRITICAL_THRESHOLD", "0.70"))


def ensure_directories() -> None:
    """Ensure all required runtime directories exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        PREDICTIONS_DIR,
        IMAGES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    ensure_directories()
    print("All project directories verified.")
