# Testing Strategy & Guidelines

## Overview

This project uses `pytest` for automated unit, validation, and integration testing.

---

## Test Directory Structure

```
tests/
├── __init__.py
├── README.md                  # This testing documentation
├── test_config.py             # Validates path resolution and directory creation
├── test_data_loader.py        # Validates schema checking and missing file handling
├── test_preprocessing.py      # Validates leakage column exclusion and transformations
├── test_feature_engineering.py# Validates feature derivation consistency
├── test_alert_engine.py       # Validates alert thresholds and classification
└── test_database.py           # Validates SQLite table creation and parameterized queries
```

---

## Running the Tests

To run the complete test suite:
```bash
pytest
```

To run with verbose output and test coverage:
```bash
pytest -v
```

To run a specific test module:
```bash
pytest tests/test_config.py
```

---

## Testing Principles

1. **Deterministic Execution:**
   - Tests must pass reliably without relying on external network requests or physical hardware.
2. **Graceful Handling of Optional Data:**
   - Tests requiring the real raw dataset (`ai4i2020.csv`) must skip gracefully with a clear informative message if the user has not yet placed the dataset into `data/raw/`.
3. **Data Leakage Safeguards:**
   - Dedicated unit tests must verify that target leakage columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are never included in model feature sets.
