# Jupyter Notebooks Guidelines

## Purpose

The `notebooks/` directory is reserved for exploratory data analysis (EDA), interactive feature experimentation, visualization prototyping, and research notes.

---

## Architecture Principle: Separation of Research and Production

To maintain a professional, production-ready codebase:

1. **Exploration in Notebooks, Logic in `src/`:**
   - Use notebooks to inspect distributions, explore correlations, and prototype algorithms.
   - Once a preprocessing function, feature transform, or evaluation technique is validated, refactor it into modular Python code within the `src/` directory.

2. **No Dependency on Notebook State:**
   - Production scripts (`main.py`, `app.py`) must never depend on or import from `.ipynb` files.
   - All critical workflows (data loading, preprocessing, model training, evaluation, inference) must be fully executable from the command-line interface or test suite.

3. **Notebook Reproducibility:**
   - Clear all outputs and restart the kernel before committing notebooks to version control.
   - Import configuration paths directly from `src.config` to avoid broken hardcoded local paths:
     ```python
     import sys
     from pathlib import Path
     sys.path.append(str(Path.cwd().parent))

     from src.config import DEFAULT_RAW_DATA_FILE
     from src.data_loader import load_dataset
     ```

---

## Suggested Notebook Structure

- `01_exploratory_data_analysis.ipynb`: Class distributions, correlation matrices, outlier analysis.
- `02_feature_engineering_experiments.ipynb`: Prototype temperature differences, power formulas, and interaction terms.
- `03_model_prototyping.ipynb`: Compare baseline classifiers and evaluate precision-recall curves.
