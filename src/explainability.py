"""
Model explainability module for global feature importance diagnostics.

Extracts Mean Decrease in Impurity (Gini importance) from the trained Random Forest
pipeline and optionally calculates Permutation Importance on held-out test data.
Maps one-hot encoded and engineered features back to clean human-readable names.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import (
    DEFAULT_FEATURE_IMPORTANCE_FILE,
    DEFAULT_FEATURE_IMPORTANCE_PLOT,
    DEFAULT_MODEL_FILE,
    DEFAULT_RAW_DATA_FILE,
    DEFAULT_SPLIT_FILE,
    RANDOM_STATE,
    ensure_directories,
)
from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import clean_dataset, split_features_and_target, validate_split_metadata
from src.utils import get_timestamp, load_json, setup_logger

logger = setup_logger("explainability")


def get_clean_feature_names(preprocessor) -> List[str]:
    """
    Extract human-readable feature names from the fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer instance.

    Returns:
        List of formatted feature name strings.
    """
    try:
        raw_names = preprocessor.get_feature_names_out()
        clean_names = []
        for name in raw_names:
            # Format OneHotEncoder output, e.g. cat__Type_H -> Type: High Variant (H)
            if name.startswith("cat__"):
                clean = name.replace("cat__", "").replace("_", ": ")
            elif name.startswith("num__"):
                clean = name.replace("num__", "")
            else:
                clean = name
            clean_names.append(clean)
        return clean_names
    except Exception as err:
        logger.warning(f"Could not retrieve get_feature_names_out ({err}). Falling back to column mapping.")
        return [f"Feature_{i}" for i in range(11)]


def extract_tree_feature_importance(pipeline) -> pd.DataFrame:
    """
    Extract Gini importance (Mean Decrease in Impurity) from the Random Forest model.

    Args:
        pipeline: Fitted Scikit-Learn Pipeline containing 'preprocessor' and 'classifier'.

    Returns:
        Sorted DataFrame with columns ['feature', 'importance', 'relative_importance_pct'].
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = get_clean_feature_names(preprocessor)
    importances = classifier.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature name count ({len(feature_names)}) does not match "
            f"classifier feature_importances_ count ({len(importances)})."
        )

    if not np.all(np.isfinite(importances)):
        raise ValueError("Model feature_importances_ contains non-finite values.")

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    df_imp["relative_importance_pct"] = (
        (df_imp["importance"] / df_imp["importance"].sum()) * 100.0
    ).round(2)
    df_imp.sort_values(by="importance", ascending=False, inplace=True)
    df_imp.reset_index(drop=True, inplace=True)
    return df_imp


def compute_permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Calculate permutation feature importance on held-out test data.

    Note: Evaluates drop in ROC-AUC or PR-AUC when each feature column is permuted.

    Args:
        pipeline: Fitted model pipeline.
        X_test: Held-out test features DataFrame.
        y_test: Held-out test target Series.
        n_repeats: Number of permutations per feature.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with permutation importance mean and standard deviation.
    """
    logger.info("Computing permutation importance on held-out test data...")
    res = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=None,
    )

    df_perm = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "perm_importance_mean": res.importances_mean,
            "perm_importance_std": res.importances_std,
        }
    ).sort_values(by="perm_importance_mean", ascending=False)
    df_perm.reset_index(drop=True, inplace=True)
    return df_perm


def plot_feature_importance(df_imp: pd.DataFrame, output_path: Path) -> None:
    """
    Generate and save a horizontal bar chart of feature importances.

    Args:
        df_imp: DataFrame with 'feature' and 'importance'.
        output_path: Target image file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.barplot(
        data=df_imp,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )

    ax.set_title("Random Forest Global Feature Importance (MDI / Gini)", fontsize=13, weight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
    ax.set_ylabel("Engineered & Sensor Features", fontsize=11)

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def analyze_feature_importance(
    model_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    split_path: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    output_png: Optional[Path] = None,
    include_permutation: bool = True,
) -> Dict[str, Any]:
    """
    Execute full feature explainability workflow:
    1. Extract tree Gini importances.
    2. Map to human-readable names.
    3. Calculate test set permutation importances if test split is available.
    4. Save metrics table to CSV and visualization plot to PNG.

    Args:
        model_path: Path to serialized model pipeline.
        dataset_path: Path to raw dataset CSV.
        split_path: Path to train_test_split.json indices file.
        output_csv: Target path for feature importance CSV.
        output_png: Target path for importance plot PNG.
        include_permutation: Whether to compute permutation importance on held-out test data.

    Returns:
        Dictionary containing summary tables, top features, and scientific disclaimer.
    """
    ensure_directories()
    m_path = model_path or DEFAULT_MODEL_FILE
    d_path = dataset_path or DEFAULT_RAW_DATA_FILE
    s_path = split_path or DEFAULT_SPLIT_FILE
    out_csv = output_csv or DEFAULT_FEATURE_IMPORTANCE_FILE
    out_png = output_png or DEFAULT_FEATURE_IMPORTANCE_PLOT

    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {m_path}. Run training first.")

    pipeline = joblib.load(m_path)
    df_tree = extract_tree_feature_importance(pipeline)

    # Save Gini importance table
    df_tree.to_csv(out_csv, index=False)
    logger.info(f"Feature importance table saved to: {out_csv}")

    # Generate visualization plot
    plot_feature_importance(df_tree, out_png)
    logger.info(f"Feature importance chart saved to: {out_png}")

    perm_dict = {}
    if include_permutation and d_path.exists():
        try:
            df_raw = load_dataset(d_path)
            df_clean = clean_dataset(df_raw)
            df_feat = engineer_features(df_clean)
            X, y = split_features_and_target(df_feat)

            if s_path.exists():
                split_info = load_json(s_path)
                validate_split_metadata(split_info, total_rows=len(X))
                test_idx = split_info["test_indices"]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                df_perm = compute_permutation_importance(pipeline, X_test, y_test)
                perm_dict = df_perm.to_dict(orient="records")
            else:
                logger.warning(
                    f"Persisted split file not found at {s_path}. "
                    f"Skipping permutation importance to preserve split integrity."
                )
        except Exception as e:
            logger.warning(f"Permutation importance skipped: {e}")

    top_feature = df_tree.iloc[0]["feature"]
    top_importance = float(df_tree.iloc[0]["importance"])

    return {
        "analyzed_at": get_timestamp(),
        "top_feature": top_feature,
        "top_importance": round(top_importance, 4),
        "tree_importances": df_tree.to_dict(orient="records"),
        "permutation_importances": perm_dict,
        "csv_path": str(out_csv),
        "plot_path": str(out_png),
        "disclaimer": (
            "Feature importance measures mathematical reliance in the trained Random Forest "
            "ensemble (reduction in split Gini impurity). It is not proof of physical causation."
        ),
    }


if __name__ == "__main__":
    try:
        res = analyze_feature_importance()
        print("Feature importance analysis complete. Top feature:", res["top_feature"])
    except Exception as err:
        print("Explainability check:", err)
