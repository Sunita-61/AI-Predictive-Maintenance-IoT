"""
Command-Line Interface for the AI-Powered Predictive Maintenance System.

Provides commands to inspect the workspace, initialize databases,
train models, run evaluations, execute predictions, and test simulations.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_engine import evaluate_alert
from src.config import (
    DEFAULT_DB_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_RAW_DATA_FILE,
    ensure_directories,
)
from src.database import init_db, log_prediction
from src.predictor import predict_single
from src.simulator import generate_simulated_reading, generate_simulated_stream
from src.utils import setup_logger

logger = setup_logger("cli")


def cmd_check(args: argparse.Namespace) -> None:
    """Validate project directories and verify dataset availability and schema quality."""
    ensure_directories()
    print("=" * 60)
    print("AI-Powered Predictive Maintenance: Environment & Schema Check")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")

    data_file = DEFAULT_RAW_DATA_FILE
    if data_file.exists():
        from src.data_loader import audit_dataset_quality, format_audit_report, load_dataset

        try:
            df = load_dataset(data_file)
            audit = audit_dataset_quality(df)
            print(f"[OK] Raw dataset found at: {data_file}")
            print(format_audit_report(audit))
        except Exception as err:
            print(f"[ERROR] Failed to audit dataset: {err}")
    else:
        print(f"[STATUS: MISSING DATASET] File not found at: {data_file}")
        print("Dataset Acquisition Instructions:")
        print("  1. Visit the official UCI Machine Learning Repository:")
        print("     https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset")
        print("  2. Download 'ai4i2020.csv'.")
        print("  3. Place it in the directory:")
        print(f"     {DEFAULT_RAW_DATA_FILE}")
        print("  4. Rerun 'python main.py check' to validate schema.")

    print("\nModel Artifact Status:")
    model_file = DEFAULT_MODEL_FILE
    if model_file.exists():
        print(f"[OK] Trained model artifact found at: {model_file}")
    else:
        print(f"[INFO] No trained model found at: {model_file}")
        print("       Run 'python main.py train' once the raw dataset is acquired.")

    print("\nDatabase Status:")
    db_file = DEFAULT_DB_FILE
    if db_file.exists():
        print(f"[OK] SQLite database found at: {db_file}")
    else:
        print(f"[INFO] SQLite database will be initialized at: {db_file}")

    print("\nArtifact Provenance Status:")
    from src.utils import verify_artifact_provenance
    prov = verify_artifact_provenance()
    if prov["is_consistent"]:
        print("[OK] All artifact cross-provenance verified (Dataset & Model SHA-256 match across metadata and metrics).")
    else:
        print("[WARNING] Provenance discrepancy detected:")
        for m in prov["mismatches"]:
            print(f"  - {m}")
        print("  Suggested fix: Run 'python main.py train && python main.py evaluate && python main.py explain'.")
    print("=" * 60)


def cmd_init_db(args: argparse.Namespace) -> None:
    """Initialize SQLite database schema."""
    init_db()
    print(f"Database successfully initialized at: {DEFAULT_DB_FILE}")


def cmd_train(args: argparse.Namespace) -> None:
    """Execute training pipeline."""
    from src.train_model import train

    print("Initiating model training...")
    try:
        results = train()
        print("[SUCCESS] Model trained and saved successfully.")
        print(f"Artifact: {DEFAULT_MODEL_FILE}")
        meta = results["metadata"]
        print(f"Training samples: {meta['train_samples']}, Test samples: {meta['test_samples']}")
    except FileNotFoundError as err:
        print(f"[BLOCKED] Cannot train model: {err}")
        sys.exit(1)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute evaluation diagnostics."""
    from src.evaluate_model import evaluate_model

    print("Initiating model evaluation...")
    try:
        metrics = evaluate_model()
        print("=" * 50)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Positive Class:          {metrics['target_positive_class']} (Failure)")
        print(f"Averaging Strategy:      {metrics['metric_averaging_strategy']}")
        print(f"Accuracy:                {metrics['accuracy']:.4f} (Baseline naive accuracy is 96.6%)")
        print(f"Precision:               {metrics['precision']:.4f}")
        print(f"Recall:                  {metrics['recall']:.4f}")
        print(f"F1-Score:                {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:                 {metrics['roc_auc']:.4f}")
        print(f"PR-AUC (Avg Precision):  {metrics['pr_auc']:.4f}")
        print("Confusion Matrix [[TN, FP], [FN, TP]]:", metrics["confusion_matrix"])
        print("=" * 50)
    except Exception as err:
        print(f"[BLOCKED] Evaluation failed: {err}")
        sys.exit(1)


def cmd_predict(args: argparse.Namespace) -> None:
    """Execute prediction on a single observation."""
    obs = {
        "Type": args.type,
        "Air temperature [K]": args.air_temp,
        "Process temperature [K]": args.process_temp,
        "Rotational speed [rpm]": args.rpm,
        "Torque [Nm]": args.torque,
        "Tool wear [min]": args.tool_wear,
    }
    try:
        pred = predict_single(obs)
        alert = evaluate_alert(pred["risk_score"], obs)

        print("=" * 50)
        print("EQUIPMENT FAILURE PREDICTION RESULT")
        print("=" * 50)
        print(f"Prediction:         {'FAILURE (1)' if pred['is_failure'] else 'NORMAL (0)'}")
        print(f"Risk Score:         {pred['risk_score']:.4f} (Uncalibrated tree ensemble score)")
        print(f"Alert Level:        {alert['alert_level']}")
        print(f"Diagnostic:         {alert['message']}")
        print(f"Recommended Action: {alert['recommended_action']}")
        print("=" * 50)

        if args.log:
            log_prediction(
                features=obs,
                predicted_failure=pred["predicted_class"],
                prediction_score=pred["risk_score"],
                alert_level=alert["alert_level"],
                is_simulated=False,
            )
            print("[INFO] Prediction logged to SQLite database.")
    except Exception as err:
        print(f"[BLOCKED] Prediction failed: {err}")
        sys.exit(1)


def cmd_simulate(args: argparse.Namespace) -> None:
    """Generate and display simulated sensor telemetry."""
    print("=" * 60)
    print("SIMULATED IoT TELEMETRY DEMONSTRATION")
    print("Disclaimer: Software-simulated readings. Not physical hardware.")
    print("=" * 60)

    if args.stream > 1:
        stream_df = generate_simulated_stream(n_samples=args.stream, random_seed=args.seed)
        print(stream_df.to_string(index=False))
    else:
        sample = generate_simulated_reading(scenario=args.scenario, random_seed=args.seed)
        for k, v in sample.items():
            print(f"  {k}: {v}")

        # If model is present, evaluate simulation through predictor
        if DEFAULT_MODEL_FILE.exists():
            try:
                pred = predict_single(sample)
                alert = evaluate_alert(pred["risk_score"], sample)
                print("-" * 60)
                print(f"Simulated Telemetry Prediction: Risk Score = {pred['risk_score']:.4f} | Alert = {alert['alert_level']}")
                print(f"Action: {alert['recommended_action']}")
            except Exception as e:
                logger.debug(f"Simulation prediction evaluation skipped: {e}")
    print("=" * 60)


def cmd_explain(args: argparse.Namespace) -> None:
    """Execute feature importance and explainability analysis."""
    from src.explainability import analyze_feature_importance

    print("Initiating model explainability analysis...")
    try:
        res = analyze_feature_importance()
        print("=" * 50)
        print("MODEL EXPLAINABILITY SUMMARY")
        print("=" * 50)
        print(f"Top Feature by Importance: {res['top_feature']} ({res['top_importance']:.4f})")
        print(f"Importance CSV exported to: {res['csv_path']}")
        print(f"Importance plot saved to:   {res['plot_path']}")
        print("Top 5 Features by Gini Importance:")
        for idx, row in enumerate(res['tree_importances'][:5], 1):
            print(f"  {idx}. {row['feature']}: {row['importance']:.4f} ({row['relative_importance_pct']}%)")
        print("-" * 50)
        print("Disclaimer:", res['disclaimer'])
        print("=" * 50)
    except Exception as err:
        print(f"[BLOCKED] Explainability analysis failed: {err}")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI command parser."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Predictive Maintenance System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: check
    p_check = subparsers.add_parser("check", help="Verify workspace directories and dataset setup.")
    p_check.set_defaults(func=cmd_check)

    # Command: init-db
    p_init = subparsers.add_parser("init-db", help="Initialize SQLite database schema.")
    p_init.set_defaults(func=cmd_init_db)

    # Command: train
    p_train = subparsers.add_parser("train", help="Train baseline RandomForest classifier.")
    p_train.set_defaults(func=cmd_train)

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model and generate performance metrics.")
    p_eval.set_defaults(func=cmd_evaluate)

    # Command: explain
    p_exp = subparsers.add_parser("explain", help="Run feature importance and explainability analysis.")
    p_exp.set_defaults(func=cmd_explain)

    # Command: predict
    p_pred = subparsers.add_parser("predict", help="Generate prediction on single machine state.")
    p_pred.add_argument("--type", choices=["L", "M", "H"], default="M", help="Product quality variant")
    p_pred.add_argument("--air-temp", type=float, default=300.0, help="Air temperature [K]")
    p_pred.add_argument("--process-temp", type=float, default=310.0, help="Process temperature [K]")
    p_pred.add_argument("--rpm", type=float, default=1500.0, help="Rotational speed [rpm]")
    p_pred.add_argument("--torque", type=float, default=40.0, help="Torque [Nm]")
    p_pred.add_argument("--tool-wear", type=float, default=120.0, help="Tool wear [min]")
    p_pred.add_argument("--log", action="store_true", help="Log prediction to SQLite database")
    p_pred.set_defaults(func=cmd_predict)

    # Command: simulate
    p_sim = subparsers.add_parser("simulate", help="Generate simulated sensor telemetry.")
    p_sim.add_argument("--scenario", choices=["normal", "wear_stress", "thermal_stress", "overstrain"], default="normal")
    p_sim.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p_sim.add_argument("--stream", type=int, default=1, help="Number of telemetry events to generate")
    p_sim.set_defaults(func=cmd_simulate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
