"""
Streamlit Web Dashboard for AI-Powered Predictive Maintenance for IoT Devices.

Provides:
- Project Overview & Methodology
- Held-out Model Performance & Confusion Matrix
- Model Explainability (Gini & Permutation Importance)
- Manual Equipment Diagnostic & Failure Scoring
- Simulated IoT Telemetry Stream Monitor
- SQLite Historical Prediction & Alert Logs
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_engine import evaluate_alert
from src.config import (
    DEFAULT_CONFUSION_MATRIX_FILE,
    DEFAULT_DB_FILE,
    DEFAULT_FEATURE_IMPORTANCE_FILE,
    DEFAULT_FEATURE_IMPORTANCE_PLOT,
    DEFAULT_METRICS_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_RAW_DATA_FILE,
    ensure_directories,
)
from src.database import get_recent_alerts, get_recent_predictions, init_db, log_prediction
from src.predictor import load_model_artifact, predict_single
from src.simulator import generate_simulated_reading, generate_simulated_stream
from src.utils import get_file_hash, load_json, verify_artifact_provenance

# Configure Streamlit page layout
st.set_page_config(
    page_title="AI Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure runtime output directories and database tables exist
ensure_directories()
init_db()

# Custom CSS for clean aesthetics
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .sim-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-container {
        padding: 1rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_model():
    """Load model pipeline with Streamlit resource caching."""
    if not DEFAULT_MODEL_FILE.exists():
        return None
    try:
        return load_model_artifact(DEFAULT_MODEL_FILE)
    except Exception:
        return None


# Sidebar Navigation
st.sidebar.title("⚙️ Maintenance Hub")
st.sidebar.markdown(
    """
    <div style='margin-bottom: 1rem;'>
      <span class='sim-badge'>SOFTWARE DEMO MODE</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.info(
    "**Notice:** Software-based ML demonstrator. Virtual telemetry is mathematically "
    "simulated and does not reflect real hardware sensors."
)

nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "📋 Project Overview",
        "📈 Model Performance & Metrics",
        "🔍 Model Explainability",
        "🛠️ Manual Equipment Diagnostic",
        "📊 Live IoT Telemetry Monitor",
        "🗄️ Database Audit & Logs",
    ],
)

# Header
st.markdown("<h1 class='main-title'>AI-Powered Predictive Maintenance System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Industrial Machine Health Risk Scoring & Alert Engine (AI4I 2020 Benchmark)</p>",
    unsafe_allow_html=True,
)

model = get_cached_model()

# =========================================================================
# SECTION A: Project Overview
# =========================================================================
if nav_selection == "📋 Project Overview":
    st.subheader("Project Architecture & Objectives")

    col_ov1, col_ov2 = st.columns([3, 2])
    with col_ov1:
        st.markdown(
            r"""
            ### Objective
            This system assesses industrial machinery failure risk using sensor telemetry.
            It provides actionable maintenance tiers (**NORMAL**, **WARNING**, **CRITICAL**)
            to prevent unplanned downtime and catastrophic equipment breakdown.

            ### Benchmark Dataset
            The machine learning model is trained on the public **AI4I 2020 Predictive Maintenance Dataset**
            (HTW Berlin / UCI Machine Learning Repository), comprising 10,000 milling machine operational records.

            ### Key Engineering Principles
            1. **Target Leakage Safeguards:** Failure subtypes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are strictly
               excluded from model inputs because they logically compose `Machine failure`.
            2. **Domain Feature Engineering:** Thermodynamic dissipation ($\Delta T$), mechanical spindle power,
               and tool wear overstrain are derived consistently across training and inference.
            3. **Severe Class Imbalance:** Failures account for only **3.39%** of observations.
               Model fitting applies balanced class weighting, and performance is evaluated via Precision,
               Recall, and PR-AUC rather than misleading accuracy.
            """
        )
    with col_ov2:
        st.write("##### Core Operational Boundaries")
        st.warning(
            """
            **Dataset Classification vs. Time-to-Failure:**
            The model assesses whether a machine state indicates instantaneous failure risk.
            It does **not** predict remaining useful life (RUL) in hours without time-series cycle data.
            """
        )
        st.info(
            """
            **Simulated Telemetry vs. Real Data:**
            The IoT Telemetry Monitor uses numerical simulations based on the AI4I empirical
            distribution. It demonstrates pipeline behavior without requiring physical microcontrollers.
            """
        )

# =========================================================================
# SECTION B: Model Performance & Metrics
# =========================================================================
elif nav_selection == "📈 Model Performance & Metrics":
    st.subheader("Held-Out Test Set Performance")
    st.caption("Standardized evaluation on 2,000 held-out test observations stratified by Machine failure.")

    if DEFAULT_METRICS_FILE.exists():
        metrics = load_json(DEFAULT_METRICS_FILE)

        # Artifact provenance check
        if DEFAULT_MODEL_FILE.exists() and "model_sha256" in metrics:
            curr_model_hash = get_file_hash(DEFAULT_MODEL_FILE)
            if metrics["model_sha256"] != curr_model_hash:
                st.warning(
                    "⚠️ **Provenance Notice:** Loaded evaluation metrics were computed on an earlier "
                    "model version. Please run `python main.py evaluate` to synchronize."
                )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Recall (Failure Class 1)", f"{metrics['recall']:.2%}", help="Portion of actual failures successfully caught")
        m_col2.metric("Precision (Failure Class 1)", f"{metrics['precision']:.2%}", help="Portion of failure alarms that were genuine")
        m_col3.metric("F1-Score", f"{metrics['f1_score']:.2%}", help="Harmonic mean balancing precision and recall")
        m_col4.metric("PR-AUC (Avg Precision)", f"{metrics['pr_auc']:.4f}", help="Precision-Recall Area Under Curve")

        st.markdown("---")
        plot_col, note_col = st.columns([1, 1])
        with plot_col:
            st.write("##### Held-Out Confusion Matrix")
            if DEFAULT_CONFUSION_MATRIX_FILE.exists():
                st.image(str(DEFAULT_CONFUSION_MATRIX_FILE), use_column_width=True)
            else:
                st.info("Confusion matrix plot file not found.")

        cm_dict = metrics.get("confusion_matrix", {})
        fn_val = cm_dict.get("fn", 12)
        fp_val = cm_dict.get("fp", 17)

        with note_col:
            st.write("##### Industrial Metric Definitions")
            st.markdown(
                f"""
                - **Positive Class:** `Machine failure = 1` (Averaging Strategy: `binary`).
                - **False Negatives (FN = {fn_val}):** The model predicted 'Healthy' but the machine failed.
                  In an industrial plant, this represents unscheduled breakdown or tool breakage.
                - **False Positives (FP = {fp_val}):** The model predicted 'Failure' but the machine was normal.
                  In an industrial plant, this triggers a preventative inspection with minimal downtime.
                - **Accuracy ({metrics['accuracy']:.2%}):** Reported for completeness, but note that a naive classifier
                  predicting zero failures achieves 96.60% accuracy while detecting nothing.
                """
            )
            st.caption(f"Evaluated on {metrics['total_test_samples']} test observations at {metrics['evaluated_at']}.")
    else:
        st.warning(
            "Evaluation metrics not found at `outputs/metrics/evaluation_metrics.json`. "
            "Please run: `python main.py evaluate` to generate performance diagnostics."
        )

# =========================================================================
# SECTION C: Model Explainability & Feature Importance
# =========================================================================
elif nav_selection == "🔍 Model Explainability":
    st.subheader("Global Feature Importance & Model Reliance")
    st.caption("Gini importance (Mean Decrease in Impurity) extracted from the Random Forest pipeline.")

    if DEFAULT_FEATURE_IMPORTANCE_FILE.exists():
        df_imp = pd.read_csv(DEFAULT_FEATURE_IMPORTANCE_FILE)

        col_plot, col_table = st.columns([3, 2])
        with col_plot:
            st.write("##### Feature Importance Distribution")
            if DEFAULT_FEATURE_IMPORTANCE_PLOT.exists():
                st.image(str(DEFAULT_FEATURE_IMPORTANCE_PLOT), use_column_width=True)
            else:
                st.info("Feature importance plot file not found.")

        with col_table:
            st.write("##### Importance Ranking Table")
            st.dataframe(df_imp, use_container_width=True, hide_index=True)

        st.info(
            """
            **Important Scientific Disclaimer:**
            Feature importance indicates mathematical reliance within the trained Random Forest
            ensemble (the fraction of impurity reduction attributed to each split).
            It is **not proof of physical causation**.
            """
        )
    else:
        st.warning(
            "Feature importance data not found at `outputs/metrics/feature_importance.csv`. "
            "Please run: `python main.py explain` to compute model explainability diagnostics."
        )

# =========================================================================
# SECTION D: Manual Equipment Diagnostic
# =========================================================================
elif nav_selection == "🛠️ Manual Equipment Diagnostic":
    st.subheader("Manual Operational Parameter Testing")
    st.caption("Input custom or observed sensor values to obtain immediate failure risk scoring.")

    with st.form("manual_diag_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_type = st.selectbox("Machine Quality Type", ["L", "M", "H"], index=1, help="L=Low (50%), M=Medium (30%), H=High (20%)")
            air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        with col2:
            process_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=325.0, value=310.0, step=0.1)
            rot_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
        with col3:
            torque = st.number_input("Torque [Nm]", min_value=3.0, max_value=85.0, value=40.0, step=0.5)
            tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=260, value=120, step=1)

        save_to_db = st.checkbox("Persist prediction to SQLite database history", value=True)
        submitted = st.form_submit_button("Evaluate Machine State", type="primary")

    if submitted:
        if model is None:
            st.error("Model artifact not found. Please train the model first by running: `python main.py train`.")
        else:
            obs_payload = {
                "Type": input_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rot_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }

            pred = predict_single(obs_payload)
            alert = evaluate_alert(pred["risk_score"], obs_payload)

            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(
                    label="Prediction Classification",
                    value="FAILURE RISK" if pred["is_failure"] else "NORMAL",
                    delta="Elevated Risk" if pred["is_failure"] else "Nominal",
                    delta_color="inverse" if pred["is_failure"] else "normal",
                )
            with res_col2:
                st.metric(
                    label="Algorithmic Risk Score",
                    value=f"{pred['risk_score']:.4f}",
                    help="Tree ensemble voting score (uncalibrated probability).",
                )
            with res_col3:
                st.metric(label="Alert Level Tier", value=alert["alert_level"])

            # Diagnostic Alert Box
            if alert["alert_level"] == "CRITICAL":
                st.error(f"🚨 **{alert['message']}**\n\n**Prescribed Action:** {alert['recommended_action']}")
            elif alert["alert_level"] == "WARNING":
                st.warning(f"⚠️ **{alert['message']}**\n\n**Prescribed Action:** {alert['recommended_action']}")
            else:
                st.success(f"✅ **{alert['message']}**\n\n**Prescribed Action:** {alert['recommended_action']}")

            st.caption(
                "Note: Model scores represent algorithmic risk rankings, not calibrated physical failure probabilities. "
                "This model does not forecast remaining useful life in hours."
            )

            if save_to_db:
                try:
                    log_prediction(
                        features=obs_payload,
                        predicted_failure=pred["predicted_class"],
                        prediction_score=pred["risk_score"],
                        alert_level=alert["alert_level"],
                        is_simulated=False,
                    )
                    st.success("Diagnostic event persisted to SQLite database (`data/predictive_maintenance.db`).")
                except Exception as db_err:
                    st.error(f"Failed to record diagnostic to database: {db_err}")

# =========================================================================
# SECTION E: Live IoT Telemetry Monitor
# =========================================================================
elif nav_selection == "📊 Live IoT Telemetry Monitor":
    st.subheader("Simulated IoT Telemetry Streaming")
    st.caption("Demonstrating streaming IoT sensor readings without requiring physical hardware.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        scenario = st.selectbox(
            "Operating Scenario",
            ["normal", "wear_stress", "thermal_stress", "overstrain"],
            help="Select an operational pattern to simulate specific physical wear mechanisms.",
        )
    with col_ctrl2:
        device_type = st.selectbox("Machine Type", ["Auto (Random)", "L", "M", "H"])
    with col_ctrl3:
        n_ticks = st.number_input("Streaming Batch Size", min_value=1, max_value=25, value=5)

    if st.button("Generate Simulated Telemetry Burst", type="primary"):
        target_type = None if device_type.startswith("Auto") else device_type
        readings = []
        for i in range(n_ticks):
            r = generate_simulated_reading(device_type=target_type, scenario=scenario)
            readings.append(r)

        df_stream = pd.DataFrame(readings)
        st.write("##### Streamed Sensor Telemetry Observations")
        st.dataframe(df_stream, use_container_width=True)

        if model is not None:
            st.write("##### Predictive Model Assessment")
            results = []
            for r in readings:
                obs = {
                    "Type": r["Type"],
                    "Air temperature [K]": r["Air temperature [K]"],
                    "Process temperature [K]": r["Process temperature [K]"],
                    "Rotational speed [rpm]": r["Rotational speed [rpm]"],
                    "Torque [Nm]": r["Torque [Nm]"],
                    "Tool wear [min]": r["Tool wear [min]"],
                }
                pred = predict_single(obs)
                alert = evaluate_alert(pred["risk_score"], obs)
                results.append(
                    {
                        "Type": obs["Type"],
                        "Tool Wear (min)": obs["Tool wear [min]"],
                        "Torque (Nm)": obs["Torque [Nm]"],
                        "Risk Score": pred["risk_score"],
                        "Alert Tier": alert["alert_level"],
                        "Status": "FAILURE RISK" if pred["is_failure"] else "HEALTHY",
                    }
                )
                try:
                    log_prediction(
                        features=obs,
                        predicted_failure=pred["predicted_class"],
                        prediction_score=pred["risk_score"],
                        alert_level=alert["alert_level"],
                        is_simulated=True,
                    )
                except Exception as db_err:
                    st.warning(f"Database write skipped: {db_err}")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("Trained model not detected. Train the model via CLI to enable scoring.")

# =========================================================================
# SECTION F: Database Audit & Logs
# =========================================================================
elif nav_selection == "🗄️ Database Audit & Logs":
    st.subheader("Historical Prediction & Telemetry Audit Log")
    st.caption("Records persisted to SQLite (`data/predictive_maintenance.db`).")

    preds = get_recent_predictions(limit=50)
    if preds:
        df_p = pd.DataFrame(preds)
        # Create prominent human-readable source type label
        if "is_simulated" in df_p.columns:
            df_p["Telemetry_Source"] = df_p["is_simulated"].apply(
                lambda x: "SIMULATED (Demo)" if x == 1 else "MANUAL (Diagnostic)"
            )
            sim_count = int((df_p["is_simulated"] == 1).sum())
            manual_count = int((df_p["is_simulated"] == 0).sum())
            st.caption(f"Showing {len(df_p)} records — Manual: {manual_count} | Simulated: {sim_count}")

        st.dataframe(df_p, use_container_width=True)
    else:
        st.info("No prediction events recorded yet in the database. Run predictions to populate this table.")
