# Student Demonstration & Portfolio Presentation Checklist

This checklist provides a structured 13-step sequence to demonstrate the **AI-Powered Predictive Maintenance System for IoT Devices** during a technical portfolio presentation or viva examination.

---

## Step 1: Project Introduction & Problem Statement
Explain why predictive maintenance is critical in modern smart manufacturing and industrial IoT environments.

**Talking Points:**
- **Industrial Challenge:** Unscheduled equipment downtime in Computer Numerical Control (CNC) and milling machinery results in costly production halts, damaged tooling, and expensive scrap workpieces.
- **Project Objective:** Implement a machine-learning-driven condition monitoring system that analyzes multi-sensor operating telemetry to detect failure patterns before catastrophic breakdown occurs.
- **Educational Scope:** A production-grade software prototype demonstrating sound ML engineering, target leakage safeguards, cryptographic provenance, and automated testing without requiring physical machine hardware.

---

## Step 2: Software-Only System Architecture
Explain the modular, software-only architecture of the repository.

**Talking Points:**
- **Software-Only Prototype:** Employs empirical benchmark telemetry and simulated data; no physical microcontrollers, sensors, or machinery are required.
- **Modular Component Separation:**
  - `src/config.py`: Centralized paths, feature schemas, physical bounds, and operational alert thresholds.
  - `src/data_loader.py`: Raw CSV ingestion, schema validation, and data quality auditing.
  - `src/preprocessing.py`: Target leakage prevention and stratified split verification.
  - `src/feature_engineering.py`: Domain-specific thermodynamic and mechanical power calculations.
  - `src/train_model.py`: Balanced Random Forest pipeline training and metadata tracking.
  - `src/evaluate_model.py`: Held-out evaluation on persisted test indices with confusion matrix generation.
  - `src/explainability.py`: Mean Decrease in Impurity (Gini) and permutation feature importance.
  - `src/predictor.py`: Physical bounds validation and dynamic positive-class probability extraction.
  - `src/alert_engine.py`: Three-tier operational alert dispatching (Normal, Warning, Critical).
  - `src/database.py`: Parameterized SQLite logging for historical auditing.
  - `src/simulator.py`: Virtual telemetry generation explicitly tagged with `is_simulated = True`.
  - `app.py`: Streamlit multi-tab interactive operations and diagnostic dashboard.

---

## Step 3: Dataset Source & Quality Audit
Verify the workspace, dependencies, raw benchmark dataset, and serialized artifacts.

**Command:**
```bash
python main.py check
```
**Talking Points:**
- **Source:** The benchmark dataset is the open-access UCI AI4I 2020 Predictive Maintenance Dataset representing a synthetic milling machine with realistic physical operating conditions.
- **Dataset Properties:** Exactly 10,000 observations with 14 columns, 0 missing values, and 0 duplicate rows.
- **Imbalance Ratio:** Severe class imbalance with only 3.39% failure records (9,661 normal operating records vs 339 failure records; approximately 28.5:1 ratio).
- **Cryptographic Provenance:** Raw dataset integrity is locked via SHA-256 (`dc6630cd...8a8e`), verified against metadata and split files.

---

## Step 4: Data Preprocessing & Target Leakage Safeguards
Explain how target leakage and identifier artifacts are strictly eliminated before machine learning.

**Talking Points:**
- **Non-Predictive Identifier Removal:** `UDI` (arbitrary row index 1–10000) and `Product ID` (variant code) are strictly excluded from the feature matrix to prevent memorization.
- **Target Leakage Safeguard:** The raw dataset includes five specific failure-mode breakdown flags (`TWF` - Tool Wear Failure, `HDF` - Heat Dissipation Failure, `PWF` - Power Failure, `OSF` - Overstrain Failure, `RNF` - Random Failure).
- **Why Dropping Them is Critical:** These flags are downstream components of the binary target `Machine failure`. In a real-world setting, an operator does not know which failure mechanism caused a breakdown before the breakdown occurs. Retaining them would result in artificial 100% test accuracy that fails completely in real inference.
- **Pre-split Dropping:** In `src/preprocessing.py`, all five leakage columns are dropped prior to model fitting and inference.

---

## Step 5: Feature Engineering Pipeline
Show how mechanical and thermal physical domain knowledge is embedded into the feature matrix.

**Talking Points:**
- Rather than relying solely on raw sensor readings, three physics-inspired interaction features are computed:
  1. `Temp_Difference_K` $= \text{Process temperature [K]} - \text{Air temperature [K]}$  
     *Physical intuition:* Captures the thermal dissipation gradient between the cutting zone and ambient surroundings. Low delta indicates heat accumulation.
  2. `Power_Proxy_W` $= \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$  
     *Physical intuition:* Quantifies active mechanical cutting power ($P = \tau \cdot \omega$) dissipated at the spindle.
  3. `Overstrain_Proxy` $= \text{Tool wear [min]} \times \text{Torque [Nm]}$  
     *Physical intuition:* Represents cumulative mechanical stress on a worn cutting insert, capturing the combined risk of tool breakage.
- **Exact Parity:** The identical transformation (`engineer_features()`) is executed during training, held-out evaluation, CLI prediction, and dashboard inference.

---

## Step 6: Random Forest Training & Stratified Evaluation Methodology
Explain the ensemble model architecture, class weighting, and train/test partitioning.

**Talking Points:**
- **Algorithm:** Random Forest Classifier (100 decision trees) wrapped in a scikit-learn Pipeline with `OneHotEncoder(handle_unknown='ignore')` for product `Type` variants (`L`, `M`, `H`).
- **Class Imbalance Strategy:** Configured with `class_weight='balanced_subsample'`, which dynamically weights minority failure samples inversely proportional to class frequencies in each bootstrap sample.
- **Reproducible Partitioning:** A stratified 80/20 train/test split is partitioned using a fixed random seed (`random_state=42`).
- **Indices Persistence:** All 8,000 training row indices and 2,000 test row indices are saved to `models/train_test_split.json`. The model is evaluated exclusively on the held-out test partition.

---

## Step 7: Confusion Matrix Interpretation & Imbalanced Metrics
Review the held-out test evaluation outputs and explain the trade-offs.

**Command:**
```bash
python main.py evaluate
```

**Confusion Matrix (Held-out Test Set: 2,000 observations):**
```
                    Predicted Normal (0)   Predicted Failure (1)
Actual Normal (0)          1,915 (TN)              17 (FP)
Actual Failure (1)            12 (FN)              56 (TP)
```

**Key Metrics:**
- **Accuracy:** `98.55%`
- **Recall (Failure Class 1):** `82.35%` (56 out of 68 genuine failures detected)
- **Precision (Failure Class 1):** `76.71%` (56 out of 73 predicted failures genuine)
- **F1-Score:** `79.43%`
- **ROC-AUC:** `0.9845`
- **PR-AUC (Average Precision):** `0.8672`

**Talking Points:**
- **Why Naive Accuracy Fails:** Because 96.6% of test observations are healthy, a non-learning dummy model predicting "Normal" for every machine achieves 96.60% accuracy while detecting 0 failures.
- **False Negatives (FN = 12):** Uncaught breakdowns resulting in unscheduled downtime, workpiece destruction, or tool spindle damage.
- **False Positives (FP = 17):** Precautionary inspections that cause brief scheduled maintenance downtime. In industrial settings, FP is preferable to catastrophic FN.
- **PR-AUC (0.8672):** Precision-Recall Area Under Curve provides the most rigorous assessment for heavily imbalanced failure detection.

---

## Step 8: Model Explainability (Gini & Permutation Feature Importance)
Demonstrate global feature importance and discuss physical non-causality.

**Command:**
```bash
python main.py explain
```

**Top 5 Features by Gini Importance:**
1. `Power_Proxy_W` (19.01%): Spindle mechanical power ($Torque \times Speed$)
2. `Torque [Nm]` (18.45%): Applied spindle torque load
3. `Rotational speed [rpm]` (18.20%): Spindle rotation rate
4. `Tool wear [min]` (14.15%): Cumulative insert wear duration
5. `Overstrain_Proxy` (12.29%): Combined load and wear ($Torque \times Tool Wear$)

**Cautionary Scientific Explanation:**
- Gini importance (Mean Decrease in Impurity) measures mathematical split variance reduction across decision trees.
- **Correlation is not Causation:** High feature importance indicates that the Random Forest mathematically relies on that feature; it does not prove physical causation.

---

## Step 9: Normal Operating Condition Prediction
Demonstrate real-time inference on a healthy machine state.

**Command:**
```bash
python main.py predict --type M --air-temp 298.5 --process-temp 308.2 --rpm 1500 --torque 40.0 --tool-wear 50
```

**Expected Result on Active Model Artifact:**
- **Prediction:** `NORMAL (0)`
- **Risk Score:** `0.0001` (Uncalibrated tree ensemble score)
- **Alert Level:** `NORMAL`
- **Diagnostic:** `NORMAL: Equipment operating within healthy parameters (Risk score: 0.00).`
- **Recommended Action:** `Continue regular production monitoring.`

---

## Step 10: High-Stress Condition Prediction & Diagnostic Rule
Demonstrate detection of high tool wear, elevated torque, and compromised heat dissipation.

**Command:**
```bash
python main.py predict --type L --air-temp 300.0 --process-temp 305.0 --rpm 1300 --torque 68.0 --tool-wear 240
```

**Expected Result on Active Model Artifact:**
- **Prediction:** `FAILURE (1)`
- **Risk Score:** `0.8131`
- **Alert Level:** `CRITICAL`
- **Diagnostic:** `CRITICAL: High algorithmic failure risk score (0.81). Elevated tool wear detected (240.0 min). High torque load detected (68.0 Nm). Suboptimal heat dissipation delta (5.0 K).`
- **Recommended Action:** `Halt machine cycle immediately for diagnostic inspection. Check cutting insert wear, spindle bearings, and thermal ventilation.`

---

## Step 11: Synthetic Telemetry Simulation & Alert Generation
Demonstrate streaming sensor simulation without physical hardware.

**Command:**
```bash
python main.py simulate --scenario overstrain --stream 3
```

**Talking Points:**
- **Demonstration Only:** Virtual sensor readings are generated from parameterized distributions matching operating scenarios (`normal`, `wear_stress`, `thermal_stress`, `overstrain`).
- **Transparency Safeguard:** Every simulated record is explicitly stamped with `is_simulated = True`.
- **Isolation:** Synthetic telemetry is never mixed into model training or the official 2,000-sample held-out evaluation dataset.

---

## Step 12: SQLite Persistence & Streamlit Interactive Dashboard Demo
Launch the interactive web dashboard and demonstrate the operational workflows.

**Command:**
```bash
python -m streamlit run app.py
```

**Workflows to Walk Through:**
1. **Project Overview Tab:** System boundaries, AI4I dataset provenance, and educational prototype disclaimers.
2. **Model Performance Tab:** Dynamic metrics cards (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) and interactive confusion matrix visualization.
3. **Model Explainability Tab:** Rendered feature importance bar charts with the scientific non-causality note.
4. **Manual Diagnostics Tab:** Form-protected input sliders and dropdowns; enter sensor readings and view the multi-tier alert badge. Point out that submitting a form logs exactly one record to SQLite.
5. **IoT Simulation Tab:** Trigger synthetic bursts under different stress scenarios.
6. **Database Audit Tab:** View recent prediction history. Point out the `Telemetry_Source` column clearly distinguishing `SIMULATED (Demo)` from `MANUAL (Diagnostic)` records.

---

## Step 13: System Limitations & Future Engineering Roadmap
Summarize honest scientific boundaries and practical extension pathways.

**Core Scientific Boundaries:**
1. **Instantaneous State Classification vs. Remaining Useful Life (RUL):** The model classifies instantaneous sensor state; it does not estimate the remaining operating hours before failure.
2. **Uncalibrated Model Scores:** The score represents the tree vote fraction ($\frac{\text{positive votes}}{100}$), serving as an algorithmic risk index rather than a true real-world failure probability.
3. **Random Partitioning vs. Temporal Forecasting:** The model is evaluated on randomly stratified held-out observations from the same machines; future work requires temporal splits across machines.
4. **Educational Prototype:** This software prototype demonstrates ML engineering and IoT analytics concepts; it is not certified for industrial safety-critical equipment control.

**Future Extension Roadmap:**
- Integration with real industrial IoT protocols (MQTT, OPC-UA, Kafka).
- Temporal sequence modeling (LSTM, Temporal Convolutional Networks) for run-to-failure RUL estimation.
- Probability calibration (Platt scaling or Isotonic Regression) against long-term maintenance ground truth.
- Edge deployment onto containerized embedded gateways (e.g., Raspberry Pi or NVIDIA Jetson).
