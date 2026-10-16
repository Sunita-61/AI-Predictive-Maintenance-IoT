# Project Implementation Plan & Progress Tracking

## Project Phases & Status Matrix

| Phase | Description | Status | Deliverables & Verified Outcomes |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Workspace & Architecture Setup | **COMPLETED & VERIFIED** | Folder structure, documentation, config paths, environment config, requirements, and gitignore. |
| **Phase 2** | Data Ingestion & Preprocessing | **COMPLETED & VERIFIED** | AI4I 2020 dataset validated (10,000 rows, 0 nulls, 0 duplicates, 3.39% failure rate). Leakage guards drop TWF, HDF, PWF, OSF, RNF, UDI, Product ID. |
| **Phase 3** | Feature Engineering Pipeline | **COMPLETED & VERIFIED** | Validated domain features: `Temp_Difference_K`, `Power_Proxy_W`, `Overstrain_Proxy`. Consistent across train & inference. |
| **Phase 4** | Model Training & Provenance | **COMPLETED & VERIFIED** | Stratified split (8,000 train / 2,000 test). Dataset SHA-256 hash verified. Exact split indices saved to `models/train_test_split.json`. Balanced Random Forest saved to `models/predictive_maintenance_rf.joblib`. |
| **Phase 5** | Model Evaluation & Metrics Export | **COMPLETED & VERIFIED** | Evaluated on held-out 2,000 observations using persisted split indices: Accuracy 98.55%, Precision 76.71%, Recall 82.35%, F1 79.43%, ROC-AUC 0.9845, PR-AUC 0.8672. Saved to `outputs/`. |
| **Phase 6** | Inference Service & Predictor | **COMPLETED & VERIFIED** | Model loads from cache. Schema feature extraction prevents extra metadata contamination. Smoke tests on real dataset rows verified. |
| **Phase 7** | Model Explainability Analysis | **COMPLETED & VERIFIED** | Global Gini & Permutation feature importance. Mapped to readable names. Saved to `outputs/metrics/feature_importance.csv` and `outputs/plots/feature_importance.png`. |
| **Phase 8** | Alert Engine & IoT Simulator | **COMPLETED & VERIFIED** | 3-tier configurable alert threshold engine (Normal, Warning, Critical) with uncalibrated score notices. Simulated telemetry marked with `is_simulated = True`. |
| **Phase 9** | SQLite Database Persistence | **COMPLETED & VERIFIED** | Parameterized schema initialized at `data/predictive_maintenance.db`. |
| **Phase 10** | Streamlit Interactive Dashboard | **COMPLETED & VERIFIED** | Multi-tab UI covering Project Overview, Performance, Explainability, Diagnostics, Simulation, and Database Audit. |
| **Phase 11** | Automated Testing & Regression | **COMPLETED & VERIFIED** | 48 passing unit & regression tests (`pytest -v`). Syntax compilation clean across all modules. |
| **Phase 12** | End-to-End Integration & Split Hardening | **COMPLETED & VERIFIED** | Strict split validation (`test_split_validation.py`), dataset SHA-256 integrity, domain physical bounds checking, and full provenance tracking. |

---

## Detailed Model Evaluation Metrics (Held-Out Test Set: 2,000 observations)

- **Target Label:** `Machine failure` (Positive Class = 1, Negative Class = 0)
- **Metric Averaging Strategy:** `binary` (strict focus on detecting the minority failure class)
- **Dataset SHA-256 Provenance:** `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`
- **Class Distribution in Test Set:** 1,932 Normal (96.6%) vs 68 Failures (3.4%)
- **Confusion Matrix:**
  - True Negatives (TN): **1,915**
  - False Positives (FP): **17**
  - False Negatives (FN): **12**
  - True Positives (TP): **56**
- **Performance Metrics:**
  - **Accuracy:** `98.55%` (Baseline non-predictive majority accuracy: `96.60%`)
  - **Precision (Failure Class 1):** `76.71%` ($\frac{56}{56 + 17}$)
  - **Recall (Failure Class 1):** `82.35%` ($\frac{56}{56 + 12}$)
  - **F1-Score (Failure Class 1):** `79.43%`
  - **ROC-AUC Score:** `0.9845`
  - **PR-AUC (Average Precision):** `0.8672`

---

## Global Feature Importance Summary (Random Forest Gini MDI)

| Rank | Feature Name | Description | Gini Importance | Relative Weight |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `Power_Proxy_W` | Spindle mechanical power ($Torque \times Speed$) | 0.1901 | 19.01% |
| 2 | `Torque [Nm]` | Applied spindle torque | 0.1845 | 18.45% |
| 3 | `Rotational speed [rpm]` | Spindle rotational speed | 0.1820 | 18.20% |
| 4 | `Tool wear [min]` | Cumulative tool contact duration | 0.1415 | 14.15% |
| 5 | `Overstrain_Proxy` | Load-wear interaction ($Torque \times Tool Wear$) | 0.1229 | 12.29% |
| 6 | `Temp_Difference_K` | Thermal dissipation delta ($\Delta T$) | 0.0997 | 9.97% |
| 7 | `Air temperature [K]` | Ambient operational temperature | 0.0465 | 4.65% |
| 8 | `Process temperature [K]` | Machine internal process temperature | 0.0236 | 2.36% |
| 9 | `Type: M` | Medium-quality product variant | 0.0039 | 0.39% |
| 10 | `Type: L` | Low-quality product variant | 0.0036 | 0.36% |
| 11 | `Type: H` | High-quality product variant | 0.0018 | 0.18% |

*Note: Importance measures mathematical reliance in the trained tree ensemble. It is not proof of physical causation.*

---

## Operational Artifact Locations

- **Trained Pipeline Artifact:** `models/predictive_maintenance_rf.joblib`
- **Model Metadata File:** `models/model_metadata.json`
- **Reproducible Split Indices:** `models/train_test_split.json`
- **Evaluation Metrics JSON:** `outputs/metrics/evaluation_metrics.json`
- **Feature Importance CSV:** `outputs/metrics/feature_importance.csv`
- **Confusion Matrix Visualization:** `outputs/plots/confusion_matrix.png`
- **Feature Importance Visualization:** `outputs/plots/feature_importance.png`
- **SQLite Database:** `data/predictive_maintenance.db`
- **Raw Benchmark Dataset:** `data/raw/ai4i2020.csv`

---

## Commands to Reproduce Pipeline

```bash
# 1. Verify environment and dataset schema
python main.py check

# 2. Train baseline Random Forest model and persist split indices
python main.py train

# 3. Evaluate model on held-out test data and generate confusion matrix plot
python main.py evaluate

# 4. Generate global feature importance metrics and visualization
python main.py explain

# 5. Run automated test suite (48 passing tests)
python -m pytest -v

# 6. Launch the Streamlit interactive dashboard
streamlit run app.py
```

---

## Known Limitations & Boundaries

1. **State-Based Classification vs. Remaining Useful Life (RUL):**  
   The AI4I dataset contains point-in-time machine sensor records, not run-to-failure continuous degradation histories. The model assesses whether a given instantaneous machine state indicates failure risk; it does not predict an exact future horizon (e.g., "will fail in exactly 48 hours").
2. **Algorithmic Risk Scores:**  
   The probability scores produced by the Random Forest ensemble reflect tree vote proportions and are designated as uncalibrated algorithmic risk scores rather than true empirical failure probabilities.
3. **Simulated Telemetry:**  
   IoT telemetry generated by the simulator is software-generated for demonstration and is explicitly labeled as `is_simulated = True`.
