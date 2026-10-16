# Technical Viva & Oral Examination Preparation Guide

**Project:** AI-Powered Predictive Maintenance System for IoT Devices  
**Target Audience:** Diploma Student preparing for project viva, technical review, and oral examination.

---

### Q1: What is Predictive Maintenance (PdM) and why is it preferred over reactive or preventive maintenance?
**Answer:**  
In industrial manufacturing, there are three primary maintenance paradigms:
1. **Reactive Maintenance (Run-to-Failure):** Equipment is repaired only after it breaks down. This causes catastrophic unscheduled downtime, damaged tooling, and ruined workpieces.
2. **Preventive Maintenance (Calendar/Time-Based):** Components are serviced or replaced at rigid, scheduled intervals (e.g., every 500 operating hours) regardless of actual condition. This often wastes remaining tool life and incurs unnecessary labor costs.
3. **Predictive Maintenance (Condition-Based):** Continuous sensor telemetry (temperatures, rotational speeds, torque, tool wear) is analyzed using machine learning algorithms to detect early warning patterns of degradation. Maintenance is scheduled only when warranted by actual equipment stress, minimizing unscheduled breakdowns while maximizing component lifespan.

---

### Q2: Why does this project use the AI4I 2020 Predictive Maintenance Dataset?
**Answer:**  
The AI4I 2020 dataset (hosted by the UCI Machine Learning Repository) is an established, peer-reviewed benchmark representing a Computer Numerical Control (CNC) milling machine. It models real-world physical phenomena, such as tool wear, thermal heat dissipation, and mechanical overstrain, across 10,000 realistic operating records. It provides an authentic industrial baseline with severe class imbalance (3.39% failure rate), allowing us to demonstrate realistic machine learning challenges without needing access to an active factory floor.

---

### Q3: Why is this project designed as a software-only implementation?
**Answer:**  
Industrial CNC milling machinery and calibrated dynamometer spindles cost tens of thousands of dollars and are not accessible for student academic coursework. The primary engineering objectives of this project are:
- Data quality auditing and target leakage prevention.
- Machine learning pipeline development with scikit-learn.
- Cryptographic artifact provenance tracking.
- Interactive dashboard engineering and SQLite logging.
- Automated testing (48 unit and integration tests).

To demonstrate operational streaming workflows without hardware dependencies, the system includes a dedicated IoT telemetry simulator that explicitly labels virtual readings as simulated.

---

### Q4: How does a Random Forest Classifier work, and why was it chosen for this task?
**Answer:**  
A Random Forest is an ensemble learning algorithm that constructs multiple decision trees (100 trees in our pipeline) during training:
- **Bootstrap Aggregating (Bagging):** Each tree is trained on a random sample of the training data drawn with replacement.
- **Random Feature Subsets:** At each split in a tree, only a random subset of available features is considered, ensuring individual trees are decorrelated.
- **Majority Voting:** For classification, the forest pools the independent predictions of all individual trees and outputs the majority class.

Random Forest was chosen because:
1. It handles non-linear relationships and complex interactions between continuous sensor measurements effectively.
2. It is resistant to overfitting compared to single decision trees.
3. It does not require feature scaling (normalization or standardization), preserving the physical interpretability of sensor units (Kelvin, RPM, Nm, minutes).
4. It natively provides Mean Decrease in Impurity (Gini importance) for explainability.

---

### Q5: Why is a stratified train/test split necessary instead of a simple random split?
**Answer:**  
The dataset has severe class imbalance: only 339 out of 10,000 observations (3.39%) represent machine failure. In an unstratified random split, the minority failure class could be distributed unevenly—for example, the test set could end up with disproportionately few failures, rendering performance metrics unreliable.  
A **stratified split** guarantees that the training set (8,000 rows) and the held-out test set (2,000 rows) both contain exactly the same 3.39% proportion of failure instances (271 failures in train, 68 failures in test).

---

### Q6: What is target leakage, and how is it prevented in this codebase?
**Answer:**  
**Target leakage** occurs when features that contain direct information about the target variable—information that would never be available at real-time inference before failure occurs—are included in the model training matrix.  
In the raw AI4I dataset, there are five specific failure-mode breakdown flags:
- `TWF` (Tool Wear Failure)
- `HDF` (Heat Dissipation Failure)
- `PWF` (Power Failure)
- `OSF` (Overstrain Failure)
- `RNF` (Random Failure)

These flags are direct components of the binary target `Machine failure` (`Machine failure = TWF | HDF | PWF | OSF | RNF`). If these columns were included, a decision tree would achieve artificial 100% test accuracy simply by checking if any flag is 1. In `src/preprocessing.py`, all five leakage columns (along with non-predictive identifiers `UDI` and `Product ID`) are strictly dropped prior to model fitting and inference.

---

### Q7: Why is classification accuracy an insufficient metric for this dataset?
**Answer:**  
Because 96.61% of the dataset is healthy (`Machine failure = 0`), a trivial "dummy" classifier that always outputs `0` regardless of sensor inputs would achieve **96.61% accuracy**. However, such a model would be completely useless in practice because it would fail to detect 100% of machine breakdowns.  
Therefore, evaluation must rely on minority-class metrics that explicitly measure performance on Class `1` (Failure): **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**.

---

### Q8: What do Precision, Recall, F1-Score, ROC-AUC, and PR-AUC mean in our system?
**Answer:**  
On our held-out test set of 2,000 observations (1,932 Normal, 68 Failures):
- **Precision (76.71%):** Out of all 73 machine states flagged by our model as failure, 56 were genuine failures ($TP / (TP + FP) = 56 / 73$).
- **Recall (82.35%):** Out of the 68 actual failures that occurred in the test set, our model caught 56 ($TP / (TP + FN) = 56 / 68$).
- **F1-Score (79.43%):** The harmonic mean of precision and recall ($2 \cdot \frac{P \cdot R}{P + R}$), balancing the penalty between false alarms and missed breakdowns.
- **ROC-AUC (0.9845):** The Area Under the Receiver Operating Characteristic curve, measuring the classifier's ability to rank failure instances above normal instances across all possible classification thresholds.
- **PR-AUC / Average Precision (0.8672):** The Area Under the Precision-Recall curve. Unlike ROC-AUC, PR-AUC focuses directly on the minority positive class and is the most honest metric for heavily imbalanced anomaly detection.

---

### Q9: In an industrial setting, what is the trade-off between False Positives (FP) and False Negatives (FN)?
**Answer:**  
- **False Negative (FN = 12):** The model predicts "Normal" but the machine actually fails. This is the worst outcome: it leads to unscheduled stoppage during active production, tool breakage, workpiece destruction, and potentially hazardous spindle seizure.
- **False Positive (FP = 17):** The model predicts "Failure" but the machine was healthy. An operator pauses the machine for a brief visual or torque check. This causes a minor 5-minute scheduled inspection delay.  
In industrial manufacturing, a False Positive is vastly preferred over a catastrophic False Negative. Our recall of 82.35% ensures over 8 out of 10 breakdowns are intercepted.

---

### Q10: What is the difference between Predicted Class, Model Score, and Operational Alert Tier?
**Answer:**  
These three concepts are intentionally kept distinct:
1. **Predicted Class (`0` or `1`):** The binary decision output of the classifier based on the default decision threshold ($\ge 0.50$).
2. **Model Risk Score ($0.0000$ to $1.0000$):** The raw ensemble voting fraction (`predict_proba()[:, 1]`) indicating the proportion of decision trees that voted for failure.
3. **Operational Alert Tier (`NORMAL`, `WARNING`, `CRITICAL`):** An operational policy defined in `src/alert_engine.py`:
   - `Score < 0.35` $\rightarrow$ **NORMAL** (Green): Continue standard production.
   - `0.35 <= Score < 0.65` $\rightarrow$ **WARNING** (Yellow): Early degradation signs; schedule end-of-shift inspection.
   - `Score >= 0.65` $\rightarrow$ **CRITICAL** (Red): High risk; halt spindle immediately for maintenance.  
This separation allows operators to take precautionary action before an observation crosses the strict classification boundary.

---

### Q11: Why is the Random Forest score not a calibrated real-world failure probability?
**Answer:**  
The score produced by `predict_proba()` in a Random Forest is simply the empirical fraction of trees in the ensemble that voted for Class 1 (e.g., 81 trees out of 100 voting failure gives a score of 0.81).  
This vote fraction is a monotonic ranking index, not a true Bayesian posterior probability. Without calibrating the model against long-term fleet operational failure frequencies (using techniques such as Platt scaling or Isotonic regression), a score of 0.81 does not mean the machine has an 81% mathematical chance of breaking down in the next hour. The system clearly documents this distinction in both the CLI and the dashboard.

---

### Q12: How are the three engineered features calculated, and what physical phenomena do they capture?
**Answer:**  
In `src/feature_engineering.py`:
1. `Temp_Difference_K` $= \text{Process temperature [K]} - \text{Air temperature [K]}$  
   *Physical Meaning:* Measures the heat dissipation gradient. When a machine operates under high load with inadequate ventilation or clogged coolant lines, process heat accumulates and this temperature difference drops.
2. `Power_Proxy_W` $= \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$  
   *Physical Meaning:* Calculates mechanical cutting power ($P = \tau \cdot \omega$) dissipated at the tool-workpiece interface. Excessive power indicates high friction or jamming.
3. `Overstrain_Proxy` $= \text{Tool wear [min]} \times \text{Torque [Nm]}$  
   *Physical Meaning:* The product of cutting insert contact duration and applied torque. High torque on a severely worn cutting insert significantly elevates the risk of sudden mechanical fracture (Overstrain Failure).

---

### Q13: What is the difference between Gini Importance and Permutation Feature Importance?
**Answer:**  
- **Gini Importance (Mean Decrease in Impurity):** Computes the total reduction in split Gini impurity brought about by a feature across all trees in the forest. It is computationally fast because it is calculated during training, but it can be biased toward high-cardinality continuous features.
- **Permutation Importance:** Evaluated on the held-out test set. A single feature column is randomly shuffled, breaking its relationship with the target, and the drop in the model's F1/accuracy score is measured. If shuffling a feature severely degrades performance, the model genuinely relies on it.

Both methods confirmed that `Power_Proxy_W`, `Torque [Nm]`, `Rotational speed [rpm]`, `Tool wear [min]`, and `Overstrain_Proxy` are the most influential features.

---

### Q14: Does high feature importance prove that a feature physically causes failure?
**Answer:**  
**No.** Feature importance measures mathematical reliance in the trained machine learning model, not physical causation. For example, high tool wear strongly correlates with failure, but a dull tool does not necessarily cause electrical power failure (`PWF`). Correlation is not causation; domain engineering and physical root-cause analyses are required to confirm physical failure mechanisms.

---

### Q15: How is SQLite used in this project, and how are SQL injection vulnerabilities prevented?
**Answer:**  
In `src/database.py`, SQLite stores operational prediction logs and telemetry records at `data/predictive_maintenance.db`.  
To prevent SQL injection vulnerabilities:
- No dynamic string concatenation or f-strings are used to construct SQL statements.
- All database queries strictly utilize parameterized SQL statements with `?` placeholders (e.g., `cursor.execute("INSERT INTO predictions VALUES (?, ?, ...)", tuple_values)`).
- This ensures user input is always treated as literal data, never executable SQL commands.

---

### Q16: How does the Streamlit dashboard prevent duplicate database writes on page refresh?
**Answer:**  
In Streamlit's execution model, any widget interaction normally causes the entire script to rerun from top to bottom. If diagnostic inputs were placed directly on the page, modifying any slider could trigger unintended database insertions.  
In `app.py`, manual diagnostic inputs are wrapped inside an explicit Streamlit form container:
```python
with st.form("manual_diag_form"):
    # sensor input sliders & dropdowns...
    submitted = st.form_submit_button("Run Diagnostics & Log")
    if submitted:
        # database logging occurs ONLY on explicit click
```
This guarantees that inference logging executes strictly when the user clicks the submission button, preventing rerun duplicates.

---

### Q17: How does the IoT telemetry simulator work, and why must it be labeled as synthetic?
**Answer:**  
In `src/simulator.py`, the simulator generates synthetic bursts of sensor readings by sampling from realistic Gaussian distributions corresponding to specific operating scenarios (`normal`, `wear_stress`, `thermal_stress`, `overstrain`).  
Every generated observation is explicitly marked with `is_simulated = True` (or `1` in SQLite). This is essential for transparency and data integrity: synthetic demonstration data must never be confused with physical IoT sensor readings, and it is strictly excluded from model evaluation.

---

### Q18: What do the SHA-256 cryptographic provenance checks protect against?
**Answer:**  
In modern ML engineering, files can easily become desynchronized (e.g., someone retrains the model on a different dataset or updates evaluation metrics without updating the model).  
In `src/utils.py`, `verify_artifact_provenance()` computes the SHA-256 hash of:
1. The raw dataset (`ai4i2020.csv`)
2. The active serialized model (`predictive_maintenance_rf.joblib`)
3. The persisted train/test split (`train_test_split.json`)
4. The evaluation metrics (`evaluation_metrics.json`)
5. The model metadata (`model_metadata.json`)

If the model is retrained, its hash changes. If `evaluation_metrics.json` still records the old hash, `main.py check` and the Streamlit dashboard immediately display a provenance mismatch alert, preventing stale results from being presented.

---

### Q19: Why does this system NOT estimate Remaining Useful Life (RUL)?
**Answer:**  
**Remaining Useful Life (RUL)** estimation requires run-to-failure continuous time-series data—historical logs tracking a specific machine unit continuously from a brand-new state until its eventual breakdown.  
The AI4I 2020 dataset consists of independent, point-in-time snapshot records of machine operating conditions. It does not provide continuous time timestamps or sequential run-to-failure trajectories for individual spindles. Therefore, our model performs **instantaneous state classification** (determining whether the current operating state indicates imminent failure risk), not RUL forecasting.

---

### Q20: What are the fundamental limitations of this project that you should acknowledge?
**Answer:**  
An honest engineering project must acknowledge its boundaries:
1. **Software-Only Prototype:** No physical machines or IoT hardware were tested.
2. **Instantaneous Classification Only:** It does not forecast time-to-failure or estimate RUL.
3. **Uncalibrated Model Scores:** Random Forest voting fractions are ranking indices, not calibrated Bayesian probabilities.
4. **Stratified Random Split Limitation:** The model was tested on randomly sampled records from the same dataset; it has not been validated across different factories or machine brands.
5. **Not a Certified Industrial Controller:** This is an educational demonstration prototype, not a safety-certified industrial SIL (Safety Integrity Level) system.

---

### Q21: What happens if an operator enters physically impossible sensor values (e.g., negative temperature)?
**Answer:**  
In `src/predictor.py`, defensive bounds checking is implemented in `extract_model_features()`:
- Air and Process temperatures must be strictly greater than $0\text{ K}$ (absolute zero boundary check).
- Rotational speed, Torque, and Tool wear cannot be negative ($min \ge 0$).
- Upper physical ceiling limits are enforced (e.g., tool wear $> 500\text{ min}$ or temperature $> 400\text{ K}$ are flagged as physically implausible).
If an invalid value or `NaN`/`Inf` is passed, the system rejects the input with a clear `ValueError` before inference is attempted.

---

### Q22: Why was `class_weight='balanced_subsample'` chosen over standard `balanced`?
**Answer:**  
`class_weight='balanced'` calculates class weights once across the entire training dataset. In contrast, `class_weight='balanced_subsample'` recalculates the class weights dynamically for each individual bootstrap sample drawn to train each decision tree. Because the failure class is rare (3.39%), some bootstrap samples may have slightly fewer or more failure samples; `balanced_subsample` ensures each individual tree is properly balanced regardless of bootstrap variation.

---

### Q23: How would you extend this project into a production-grade industrial IoT deployment?
**Answer:**  
To transition this prototype into a full industrial deployment:
1. **IoT Connectivity:** Implement real industrial protocols like MQTT or OPC-UA to ingest live telemetry from machine PLC (Programmable Logic Controller) controllers.
2. **Sequential Time-Series Modeling:** Collect run-to-failure continuous telemetry and implement LSTM (Long Short-Term Memory) or Temporal Convolutional Networks to predict exact RUL (hours remaining).
3. **Probability Calibration:** Apply Platt scaling (logistic regression on ensemble margins) or Isotonic regression on a validation set to calibrate risk scores into empirical failure probabilities.
4. **Edge Computing:** Containerize the inference pipeline with Docker and deploy it to an edge gateway (e.g., Siemens IOT2050 or NVIDIA Jetson) on the factory floor for sub-millisecond local inference.
5. **Automated Alert Dispatch:** Connect `src/alert_engine.py` to factory enterprise systems (e.g., SAP PM or webhooks to maintenance technician mobile apps).
