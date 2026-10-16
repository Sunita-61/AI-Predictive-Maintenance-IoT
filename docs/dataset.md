# Dataset Analysis & Specification: AI4I 2020

## 1. Dataset Origin & Metadata

- **Dataset Name:** AI4I 2020 Predictive Maintenance Dataset
- **Hosting Organization:** UCI Machine Learning Repository
- **URL:** [https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Creators:** Stephan Matzka, School of Engineering - Technology and Life, HTW Berlin, Germany.
- **Record Count:** 10,000 rows
- **Raw Feature Count:** 14 columns

---

## 2. Feature Dictionary

| Raw Column Header | Data Type | Physical Range / Units | Description & Purpose |
| :--- | :--- | :--- | :--- |
| `UDI` | Integer | [1, 10000] | Unique row index. Excluded from modeling. |
| `Product ID` | String | e.g., `M14860` | Letter denotes product quality variant (L/M/H), followed by variant-specific serial number. Excluded from modeling. |
| `Type` | Categorical | `L`, `M`, `H` | Quality variant: Low (50% of records), Medium (30%), High (20%). Included via one-hot encoding. |
| `Air temperature [K]` | Float | ~295.3 K to 304.5 K | Ambient air temperature generated via random walk normalized to 300 K. |
| `Process temperature [K]` | Float | ~305.7 K to 313.8 K | Machine process temperature, typically ~10 K higher than ambient air. |
| `Rotational speed [rpm]` | Integer | ~1168 rpm to 2886 rpm | Calculated from spindle electric power (2860 W) with overlaid normal noise. |
| `Torque [Nm]` | Float | ~3.8 Nm to 76.6 Nm | Torque values normally distributed around 40 Nm with no negative values. |
| `Tool wear [min]` | Integer | 0 min to 253 min | Cumulative tool contact time. Quality variants H/M/L add 5/3/2 minutes of wear per used tool. |
| `Machine failure` | Binary | 0 or 1 | **Target variable**: 1 indicates machine failure has occurred. |
| `TWF` | Binary | 0 or 1 | Tool Wear Failure mode: failure caused by wear exceeding threshold. |
| `HDF` | Binary | 0 or 1 | Heat Dissipation Failure mode: failure caused by insufficient $\Delta T$. |
| `PWF` | Binary | 0 or 1 | Power Failure mode: failure caused by electrical power anomalies. |
| `OSF` | Binary | 0 or 1 | Overstrain Failure mode: failure caused by excessive torque $\times$ wear. |
| `RNF` | Binary | 0 or 1 | Random Failure mode: minor unpredictable breakdown. |

---

## 3. Data Leakage & Safeguards

### Target Leakage Rule
In the raw AI4I dataset, `Machine failure` is defined as the logical union of the five specific failure modes:
$$\text{Machine failure} = \text{TWF} \lor \text{HDF} \lor \text{PWF} \lor \text{OSF} \lor \text{RNF}$$

If any of these five columns are included as predictors during training:
1. The machine learning model will learn a trivial mathematical shortcut rather than learning from sensor telemetry.
2. Training and test scores will display artificially inflated metrics (approaching 100% accuracy and F1).
3. The model will utterly fail in production where only raw sensor telemetry is available.

**Mandatory Action:** The columns `['TWF', 'HDF', 'PWF', 'OSF', 'RNF']` are strictly filtered out during feature preparation.

### Identifier Exclusion Rule
`UDI` and `Product ID` are arbitrary identifiers that provide no generalizable mechanical insight. `Product ID` contains the product variant letter (`L`, `M`, or `H`), which is already cleanly isolated in the `Type` feature. Both identifiers are discarded.

---

## 4. Class Imbalance Characteristics

- **Total Observations:** 10,000
- **Negative Class (`Machine failure = 0`):** 9,661 (96.61%)
- **Positive Class (`Machine failure = 1`):** 339 (3.39%)

### Imbalance Implications:
- A dummy classifier predicting `0` (Normal) on all inputs achieves **96.61% accuracy** while detecting zero failures.
- **Accuracy is an invalid primary evaluation metric** for this task.
- Model performance must be evaluated using **Precision**, **Recall**, **F1-Score**, **Area Under the Precision-Recall Curve (PR-AUC)**, and **Confusion Matrix inspection**.
- Stratified sampling (`StratifiedShuffleSplit` or `stratify=y` in `train_test_split`) must be used to ensure both training and test sets contain proportional representations of the 3.39% failure class.

---

## 5. Scope & Prognostic Horizon Limitations

1. **State Classification vs. RUL Forecasting:**
   - The AI4I dataset consists of snapshot observations rather than an explicitly indexed multi-sensor time series.
   - The model learns to classify whether a given sensor state indicates anomalous/failure operation.
   - It is scientifically improper to claim this model predicts failure "24 hours in advance" or calculates exact "Remaining Useful Life (RUL)" without continuous run-to-failure cycle data.

2. **Probability Calibration:**
   - Default tree ensemble scores (`predict_proba`) indicate tree voting fractions, not statistically calibrated empirical probabilities of hardware failure.
   - Scores are referred to as **algorithmic risk scores** unless formal probability calibration (e.g., CalibratedClassifierCV) is executed.
