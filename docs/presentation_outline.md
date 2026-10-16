# Student Presentation Outline & Slide Deck Guide

**Project:** AI-Powered Predictive Maintenance System for IoT Devices  
**Target Duration:** 5 to 7 Minutes  
**Target Audience:** Project Examiners, Technical Faculty, and Peer Evaluators  

---

## Presentation Structure Overview

| Slide # | Slide Title | Allotted Time | Key Focus |
| :---: | :--- | :---: | :--- |
| **1** | Title & Project Overview | 0.5 min | Problem statement & industrial motivation |
| **2** | System Architecture & Software Scope | 0.5 min | Modular design, software-only prototype boundary |
| **3** | Dataset Ingestion & Target Leakage Prevention | 1.0 min | AI4I 2020 dataset, dropping failure flags & identifiers |
| **4** | Physics-Inspired Feature Engineering | 0.5 min | Power proxy, temperature delta, overstrain proxy |
| **5** | Model Training & Evaluation Results | 1.0 min | Random Forest, class imbalance, confusion matrix, metrics |
| **6** | Model Explainability & Feature Importance | 0.5 min | Top features (Gini/Permutation) & non-causality disclaimer |
| **7** | Live Demonstration (CLI & Streamlit) | 1.5 min | Healthy vs stress inference, simulation, database audit |
| **8** | Limitations & Future Engineering Roadmap | 0.5 min | Scope honesty (no RUL, uncalibrated score) & extension pathways |

---

## Detailed Slide Content & Talking Points

### Slide 1: Title & Industrial Motivation (0:00 - 0:30)
- **Slide Title:** AI-Powered Predictive Maintenance System for IoT Devices
- **Key Points:**
  - Modern milling and CNC machining operations suffer heavy financial losses from unscheduled equipment downtime and tool breakage.
  - Traditional maintenance paradigms (reactive run-to-failure vs. calendar preventive schedules) are either hazardous or inefficient.
  - Predictive Maintenance (PdM) leverages multi-sensor operating telemetry to identify degradation patterns before catastrophic breakdown.
- **Presenter Cue:**  
  *"Good morning. My project presents an AI-powered predictive maintenance system that analyzes multi-sensor machine telemetry to detect impending failures in manufacturing equipment."*

---

### Slide 2: Software-Only System Architecture (0:30 - 1:00)
- **Slide Title:** System Architecture & Software Scope
- **Key Points:**
  - **Educational Prototype Scope:** Designed as a clean software-only prototype; does not require physical milling hardware or microcontrollers.
  - **Modular Pipeline Structure:**
    - Data Ingestion & Quality Audit (`src/data_loader.py`)
    - Leakage Safeguards & Preprocessing (`src/preprocessing.py`)
    - Feature Engineering (`src/feature_engineering.py`)
    - Model Pipeline & Provenance (`src/train_model.py`, `src/utils.py`)
    - Inference & Bounds Validation (`src/predictor.py`)
    - Multi-Tier Alert Engine (`src/alert_engine.py`)
    - Parameterized Persistence (`src/database.py` with SQLite)
    - Interactive Dashboard (`app.py` with Streamlit)
  - **Verification:** Backed by 48 automated tests passing in ~2 seconds.

---

### Slide 3: Dataset Ingestion & Target Leakage Prevention (1:00 - 2:00)
- **Slide Title:** Dataset Properties & Rigorous Leakage Prevention
- **Key Points:**
  - **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (10,000 records, 14 features, 0 null values, 0 duplicate rows).
  - **Severe Class Imbalance:** Only 339 failure records (3.39%) vs 9,661 healthy records.
  - **Target Leakage Safeguard:**
    - The raw dataset includes five breakdown cause flags: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
    - These flags are downstream components of failure. In real-time factory operations, an operator cannot know the breakdown cause prior to failure.
    - Including these flags produces artificial 100% test accuracy that fails in real inference.
    - Our pipeline strictly drops all five failure flags and arbitrary identifiers (`UDI`, `Product ID`) before model fitting.

---

### Slide 4: Physics-Inspired Feature Engineering (2:00 - 2:30)
- **Slide Title:** Feature Engineering: Embedding Physical Domain Knowledge
- **Key Points:**
  - Enhanced raw sensor data by deriving three physics-inspired domain features:
    1. **`Temp_Difference_K`** ($Process\ Temp - Air\ Temp$): Thermal dissipation gradient across cutting zone.
    2. **`Power_Proxy_W`** ($Torque \times Speed \times \frac{2\pi}{60}$): Spindle mechanical cutting power ($P = \tau \cdot \omega$).
    3. **`Overstrain_Proxy`** ($Tool\ Wear \times Torque$): Cumulative mechanical stress on worn cutting inserts.
  - Transformations are strictly identical during training, evaluation, CLI inference, and dashboard simulation.

---

### Slide 5: Model Training & Evaluation Results (2:30 - 3:30)
- **Slide Title:** Model Performance: Handling Severe Class Imbalance
- **Key Points:**
  - **Model:** Random Forest Classifier (100 trees) with `class_weight='balanced_subsample'` and stratified 80/20 train/test split (persisted in `models/train_test_split.json`).
  - **Held-Out Test Results (2,000 observations: 1,932 Normal, 68 Failures):**
    - **Accuracy:** `98.55%` (Baseline non-learning accuracy: 96.60%)
    - **Recall (Failure Class):** `82.35%` (56 of 68 actual failures detected)
    - **Precision (Failure Class):** `76.71%` (56 of 73 predicted failures genuine)
    - **F1-Score:** `79.43%`
    - **ROC-AUC:** `0.9845` | **PR-AUC:** `0.8672`
  - **Confusion Matrix:** `TN: 1,915 | FP: 17 | FN: 12 | TP: 56`
  - **Industrial Context:** In manufacturing, False Positives (brief inspection) are vastly preferred over False Negatives (catastrophic spindle destruction).

---

### Slide 6: Model Explainability (3:30 - 4:00)
- **Slide Title:** Explainability: Gini & Permutation Feature Importance
- **Key Points:**
  - **Top Features by Importance:**
    1. `Power_Proxy_W` (19.01%)
    2. `Torque [Nm]` (18.45%)
    3. `Rotational speed [rpm]` (18.20%)
    4. `Tool wear [min]` (14.15%)
    5. `Overstrain_Proxy` (12.29%)
  - **Permutation Importance:** Confirms that mechanical power and rotational speed cause the steepest performance drops when randomly shuffled.
  - **Scientific Disclaimer:** Feature importance reflects algorithmic tree split reliance; it is not proof of physical causation.

---

### Slide 7: Live System Demonstration (4:00 - 5:30)
- **Slide Title:** Live Demonstration: CLI Inference & Streamlit Dashboard
- **Suggested Demonstration Steps:**
  1. **Quick Provenance Check:**
     ```bash
     python main.py check
     ```
     *(Highlight matching SHA-256 hashes across dataset, model, and evaluation metrics).*
  2. **Contrasting Predictions:**
     - **Nominal State:**
       ```bash
       python main.py predict --type M --air-temp 298.5 --process-temp 308.2 --rpm 1500 --torque 40.0 --tool-wear 50
       ```
       $\rightarrow$ **NORMAL (0)**, Score: `0.0001`, Alert: `NORMAL`.
     - **High-Stress State:**
       ```bash
       python main.py predict --type L --air-temp 300.0 --process-temp 305.0 --rpm 1300 --torque 68.0 --tool-wear 240
       ```
       $\rightarrow$ **FAILURE (1)**, Score: `0.8131`, Alert: `CRITICAL`.
  3. **Streamlit UI Walkthrough:**
     - Launch `streamlit run app.py` (or showcase running UI).
     - Show **Model Performance** tab with dynamic confusion matrix.
     - Show **Manual Diagnostics** form with input bounds checking.
     - Show **Database Audit** tab distinguishing `SIMULATED (Demo)` from `MANUAL (Diagnostic)` records.

---

### Slide 8: Limitations & Future Engineering Roadmap (5:30 - 6:30)
- **Slide Title:** System Limitations & Future Extensions
- **Honest System Boundaries:**
  - **Point-in-Time Classification Only:** Evaluates instantaneous sensor states; does **not** predict Remaining Useful Life (RUL) or forecast time-to-failure.
  - **Uncalibrated Risk Score:** Score is the tree voting fraction, not an empirical Bayesian failure probability.
  - **Software Demonstration:** Telemetry simulator provides synthetic demonstration bursts labeled with `is_simulated = True`.
  - **Educational Prototype:** Academic demonstration, not certified for industrial safety-critical equipment control.
- **Future Extension Pathways:**
  - Ingestion of live machine data via MQTT or OPC-UA protocols.
  - Deep sequential modeling (LSTM / TCN) for continuous run-to-failure RUL prediction.
  - Model probability calibration (Platt scaling / Isotonic regression).
  - Edge deployment to industrial micro-gateways (Siemens IOT2050 / Raspberry Pi).

---

## Presenter Tips for Viva Success

1. **Be Honest About Scope:** If an examiner asks *"Did you test this on a real CNC machine?"*, immediately answer: *"No, sir/ma'am. This is an academic software-only prototype using the UCI AI4I 2020 benchmark dataset and a synthetic telemetry generator."* Examiners respect transparency.
2. **Emphasize Target Leakage:** If asked why baseline accuracy was improved, explain why dropping `TWF`, `HDF`, `PWF`, `OSF`, `RNF` was necessary to prevent target leakage.
3. **Know Your Metrics:** Do not just say *"Accuracy is 98.55%"*. Immediately add: *"However, because the naive baseline is 96.60%, we prioritize our F1-score of 79.43% and PR-AUC of 0.8672 to properly evaluate the minority failure class."*
4. **Distinguish Score from Probability:** Emphasize that the Random Forest score represents the ensemble vote proportion, not a calibrated real-world failure probability.
