# 💳 Credit Card Fraud Detection API

An end-to-end machine learning project for detecting fraudulent credit card transactions using **LightGBM** and **XGBoost**, with **FastAPI** for real-time predictions and **Docker** for deployment.

---

## 🚀 Overview

This project builds a fraud detection system on highly imbalanced transaction data and exposes it as a production-style API.

Key highlights:
- Handles severe class imbalance using **SMOTE**
- Compares multiple models (**XGBoost vs LightGBM**)
- Performs **threshold tuning** for better fraud detection
- Provides real-time predictions using **FastAPI**
- Includes **automated tests**
- Fully **Dockerized** for reproducible deployment

---

## 📊 Dataset

- **Dataset:** Credit Card Fraud Detection
- **Total transactions:** 284,807
- **Fraud cases:** 492
- **Fraud rate:** 0.173%

⚠️ Highly imbalanced dataset — special handling required.

---

## ⚙️ ML Pipeline

1. Data Cleaning
   - Removed duplicates

2. Feature Engineering
   - Scaled `Amount` and `Time` using `StandardScaler`

3. Handling Imbalance
   - Applied **SMOTE** on training data

4. Model Training
   - XGBoost
   - LightGBM (selected)

5. Evaluation Metrics
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

6. Threshold Tuning
   - Evaluated thresholds: `0.3, 0.4, 0.5, 0.6`
   - Selected: **0.4** (better fraud recall + balance)

---

## 🧠 Model Performance (LightGBM)

| Metric        | Value |
|--------------|------|
| ROC-AUC      | ~0.97 |
| Fraud Recall | ~0.79 |
| Fraud Precision | ~0.79 |

---

## 🏗️ Project Structure
fraud-detection/
│
├── data/
├── models/
├── notebooks/
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── predict.py
│
├── api/
│ └── main.py
│
├── tests/
│ └── test_api.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚡ How to Run

### 1. Clone repo

```bash
git clone <your-repo-link>
cd fraud-detection

