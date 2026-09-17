# Enterprise Machine Learning Production Portfolio

This repository contains a suite of enterprise-grade Machine Learning pipelines designed to solve high-impact financial, operational, and regulatory business challenges.

---

## 📊 Projects Overview

### 1. Enterprise Credit Risk & Default Prediction System (`Credit_Default_Risk_Pipeline.ipynb`)
* **Business Context:** SME Credit Default assessment operating under strict regulatory constraints (no demographic bias) and severe class imbalance (~2%).
* **Key Features:** Native missing value handling (40% missingness), feature engineering (DSCR Proxy), TreeSHAP explainability, 4-tier risk segmentation, and **ONNX** model export for sub-15ms local execution.

### 2. Financial Fraud Detection Pipeline
* **Business Context:** Real-time transaction monitoring on highly imbalanced financial streams.
* **Key Features:** Cost-sensitive learning, Precision-Recall curve optimization, stratified data splitting, and interactive performance dashboards.

### 3. Customer Churn Prediction & ROI Optimization (`Customer_Churn_Prediction_Pipeline.ipynb`)
* **Business Context:** Churn risk modeling linked directly to financial retention strategies.
* **Key Features:** Probability threshold tuning, LightGBM classification, and direct ROI mapping for targeted marketing campaigns.

### 4. Sales & Real Estate Valuation Regression (`Sales_Forecasting_ML_Pipeline.ipynb`)
* **Business Context:** Automated Valuation Modeling (AVM) for property price estimation.
* **Key Features:** Multi-model regression benchmarking (Ridge, Random Forest, XGBoost) achieving **$R^2 = 0.83$** and MAE reduction down to $31.3k.

---

## 🛠️ Tech Stack & Tools
* **Languages & Frameworks:** Python, Scikit-Learn, LightGBM, XGBoost, ONNX.
* **Explainability & Optimization:** SHAP (TreeSHAP), Precision-Recall Thresholding.
* **Data Processing & Viz:** Pandas, NumPy, Seaborn, Matplotlib.

---

## 🚀 Deployment & Production Readiness
All champion models are optimized for ultra-low latency deployment (<15ms) on resource-constrained hardware (8GB RAM, No GPU) via ONNX Runtime and local FastAPI wrappers.
