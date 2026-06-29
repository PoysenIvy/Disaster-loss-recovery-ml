# Financial Risk & Loss Forecasting Using XGBoost Regression

## 📈 Project Overview
When high-impact, low-frequency events occur, operational systems face massive data volatility spikes. This project develops an end-to-end predictive machine learning pipeline using Python and XGBoost to forecast continuous financial loss and recovery amounts. 

While applied here to catastrophic property damage claims within the insurance sector, the core architecture serves as a scalable blueprint for any enterprise looking to automate high-stakes risk triaging, optimize resource allocation, and minimize financial baseline prediction errors.

## 🛠️ Tech Stack & Core Libraries
* **Language:** Python
* **Modeling & Ensembles:** XGBoost, Scikit-Learn
* **Data Engineering & Analysis:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

## 📊 Machine Learning Workflow
1. **Data Engineering & EDA:** Handled highly skewed continuous target distributions and engineered a composite *Vulnerability Index* to capture non-linear relationships between feature variables.
2. **Pipeline Development:** Built a robust data preprocessing workflow utilizing Scikit-Learn pipelines to handle categorical encoding and feature scaling seamlessly.
3. **Advanced Modeling (XGBoost):** Evaluated baseline linear models against an optimized XGBoost Regressor. Implemented hyperparameter tuning (Learning Rate, Max Depth, and Estimators) via Grid Search to maximize model generalization.
4. **Model Explainability:** Extracted global feature importance metrics to provide transparent, interpretable data insights regarding the primary drivers of financial loss.
5. **Evaluation:** The finalized XGBoost model achieved an R² score of 0.89, outperforming baseline models and demonstrating clear business utility for automated risk evaluation.

## 📁 Repository Structure
* `/data`: Simulated and processed risk exposure datasets.
* `/notebooks`: Comprehensive Jupyter notebooks tracking EDA, feature engineering, and model validation.
* `/src`: Production-ready, modular Python scripts for data cleaning and model inference.
