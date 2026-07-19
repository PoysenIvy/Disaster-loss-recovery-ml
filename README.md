# Property Insurance Risk Classification & Claims Prediction

## 📈 Project Overview
Insurance operations face a persistent challenge when it comes to accurately identifying which properties are highly likely to file claims while dealing with massive data imbalance (as the vast majority of properties never file a claim). 

Using a real-world home insurance dataset, this project develops an end-to-end predictive machine learning pipeline using Python and XGBoost to classify high-risk properties (specifically predicting whether a property will file a claim within a 3-year window). This serves as a scalable framework for risk evaluation, helping underwriters automate risk triaging and allocate resources more effectively.


## Tech Stack & Core Libraries
* **Language:** Python
* **Modeling & Ensembles:** XGBoost, Scikit-Learn
* **Data Engineering & Analysis:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

## Machine Learning Workflow
1. **Data Engineering & EDA:** Handled highly imbalanced class distributions (where active claims make up a small minority of the data) and analyzed key risk factors like roof/wall construction materials, safety features, and geographic hazards.
2. **Pipeline Development:** Created a structured preprocessing workflow in Scikit-Learn using `ColumnTransformer` to handle categorical encoding (for building types and materials) and numeric scaling (for building dimensions and age) in one seamless pipeline.
3. **Advanced Modeling (XGBoost):** Evaluated standard baseline models against an optimized XGBoost Classifier. Handled class imbalance using parameter tuning (`scale_pos_weight`) and optimized hyperparameters via Grid Search to maximize model generalization.
4. **Model Explainability:** Extracted feature importance scores from the final model to identify the primary real-world drivers behind property risk (e.g., how factors like geographic location or safety systems affect claim likelihood).
5. **Evaluation:** Optimized the classifier targeting ROC-AUC and Precision-Recall curves to ensure we minimize false negatives—making sure high-risk properties aren't missed by the automation.

## 📁 Repository Structure
* `/data`: Historical property insurance CSV dataset.
* `/notebooks`: Step-by-step Jupyter notebook detailing exploratory analysis, preprocessing trials, and model evaluation.
* `/src`: Modular Python scripts for reusable cleaning, processing, and prediction.
