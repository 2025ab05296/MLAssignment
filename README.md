# 🫀 Heart Disease Prediction using Machine Learning
---

## 📌 Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease based on clinical attributes.

This is a binary classification problem:

- 0 → Absence of Heart Disease  
- 1 → Presence of Heart Disease  

The project demonstrates:

- Data preprocessing  
- Model training  
- Model evaluation  
- Model comparison  
---

## 📊 Dataset Description

The Heart Disease dataset contains patient medical information used to predict heart disease.

**Dataset Characteristics:**

- Number of Instances: 900+  
- Number of Features: 13+  
- Target Variable: `Heart Disease`  
- Problem Type: Binary Classification  

**Example Features:**
- Age
- Sex
- Chest pain type
- BP
- Cholesterol
- FBS over 120
- EKG results
- Max HR
- Exercise angina
- ST depression
- Slope of ST
- Number of vessels fluro
- Thallium

The original target column contained categorical values ("Absence", "Presence") which were encoded to 0 and 1 before training.

---

## 🤖 Machine Learning Models Implemented

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble Boosting)  

---

## 📈 Evaluation Metrics Used

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 📊 Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------|----------|------|-----------|--------|-----|------|
| Logistic Regression | 0.852 | 0.899 | 0.857 | 0.835 | 0.846 | 0.713 |
| Decision Tree | 0.796 | 0.800 | 0.800 | 0.769 | 0.784 | 0.596 |
| KNN | 0.796 | 0.875 | 0.808 | 0.778 | 0.792 | 0.606 |
| Naive Bayes | 0.852 | 0.893 | 0.857 | 0.824 | 0.840 | 0.704 |
| Random Forest | 0.815 | 0.867 | 0.815 | 0.786 | 0.800 | 0.630 |
| XGBoost | 0.815 | 0.882 | 0.815 | 0.786 | 0.800 | 0.630 |

---

## 🔍 Observations on Model Performance

| ML Model | Observation |
|-----------|------------|
| Logistic Regression | Achieved highest overall performance. Dataset appears relatively linearly separable. |
| Decision Tree | Lower performance compared to ensemble methods. Slight overfitting observed. |
| KNN | Moderate performance. Sensitive to scaling and neighbor selection. |
| Naive Bayes | Performance close to Logistic Regression. Independence assumption worked reasonably well. |
| Random Forest | Improved over Decision Tree due to ensemble learning. |
| XGBoost | Similar to Random Forest. Boosting improved AUC slightly. |

---

## 📁 Project Structure

heart-disease-ml/
- app.py  
- models.py  
- requirements.txt  
- README.md  
- model/  
  - logistic.pkl  
  - decision_tree.pkl  
  - knn.pkl  
  - naive_bayes.pkl  
  - random_forest.pkl  
  - xgboost.pkl  
  - scaler.pkl  

---

## ▶ How to Run Locally

Install dependencies:

pip install -r requirements.txt

Run Streamlit app:

streamlit run app.py

Then open:

http://localhost:8501

---
---

# End-to-End Machine Learning Workflow Completed
