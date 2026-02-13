🫀 Heart Disease Prediction using Machine Learning
📌 Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease in a patient based on medical and clinical attributes.

This is a binary classification problem, where:

0 → Absence of Heart Disease

1 → Presence of Heart Disease

The project demonstrates a complete end-to-end machine learning workflow including:

Data preprocessing

Model training

Model evaluation

Model comparison

Interactive web application deployment using Streamlit

📊 Dataset Description

The Heart Disease dataset contains patient medical information used to predict the presence of heart disease.

Dataset Characteristics

Number of Instances: 900+

Number of Features: 13+

Target Variable: target

Problem Type: Binary Classification

Example Features

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol

Fasting blood sugar

Resting ECG

Maximum heart rate achieved

Exercise-induced angina

ST depression (oldpeak)

Slope of ST segment

Number of major vessels (ca)

Thalassemia

The original target column contained categorical labels (Absence, Presence) which were encoded into numerical values (0 and 1) before model training.

🤖 Machine Learning Models Implemented

The following six classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Boosting Model)

📈 Evaluation Metrics Used

For each model, the following evaluation metrics were calculated:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

📊 Model Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.852	0.899	0.857	0.835	0.846	0.713
Decision Tree	0.796	0.800	0.800	0.769	0.784	0.596
KNN	0.796	0.875	0.808	0.778	0.792	0.606
Naive Bayes	0.852	0.893	0.857	0.824	0.840	0.704
Random Forest	0.815	0.867	0.815	0.786	0.800	0.630
XGBoost	0.815	0.882	0.815	0.786	0.800	0.630
🔍 Observations on Model Performance
ML Model	Observation about Model Performance
Logistic Regression	Achieved the highest overall performance with best Accuracy and MCC. Indicates the dataset is relatively linearly separable.
Decision Tree	Lower performance compared to ensemble methods. Single-tree structure may have slightly overfitted.
KNN	Moderate performance. Sensitive to scaling and choice of neighbors.
Naive Bayes	Performed very close to Logistic Regression. Feature independence assumption worked reasonably well.
Random Forest	Improved performance over Decision Tree due to ensemble learning but did not surpass Logistic Regression.
XGBoost	Similar performance to Random Forest. Boosting improved AUC slightly but did not significantly increase accuracy.
📌 Key Insight

Logistic Regression performed best among all models, suggesting that the dataset does not require highly complex ensemble methods for optimal performance.

🌐 Streamlit Web Application Features

The project includes an interactive Streamlit web application with the following features:

Upload custom test dataset (CSV)

Select trained model from dropdown

Display evaluation metrics

Display confusion matrix
