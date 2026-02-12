import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



df = pd.read_csv("heart.csv")

print("Dataset Shape:", df.shape)
print(df.head())


df["Heart Disease"] = df["Heart Disease"].map({
    "Absence": 0,
    "Presence": 1
})

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    tree_method='hist'
)
xgb_model.fit(X_train, y_train)

results = {
    "Logistic Regression": evaluate_model(log_model, X_test_scaled, y_test),
    "Decision Tree": evaluate_model(dt_model, X_test, y_test),
    "KNN": evaluate_model(knn_model, X_test_scaled, y_test),
    "Naive Bayes": evaluate_model(nb_model, X_test_scaled, y_test),
    "Random Forest": evaluate_model(rf_model, X_test, y_test),
    "XGBoost": evaluate_model(xgb_model, X_test, y_test)
}

comparison_df = pd.DataFrame(results).T
print("\nModel Comparison:\n")
print(comparison_df)

joblib.dump(log_model, "model/logistic.pkl")
joblib.dump(dt_model, "model/decision_tree.pkl")
joblib.dump(knn_model, "model/knn.pkl")
joblib.dump(nb_model, "model/naive_bayes.pkl")
joblib.dump(rf_model, "model/random_forest.pkl")
joblib.dump(xgb_model, "model/xgboost.pkl")
joblib.dump(scaler, "model/scaler.pkl")

comparison_df.to_csv("model/model_comparison.csv")
