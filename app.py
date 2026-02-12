import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.write("App Started Successfully")

# Page config
st.set_page_config(page_title="Heart Disease Classification", layout="wide")

st.title("🫀 Heart Disease Classification App")

st.write("Upload test dataset (CSV) and select a trained model to evaluate performance.")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Map model names to filenames
model_files = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check if Heart Disease exists
    if "Heart Disease" not in df.columns:
        st.error("Uploaded CSV must contain 'Heart Disease' column.")
    else:
        X = df.drop("Heart Disease", axis=1)
        y_true = df["Heart Disease"]

        # Convert text labels if needed
        if y_true.dtype == object:
            y_true = y_true.map({"Absence": 0, "Presence": 1})

        model = joblib.load(model_files[model_choice])

        # Scale only required models
        if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
            X_processed = scaler.transform(X)
        else:
            X_processed = X

        y_pred = model.predict(X_processed)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_processed)[:,1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.3f}")
        col1.metric("AUC", f"{auc:.3f}")

        col2.metric("Precision", f"{precision:.3f}")
        col2.metric("Recall", f"{recall:.3f}")

        col3.metric("F1 Score", f"{f1:.3f}")
        col3.metric("MCC", f"{mcc:.3f}")

        # Confusion Matrix
        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
