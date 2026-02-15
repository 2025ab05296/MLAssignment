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

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

st.title("🫀 Heart Disease Classification App")
st.write(
    "Upload a test dataset (CSV) and evaluate different trained machine learning models."
)

# ===============================
# Download Sample Dataset Section
# ===============================

sample_csv_url = "https://raw.githubusercontent.com/2025ab05296/MLAssignment/main/Heart.csv"

st.markdown(
    f"""
    📥 Don't have a dataset?  
    [Click here to download the sample Heart Disease dataset]({sample_csv_url})
    """
)

st.divider()

# ===============================
# Load Scaler
# ===============================

scaler = joblib.load("model/scaler.pkl")

# ===============================
# Model Selection
# ===============================

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

model_files = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# ===============================
# File Upload
# ===============================

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# ===============================
# Prediction Section
# ===============================

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    # Ensure correct target column
    if "Heart Disease" not in df.columns:
        st.error("Uploaded CSV must contain 'Heart Disease' column.")
    else:
        X = df.drop("Heart Disease", axis=1)
        y_true = df["Heart Disease"]

        # Convert categorical labels if needed
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
            y_prob = model.predict_proba(X_processed)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0

        # ===============================
        # Metrics
        # ===============================

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

        # ===============================
        # Confusion Matrix (Smaller + Centered)
        # ===============================

        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            annot_kws={"size": 12},
            ax=ax
        )

        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.tick_params(labelsize=10)

        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            st.pyplot(fig)
