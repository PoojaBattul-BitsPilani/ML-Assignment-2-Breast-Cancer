import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Breast Cancer Classification - ML Assignment 2")

# Model selection
st.subheader("Select Model")

model_option = st.selectbox(
    "Choose a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# File uploader
st.subheader("Upload CSV Test File")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model_files = {
        "Logistic Regression": "Logistic_Regression.pkl",
        "Decision Tree": "Decision_Tree.pkl",
        "KNN": "KNN.pkl",
        "Naive Bayes": "Naive_Bayes.pkl",
        "Random Forest": "Random_Forest.pkl",
        "XGBoost": "XGBoost.pkl"
    }

    model = joblib.load(f"models/{model_option}.pkl")

    y_pred = model.predict(X)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC: {mcc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)