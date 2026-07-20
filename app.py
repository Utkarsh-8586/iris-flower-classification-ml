"""
app.py
------
Streamlit web app for Iris Flower Classification.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

# ----------------------------------------------------------------------
# Load model, scaler, metadata
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler, metadata

model, scaler, metadata = load_artifacts()
target_names = metadata["target_names"]

SPECIES_INFO = {
    "setosa": "Small flower with short, wide petals. Native to North America / Arctic regions.",
    "versicolor": "Medium-sized, found in wetlands across eastern North America.",
    "virginica": "Larger flower with long petals, common in the southeastern US.",
}
SPECIES_IMG_COLOR = {"setosa": "#4C72B0", "versicolor": "#55A868", "virginica": "#C44E52"}

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🌸 Iris Flower Species Classifier")
st.write(
    f"This app predicts the species of an Iris flower from its sepal and petal "
    f"measurements using a trained **{metadata['best_model_name']}** model "
    f"(macro F1-score: **{metadata['f1_score']}** on held-out test data)."
)

st.divider()

# ----------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------
st.sidebar.header("Input Measurements (cm)")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0, 0.1)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.2, 0.1)

st.sidebar.divider()
st.sidebar.caption("Built with Scikit-Learn + Streamlit")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=metadata["feature_names"],
)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
predicted_species = target_names[prediction]

# probabilities (works for models with predict_proba)
proba = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(input_scaled)[0]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Your Input")
    st.table(input_df.T.rename(columns={0: "Value (cm)"}))

with col2:
    st.subheader("Prediction")
    st.markdown(f"### 🌼 **{predicted_species.capitalize()}**")
    st.caption(SPECIES_INFO.get(predicted_species, ""))

if proba is not None:
    st.subheader("Prediction Confidence")
    proba_df = pd.DataFrame({"Species": target_names, "Probability": proba})
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [SPECIES_IMG_COLOR.get(s, "#999") for s in target_names]
    ax.bar(proba_df["Species"], proba_df["Probability"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    for i, v in enumerate(proba_df["Probability"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    st.pyplot(fig)

st.divider()

# ----------------------------------------------------------------------
# Model comparison table (if available)
# ----------------------------------------------------------------------
comparison_path = os.path.join(BASE_DIR, "model_comparison_results.csv")
if os.path.exists(comparison_path):
    with st.expander("📊 See how all trained models compared"):
        comp_df = pd.read_csv(comparison_path)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        - **Dataset**: Fisher's Iris dataset (150 samples, 3 species, 4 features)
        - **Models compared**: KNN, Decision Tree, Logistic Regression, Random Forest, SVM
        - **Best model selected** by macro F1-score on a held-out test set
        - **Pipeline**: EDA → scaling → train/test split → training → evaluation → deployment
        """
    )
