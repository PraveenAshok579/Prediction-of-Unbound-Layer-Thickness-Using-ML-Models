import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Unbound Layer Thickness Prediction",
    layout="wide"
)

st.title("Unbound Layer Thickness Prediction System")
st.markdown("Advanced Ensemble Machine Learning Framework")

# --------------------------------------------------
# Load Model & Scaler Safely
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("et_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error("Model files not found. Please check repository.")
        st.stop()

model, scaler = load_artifacts()

# --------------------------------------------------
# Get Correct Feature Names From Training
# --------------------------------------------------
feature_names = list(scaler.feature_names_in_)

# --------------------------------------------------
# Sidebar Inputs (Auto-Matched to Model)
# --------------------------------------------------
st.sidebar.header("Input Parameters")

def user_input():
    input_dict = {}

    for feature in feature_names:
        input_dict[feature] = st.sidebar.number_input(
            label=feature.replace("_", " "),
            value=0.0
        )

    return pd.DataFrame([input_dict])

input_df = user_input()

# --------------------------------------------------
# Display Input Summary
# --------------------------------------------------
st.subheader("Input Summary")
st.dataframe(input_df)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
if st.button("Predict Unbound Layer Thickness"):

    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)

        st.success(
            f"Predicted Unbound Layer Thickness: {prediction[0]:.2f}"
        )

    except Exception as e:
        st.error("Prediction failed. Please verify inputs.")
        st.exception(e)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown("Developed for LTPP-based Unbound Layer Thickness Prediction")
