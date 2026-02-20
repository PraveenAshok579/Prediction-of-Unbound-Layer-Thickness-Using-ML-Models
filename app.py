import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Unbound Layer Thickness Prediction",
    layout="wide"
)

st.title("Unbound Layer Thickness Prediction System")
st.markdown("### LTPP-Based Machine Learning Model (Extra Trees)")

# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("et_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# Get Correct Feature Names From Scaler
# --------------------------------------------------
feature_names = list(scaler.feature_names_in_)

st.sidebar.header("Input Parameters")

# --------------------------------------------------
# Dynamic Input Creation (Prevents Feature Errors)
# --------------------------------------------------
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
# Show Input Summary
# --------------------------------------------------
st.subheader("Input Summary")
st.write(input_df)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("Predict Unbound Layer Thickness"):

    try:
        # Scale using trained scaler
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)

        st.success(f"Predicted Unbound Layer Thickness: {prediction[0]:.2f}")

    except Exception as e:
        st.error("Prediction failed. Please check feature inputs.")
        st.exception(e)

