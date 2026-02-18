import streamlit as st
import numpy as np
import pandas as pd
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
# Load Models (ET, RF, XGB only)
# --------------------------------------------------
@st.cache_resource
def load_models():

    required_files = [
        "et_model.pkl",
        "rf_model.pkl",
        "xgb_model.pkl",
        "scaler.pkl"
    ]

    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Missing file: {file}")
            st.stop()

    et = joblib.load("et_model.pkl")
    rf = joblib.load("rf_model.pkl")
    xgb = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return et, rf, xgb, scaler


et_model, rf_model, xgb_model, scaler = load_models()

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Input Parameters")

def user_input():

    data = {
        "Cementitious (%)": st.sidebar.number_input("Cementitious (%)", 0.0, 50.0, 5.0),
        "Ash (%)": st.sidebar.number_input("Ash (%)", 0.0, 100.0, 10.0),
        "Specific Gravity": st.sidebar.number_input("Specific Gravity", 1.5, 3.0, 2.65),
        "Liquid Limit (%)": st.sidebar.number_input("Liquid Limit (%)", 0.0, 100.0, 30.0),
        "Plastic Limit (%)": st.sidebar.number_input("Plastic Limit (%)", 0.0, 50.0, 20.0),
        "Plasticity Index (%)": st.sidebar.number_input("Plasticity Index (%)", 0.0, 50.0, 10.0),
        "Optimum Moisture Content (%)": st.sidebar.number_input("OMC (%)", 0.0, 30.0, 12.0),
        "Max Lab Dry Density": st.sidebar.number_input("MDD (kN/m³)", 10.0, 30.0, 18.0),
        "No.200 Passing (%)": st.sidebar.number_input("No.200 Passing (%)", 0.0, 100.0, 20.0),
        "No.40 Passing (%)": st.sidebar.number_input("No.40 Passing (%)", 0.0, 100.0, 30.0),
        "No.10 Passing (%)": st.sidebar.number_input("No.10 Passing (%)", 0.0, 100.0, 40.0),
        "No.4 Passing (%)": st.sidebar.number_input("No.4 Passing (%)", 0.0, 100.0, 60.0),
        "Three_Fourths Passing (%)": st.sidebar.number_input("3/4 Passing (%)", 0.0, 100.0, 80.0),
        "One_and_Half Passing (%)": st.sidebar.number_input("1.5 Passing (%)", 0.0, 100.0, 95.0)
    }

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("Input Summary")
st.write(input_df)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Unbound Layer Thickness"):

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Model predictions
    et_pred = et_model.predict(scaled_input)[0]
    rf_pred = rf_model.predict(scaled_input)[0]
    xgb_pred = xgb_model.predict(scaled_input)[0]

    # Ensemble average
    final_prediction = np.mean([et_pred, rf_pred, xgb_pred])

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Extra Trees (ET)", f"{et_pred:.2f}")
    col2.metric("Random Forest (RF)", f"{rf_pred:.2f}")
    col3.metric("XGBoost (XGB)", f"{xgb_pred:.2f}")

    st.success(f"Final Ensemble Thickness = {final_prediction:.2f}")
