import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Unbound Layer Thickness Prediction",
    layout="wide"
)

st.title("📊 Unbound Layer Thickness Prediction System")
st.markdown("Advanced Ensemble Machine Learning Framework")

# ---------------------------------------------------------
# Load Models and Scaler
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    et = joblib.load("et_model.pkl")
    rf = joblib.load("rf_model.pkl")
    xgb = joblib.load("xgb_model.pkl")
    stack = joblib.load("stack_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return et, rf, xgb, stack, scaler

et_model, rf_model, xgb_model, stack_model, scaler = load_models()

# ---------------------------------------------------------
# Sidebar Input Features
# ---------------------------------------------------------
st.sidebar.header("Input Parameters")

def user_input():
    data = {
        "Cementitious (%)": st.sidebar.number_input("Cementitious (%)", 0.0, 20.0, 5.0),
        "Ash (%)": st.sidebar.number_input("Ash (%)", 0.0, 60.0, 10.0),
        "Specific Gravity": st.sidebar.number_input("Specific Gravity", 1.0, 3.0, 2.65),
        "Liquid Limit (%)": st.sidebar.number_input("Liquid Limit (%)", 10.0, 150.0, 40.0),
        "Plastic Limit (%)": st.sidebar.number_input("Plastic Limit (%)", 5.0, 60.0, 20.0),
        "Plasticity Index (%)": st.sidebar.number_input("Plasticity Index (%)", 0.0, 50.0, 15.0),
        "OMC (%)": st.sidebar.number_input("Optimum Moisture Content (%)", 5.0, 30.0, 12.0),
        "MDD (kN/m3)": st.sidebar.number_input("Max Dry Density (kN/m3)", 10.0, 25.0, 18.0),
        "No_200 Passing (%)": st.sidebar.number_input("No.200 Passing (%)", 0.0, 100.0, 20.0),
        "No_40 Passing (%)": st.sidebar.number_input("No.40 Passing (%)", 0.0, 100.0, 30.0),
        "One_Half Passing (%)": st.sidebar.number_input("1/2\" Passing (%)", 0.0, 100.0, 80.0),
        "Three_Fourths Passing (%)": st.sidebar.number_input("3/4\" Passing (%)", 0.0, 100.0, 90.0)
    }

    return pd.DataFrame([data])

input_df = user_input()

# ---------------------------------------------------------
# Display Input Data
# ---------------------------------------------------------
st.subheader("Input Summary")
st.write(input_df)

# ---------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------
if st.button("Predict Thickness"):

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predictions
    et_pred = et_model.predict(scaled_input)[0]
    rf_pred = rf_model.predict(scaled_input)[0]
    xgb_pred = xgb_model.predict(scaled_input)[0]
    stack_pred = stack_model.predict(scaled_input)[0]

    # -----------------------------------------------------
    # Display Results
    # -----------------------------------------------------
    st.subheader("Predicted Unbound Layer Thickness (mm)")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Extra Trees (ET)", f"{et_pred:.2f}")
    col2.metric("Random Forest (RF)", f"{rf_pred:.2f}")
    col3.metric("XGBoost (XGB)", f"{xgb_pred:.2f}")
    col4.metric("Stacking Model", f"{stack_pred:.2f}")

    # -----------------------------------------------------
    # Model Comparison Plot
    # -----------------------------------------------------
    st.subheader("Model Comparison")

    models = ["ET", "RF", "XGB", "STACK"]
    predictions = [et_pred, rf_pred, xgb_pred, stack_pred]

    fig, ax = plt.subplots()
    ax.bar(models, predictions)
    ax.set_ylabel("Predicted Thickness (mm)")
    ax.set_title("Comparison of Ensemble Models")

    st.pyplot(fig)

    # -----------------------------------------------------
    # Simple Uncertainty Estimate
    # -----------------------------------------------------
    st.subheader("Prediction Spread (Model Variability)")

    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)

    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred

    st.write(f"Mean Prediction: {mean_pred:.2f} mm")
    st.write(f"Approximate 95% Interval: [{lower:.2f} , {upper:.2f}] mm")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("Developed using Ensemble Machine Learning Framework")
