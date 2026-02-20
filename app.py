import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Unbound Thickness ML", layout="wide")

# ---------- Custom Dark Theme ----------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.big-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}
.stButton>button {
    background-color: #16a34a;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<p class="big-title">Unbound Layer Thickness Prediction using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("Decision-support tool for pavement engineering applications")

st.markdown('<p class="section-title">Input Parameters</p>', unsafe_allow_html=True)

# ---------- Input Fields ----------
col1, col2 = st.columns(2)

with col1:
    CBR = st.number_input("CBR (%)", 1.0, 50.0, 10.0)
    PI = st.number_input("Plasticity Index", 0.0, 40.0, 12.0)
    OMC = st.number_input("Optimum Moisture Content (%)", 5.0, 30.0, 15.0)

with col2:
    MDD = st.number_input("Maximum Dry Density (kN/m³)", 10.0, 25.0, 18.0)
    LL = st.number_input("Liquid Limit (%)", 10.0, 80.0, 40.0)
    Sand = st.number_input("Sand Content (%)", 0.0, 100.0, 50.0)

# ---------- Dummy Model (Replace with trained model later) ----------
def predict_thickness(inputs):
    # Placeholder prediction logic
    return 150 + 2*inputs[0] - 1.2*inputs[1] + 0.5*inputs[2]

if st.button("Predict Thickness"):

    inputs = np.array([CBR, PI, OMC, MDD, LL, Sand])
    prediction = predict_thickness(inputs)

    st.markdown("## Predicted Unbound Layer Thickness")

    # ---------- Gauge Chart ----------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Thickness (mm)"},
        gauge={
            'axis': {'range': [0, 400]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 150], 'color': "red"},
                {'range': [150, 250], 'color': "yellow"},
                {'range': [250, 400], 'color': "green"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------- Engineering Conclusion ----------
    st.markdown('<p class="section-title">Engineering Conclusion</p>', unsafe_allow_html=True)

    if prediction < 150:
        conclusion = "Subgrade Quality: Weak – Increase Thickness Recommended"
    elif prediction < 250:
        conclusion = "Subgrade Quality: Moderate – Suitable for Medium Traffic"
    else:
        conclusion = "Subgrade Quality: Good – Suitable for Heavy Traffic"

    st.success(conclusion)

    # ---------- Download Report ----------
    report = f"""
    Unbound Layer Thickness Prediction Report

    Input Parameters:
    CBR: {CBR}
    PI: {PI}
    OMC: {OMC}
    MDD: {MDD}
    LL: {LL}
    Sand Content: {Sand}

    Predicted Thickness: {round(prediction,2)} mm

    Conclusion:
    {conclusion}
    """

    st.download_button(
        label="Download Report",
        data=report,
        file_name="Thickness_Report.txt",
        mime="text/plain"
    )
