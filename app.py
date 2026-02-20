import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Unbound Layer Thickness ML", layout="wide")

# ------------------ Custom Styling ------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
}
.sub-text {
    text-align: center;
    font-size: 16px;
    color: #a1a1aa;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 25px;
}
.feature-box {
    background-color: #111827;
    padding: 12px;
    border-radius: 8px;
}
.stButton>button {
    background-color: #16a34a;
    color: white;
    height: 3em;
    border-radius: 8px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Unbound Layer Thickness Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Decision-support tool for pavement design | Output Unit: mm</p>', unsafe_allow_html=True)

# ------------------ Load Dataset ------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("LTPP_Unbound_Thickness_ML.csv")
    df = df.select_dtypes(include=np.number)

    target = "REPR_THICKNESS"   # CHANGE if needed

    X = df.drop(columns=[target])
    y = df[target]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    return model, X.columns, X, y

model, features, X_data, y_data = load_model()

# ------------------ Feature Inputs ------------------
st.markdown('<p class="section-title">Input Parameters</p>', unsafe_allow_html=True)

cols = st.columns(3)
user_inputs = []

for i, feature in enumerate(features):
    col = cols[i % 3]
    with col:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        value = st.number_input(
            feature,
            float(X_data[feature].min()),
            float(X_data[feature].max()),
            float(X_data[feature].mean())
        )
        st.markdown('</div>', unsafe_allow_html=True)
        user_inputs.append(value)

# ------------------ Prediction ------------------
if st.button("Predict Unbound Layer Thickness"):

    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    st.markdown("## Predicted Unbound Layer Thickness (inch)")

    # --------- Custom Semi Circular Meter ----------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        number={'suffix': " mm"},
        gauge={
            'shape': "angular",
            'axis': {'range': [float(y_data.min()), float(y_data.max())]},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range': [float(y_data.min()), float(y_data.quantile(0.33))], 'color': "#dc2626"},
                {'range': [float(y_data.quantile(0.33)), float(y_data.quantile(0.66))], 'color': "#facc15"},
                {'range': [float(y_data.quantile(0.66)), float(y_data.max())], 'color': "#16a34a"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ------------------ Engineering Conclusion ------------------
    st.markdown('<p class="section-title">Engineering Conclusion</p>', unsafe_allow_html=True)

    if prediction < y_data.quantile(0.33):
        conclusion = "Subgrade Condition: Weak – Higher structural thickness recommended."
    elif prediction < y_data.quantile(0.66):
        conclusion = "Subgrade Condition: Moderate – Suitable for medium traffic loading."
    else:
        conclusion = "Subgrade Condition: Good – Suitable for heavy traffic applications."

    st.success(conclusion)


