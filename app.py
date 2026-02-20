import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Unbound Thickness ML Tool", layout="wide")

# ---------- Dark Academic Styling ----------
st.markdown("""
<style>
.big-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 25px;
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

st.markdown('<p class="big-title">Unbound Layer Thickness Prediction using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("Decision-support tool for pavement engineering applications")

# ---------- Upload Dataset ----------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df = df.select_dtypes(include=np.number)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ---------- Select Target ----------
    target = st.selectbox("Select Target Column", df.columns)

    # Separate features
    features = df.drop(columns=[target]).columns

    # ---------- Train Model (simple RF for deployment demo) ----------
    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    st.markdown('<p class="section-title">Input Parameters</p>', unsafe_allow_html=True)

    # ---------- Dynamic Input Fields ----------
    user_inputs = []
    cols = st.columns(2)

    for i, feature in enumerate(features):
        col = cols[i % 2]
        value = col.number_input(
            feature,
            float(X[feature].min()),
            float(X[feature].max()),
            float(X[feature].mean())
        )
        user_inputs.append(value)

    # ---------- Prediction ----------
    if st.button("Predict Thickness"):

        input_array = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        st.markdown("## Predicted Unbound Layer Thickness")

        # ---------- Gauge ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': f"{target}"},
            gauge={
                'axis': {'range': [float(y.min()), float(y.max())]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [float(y.min()), float(y.quantile(0.33))], 'color': "red"},
                    {'range': [float(y.quantile(0.33)), float(y.quantile(0.66))], 'color': "yellow"},
                    {'range': [float(y.quantile(0.66)), float(y.max())], 'color': "green"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ---------- Engineering Interpretation ----------
        st.markdown('<p class="section-title">Engineering Conclusion</p>', unsafe_allow_html=True)

        if prediction < y.quantile(0.33):
            conclusion = "Subgrade Quality: Weak – Higher thickness recommended."
        elif prediction < y.quantile(0.66):
            conclusion = "Subgrade Quality: Moderate – Suitable for medium traffic."
        else:
            conclusion = "Subgrade Quality: Good – Suitable for heavy traffic."

        st.success(conclusion)

        # ---------- Download Report ----------
        report = f"""
Unbound Layer Thickness Prediction Report

Input Parameters:
"""

        for f, val in zip(features, user_inputs):
            report += f"{f}: {val}\n"

        report += f"""
Predicted {target}: {round(prediction,2)}

Conclusion:
{conclusion}
"""

        st.download_button(
            label="Download Report",
            data=report,
            file_name="Thickness_Report.txt",
            mime="text/plain"
        )
