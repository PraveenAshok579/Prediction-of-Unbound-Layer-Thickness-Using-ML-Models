import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Unbound Layer Thickness ML Tool", layout="wide")

# -------------------------------------------------
# Custom Styling
# -------------------------------------------------
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
    margin-bottom: 30px;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 15px;
}
.feature-box {
    background-color: #111827;
    padding: 10px;
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
st.markdown('<p class="sub-text">Decision-support tool for pavement design | Output Unit: inches</p>', unsafe_allow_html=True)

# -------------------------------------------------
# Load Dataset & Train Model (Cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("LTPP_Unbound_Thickness_ML.csv")
    df = df.select_dtypes(include=np.number)

    target = df.columns[-1]   # automatically detect target
    X = df.drop(columns=[target])
    y = df[target]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    return model, X.columns, X, y, target

model, features, X_data, y_data, target_name = load_model()

# -------------------------------------------------
# Input Section
# -------------------------------------------------
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

# -------------------------------------------------
# Modern Radial Meter Function
# -------------------------------------------------
def modern_meter(prediction, y_min, y_max):

    percentage = (prediction - y_min) / (y_max - y_min)
    percentage = max(0, min(percentage, 1))

    fig = go.Figure()

    # Background circle
    fig.add_trace(go.Pie(
        values=[1],
        hole=0.75,
        marker_colors=['#1f2937'],
        textinfo='none',
        hoverinfo='none'
    ))

    # Progress arc
    fig.add_trace(go.Pie(
        values=[percentage, 1 - percentage],
        hole=0.75,
        rotation=90,
        marker_colors=['#22c55e', 'rgba(0,0,0,0)'],
        textinfo='none',
        hoverinfo='none'
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        annotations=[
            dict(
                text=f"<b>{round(prediction,2)} in</b>",
                x=0.5,
                y=0.5,
                font_size=32,
                showarrow=False,
                font_color="white"
            )
        ],
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a'
    )

    return fig

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Unbound Layer Thickness"):

    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    st.markdown("## Predicted Unbound Layer Thickness (inches)")

    fig = modern_meter(
        prediction,
        float(y_data.min()),
        float(y_data.max())
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Engineering Interpretation
    # -------------------------------------------------
    st.markdown('<p class="section-title">Engineering Interpretation</p>', unsafe_allow_html=True)

    q1 = y_data.quantile(0.33)
    q2 = y_data.quantile(0.66)

    if prediction < q1:
        conclusion = "Subgrade Condition: Weak – Higher structural thickness recommended."
    elif prediction < q2:
        conclusion = "Subgrade Condition: Moderate – Suitable for medium traffic loading."
    else:
        conclusion = "Subgrade Condition: Good – Suitable for heavy traffic applications."

    st.success(conclusion)
