import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

st.set_page_config(page_title="Unbound Layer Thickness ML Tool", layout="wide")

st.title("Unbound Layer Thickness Prediction Using Ensemble ML Models")
st.markdown("### Journal Supplementary Demonstration Tool")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df = df.select_dtypes(include=np.number)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train & Compare Models"):

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(objective='reg:squarederror', verbosity=0),
            "AdaBoost": AdaBoostRegressor()
        }

        results = []

        for name, model in models.items():

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)

            cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

            results.append([
                name,
                round(r2, 4),
                round(rmse, 4),
                round(mape, 4),
                round(cv_r2, 4)
            ])

        results_df = pd.DataFrame(
            results,
            columns=["Model", "Test R2", "RMSE", "MAPE", "CV R2"]
        )

        st.write("### Model Performance Comparison")
        st.dataframe(results_df)

        # Download option
        excel_file = "ML_Model_Results.xlsx"
        results_df.to_excel(excel_file, index=False)

        with open(excel_file, "rb") as f:
            st.download_button(
                label="Download Results as Excel",
                data=f,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
