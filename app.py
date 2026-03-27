st.title("🏠 Bangalore House Price Prediction App")
st.write("Enter the details below to predict the house price.")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load new 10-feature model
model = joblib.load("ridge_10feature_model.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1500)
garage_cars = st.number_input("Garage Cars", value=2)
total_bsmt_sf = st.number_input("Total Basement SF", value=800)
first_flr_sf = st.number_input("1st Floor SF", value=1000)
year_built = st.number_input("Year Built", value=2000)
total_bath = st.number_input("Total Bathrooms", value=2.0)
bedroom_abvgr = st.number_input("Bedrooms Above Ground", value=3)
fireplaces = st.number_input("Number of Fireplaces", value=1)
lot_area = st.number_input("Lot Area", value=8000)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'Gr Liv Area': [gr_liv_area],
        'Garage Cars': [garage_cars],
        'Total Bsmt SF': [total_bsmt_sf],
        '1st Flr SF': [first_flr_sf],
        'Year Built': [year_built],
        'TotalBath': [total_bath],
        'Bedroom AbvGr': [bedroom_abvgr],
        'Fireplaces': [fireplaces],
        'Lot Area': [lot_area]
    })

    prediction_log = model.predict(input_data)
    prediction_real = np.expm1(prediction_log)

    st.success(f"Predicted House Price: ${prediction_real[0]:,.2f}")
