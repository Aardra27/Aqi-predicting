import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("best_model.pkl")
features = joblib.load("feature_columns.pkl")

# Streamlit UI
st.set_page_config(page_title="AQI Predictor", layout="centered")
st.title("🌫️ Air Quality Index (AQI) Predictor")
st.markdown("Enter pollutant values to predict AQI:")

# Create input fields for each feature
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, format="%.2f")

# Predict AQI
if st.button("🔍 Predict AQI"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted AQI: **{round(prediction)}**")

    # AQI Category
    aqi = prediction
    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Satisfactory"
    elif aqi <= 200:
        category = "Moderate"
    elif aqi <= 300:
        category = "Poor"
    elif aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    st.info(f"AQI Category: **{category}**")
