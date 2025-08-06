import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("best_model.pkl")
features = joblib.load("feature_columns.pkl")

# Streamlit UI
st.title("Air Quality Index (AQI) Predictor")

st.markdown("### Enter pollutant concentrations:")
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}:", step=0.01)

# Predict AQI
if st.button("Predict AQI"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    # Display numerical AQI
    st.success(f"Predicted AQI: {round(prediction)}")

    # Categorize AQI
    def categorize_aqi(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Satisfactory"
        elif aqi <= 200:
            return "Moderate"
        elif aqi <= 300:
            return "Poor"
        elif aqi <= 400:
            return "Very Poor"
        else:
            return "Severe"

    category = categorize_aqi(prediction)
    st.info(f"AQI Category: {category}")
