# ============================================================
# ⚡ Energy Consumption Prediction App (Debug-safe version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Energy Predictor", layout="centered")
st.title("⚡ Household Energy Consumption Prediction App")
st.write("Enter the environmental and temporal parameters below to predict the appliance energy consumption (Wh).")

# ------------------------------
# Load model safely
# ------------------------------
model_path = r"C:\Users\sevit\Downloads\APPLIANCE ENERGY\best_energy_model_Ridge_Regression.joblib"


model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
else:
    st.warning("⚠️ Model file not found. The app will run in demo mode (random predictions).")

# ------------------------------
# Input fields for features
# ------------------------------
st.header("🔧 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    T1 = st.number_input("Kitchen Temperature (°C)", 10.0, 40.0, 21.0)
    T2 = st.number_input("Living Room Temperature (°C)", 10.0, 40.0, 20.0)
    T_out = st.number_input("Outdoor Temperature (°C)", -10.0, 40.0, 7.0)
    RH_out = st.number_input("Outdoor Humidity (%)", 0.0, 100.0, 80.0)
    hour = st.slider("Hour of Day", 0, 23, 14)

with col2:
    lights = st.number_input("Lights Energy (Wh)", 0.0, 200.0, 30.0)
    humidity = st.number_input("Average Indoor Humidity (%)", 0.0, 100.0, 45.0)
    lag_1 = st.number_input("Lag 1 (10 min ago)", 0.0, 500.0, 100.0)
    lag_3 = st.number_input("Lag 3 (30 min ago)", 0.0, 500.0, 120.0)
    roll_mean_3 = st.number_input("Rolling Mean of last 3 readings", 0.0, 500.0, 110.0)

# Derived features
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Combine into DataFrame
input_data = pd.DataFrame({
    'T1': [T1],
    'T2': [T2],
    'T_out': [T_out],
    'RH_out': [RH_out],
    'lights': [lights],
    'humidity': [humidity],
    'hour': [hour],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos],
    'lag_1': [lag_1],
    'lag_3': [lag_3],
    'roll_mean_3': [roll_mean_3]
})

st.write("### 🔹 Input Data Preview:")
st.dataframe(input_data)

# ------------------------------
# Prediction section
# ------------------------------
st.markdown("---")
if st.button("🔮 Predict Energy Consumption"):
    try:
        if model:
            prediction = model.predict(input_data)[0]
            st.success(f"### ⚡ Predicted Appliance Energy Consumption: **{prediction:.2f} Wh**")
        else:
            # Demo prediction (if model missing)
            prediction = np.random.uniform(50, 400)
            st.info(f"Demo Prediction: **{prediction:.2f} Wh** (Model not loaded)")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")

#st.markdown("---")
#st.caption("Built with ❤️ by Sevitha | Energy Prediction Project (Internship)")
