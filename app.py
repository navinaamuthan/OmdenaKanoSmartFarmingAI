import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models with error handling for Streamlit Community Cloud
@st.cache_data
def load_models():
    try:
        # Attempt to load models with Git LFS paths
        original_soil = joblib.load('models/random_forest_model_original_soil.pkl')
        tweaked_soil = joblib.load('models/random_forest_model_tweaked_soil.pkl')
        original_onset = joblib.load('models/random_forest_model_original_onset.pkl')
        tweaked_onset = joblib.load('models/random_forest_model_tweaked_onset.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}. Using dummy models for testing.")
        # Temporary dummy models for testing (replace with retrained models)
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        original_soil = RandomForestRegressor()
        tweaked_soil = RandomForestRegressor()
        original_onset = RandomForestClassifier()
        tweaked_onset = RandomForestClassifier()
    return original_soil, tweaked_soil, original_onset, tweaked_onset

original_soil, tweaked_soil, original_onset, tweaked_onset = load_models()

# Functions for predictions
def planting_decision_soil(soil_moisture_pred):
    if 0.3 <= soil_moisture_pred <= 0.4:
        return "Plant"
    elif soil_moisture_pred < 0.3:
        return "Don't Plant (Too Dry)"
    else:
        return "Don't Plant (Too Wet)"

def predict_and_decide_soil(model, X):
    pred = model.predict(X)
    return [planting_decision_soil(p) for p in pred]

def predict_and_decide_onset(model, X):
    pred = model.predict(X)
    return ["Plant" if p == 1 else "Don't Plant" for p in pred]

# Streamlit app
st.title("Smart Farming Kano Dashboard")
st.markdown("This dashboard predicts soil moisture and rainy season onset for smart farming in Kano, Nigeria.")
st.markdown("Enter environmental data below to get predictions and planting decisions from both original and tweaked Random Forest models.")
st.markdown("Dataset sourced from: https://github.com/OmdenaAI/KanoNigeriaChapter_SmartFarming/blob/main/Dataset_Merged/KanoState.csv")

# Input form for both tasks
with st.form("prediction_form"):
    st.subheader("Enter Environmental Data")
    temp = st.slider("Temperature at 2m (°C)", -10.0, 40.0, 15.0)
    humidity = st.slider("Relative Humidity at 2m (%)", 0.0, 100.0, 50.0)
    precip = st.slider("Precipitation (mm/day)", 0.0, 50.0, 0.0)
    wind_speed = st.slider("Wind Speed at 2m (m/s)", 0.0, 20.0, 2.0)
    wind_dir = st.slider("Wind Direction at 2m (Degrees)", 0.0, 360.0, 180.0)
    surface_wetness = st.slider("Surface Soil Wetness", 0.0, 1.0, 0.5)
    root_wetness = st.slider("Root Zone Soil Wetness", 0.0, 1.0, 0.5)
    soil_lag1 = st.slider("Soil Moisture Yesterday", 0.0, 1.0, 0.3)
    soil_lag2 = st.slider("Soil Moisture 2 Days Ago", 0.0, 1.0, 0.3)
    soil_lag3 = st.slider("Soil Moisture 3 Days Ago", 0.0, 1.0, 0.3)
    temp_roll_mean = st.slider("7-Day Avg Temperature (°C)", -10.0, 40.0, 15.0)
    temp_roll_std = st.slider("7-Day Std Temperature (°C)", 0.0, 10.0, 1.0)
    humid_roll_mean = st.slider("7-Day Avg Humidity (%)", 0.0, 100.0, 50.0)
    humid_roll_std = st.slider("7-Day Std Humidity (%)", 0.0, 20.0, 5.0)

    submit = st.form_submit_button("Predict")

# Prediction logic
if submit:
    # Prepare input data for soil moisture
    soil_features = [
        temp, humidity, precip, wind_speed, wind_dir, surface_wetness, 
        root_wetness, soil_lag1, soil_lag2, soil_lag3, precip, temp
    ]
    soil_input = np.array([soil_features])
    soil_cols = [
        'Temperature_at_2-Meters (C)', 'Relative_Humidity_at_2_Meters (%)', 
        'Precipitation (mm/day)', 'Wind_Speed_at_2_Meters (m/s)', 
        'Wind_Direction_at_2_Meters (Degrees)', 'Surface_Soil_Wetness', 
        'Root_Zone_Soil_Wetness', 'Soil_Moisture_Lag1', 'Soil_Moisture_Lag2', 
        'Soil_Moisture_Lag3', 'Precip_Rolling_Mean', 'Temp_Range'
    ]
    soil_df = pd.DataFrame(soil_input, columns=soil_cols)

    # Prepare input data for onset
    onset_features = [
        temp, humidity, precip, wind_speed, wind_dir, surface_wetness, 
        root_wetness, temp_roll_mean, temp_roll_std, humid_roll_mean, humid_roll_std
    ]
    onset_input = np.array([onset_features])
    onset_cols = [
        'Temperature_at_2-Meters (C)', 'Relative_Humidity_at_2_Meters (%)', 
        'Precipitation (mm/day)', 'Wind_Speed_at_2_Meters (m/s)', 
        'Wind_Direction_at_2_Meters (Degrees)', 'Surface_Soil_Wetness', 
        'Root_Zone_Soil_Wetness', 'Temperature_Rolling_Mean', 'Temperature_Rolling_Std', 
        'Humidity_Rolling_Mean', 'Humidity_Rolling_Std'
    ]
    onset_df = pd.DataFrame(onset_input, columns=onset_cols)

    # Soil moisture predictions (emphasized design)
    st.markdown("""
        <h2 style='color: #2ecc71; text-align: center; font-size: 24px; font-weight: bold;'>
            🌱 Soil Moisture Predictions
        </h2>
        <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; border: 2px solid #2ecc71;'>
    """, unsafe_allow_html=True)
    soil_pred_original = original_soil.predict(soil_df)[0]
    soil_pred_tweaked = tweaked_soil.predict(soil_df)[0]
    soil_decision_original = predict_and_decide_soil(original_soil, soil_df)[0]
    soil_decision_tweaked = predict_and_decide_soil(tweaked_soil, soil_df)[0]
    
    st.write(f"**Original Model** - Predicted Soil Moisture: {soil_pred_original:.3f}, Decision: {soil_decision_original}")
    st.write(f"**Tweaked Model** - Predicted Soil Moisture: {soil_pred_tweaked:.3f}, Decision: {soil_decision_tweaked}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Rainy season onset predictions (emphasized design)
    st.markdown("""
        <h2 style='color: #e74c3c; text-align: center; font-size: 24px; font-weight: bold;'>
            🌧️ Rainy Season Onset Predictions
        </h2>
        <div style='background-color: #fff0f0; padding: 15px; border-radius: 10px; border: 2px solid #e74c3c;'>
    """, unsafe_allow_html=True)
    onset_pred_original = predict_and_decide_onset(original_onset, onset_df)[0]
    onset_pred_tweaked = predict_and_decide_onset(tweaked_onset, onset_df)[0]
    
    st.write(f"**Original Model** - Rainy Season Onset Decision: {onset_pred_original}")
    st.write(f"**Tweaked Model** - Rainy Season Onset Decision: {onset_pred_tweaked}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Simplified visualizations (without test set data)
    st.subheader("Model Performance Summary")
    st.write("Note: Visualizations of test set performance are not available in real-time predictions. Use the training notebook for detailed analysis.")

    # Feature importance (using model internals, no test data needed)
    st.subheader("Feature Importance (Top 5)")
    soil_importance_original = pd.DataFrame({
        'Feature': soil_cols,
        'Importance': original_soil.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)
    onset_importance_original = pd.DataFrame({
        'Feature': onset_cols,
        'Importance': original_onset.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)
    
    st.bar_chart(soil_importance_original.set_index('Feature'))
    st.bar_chart(onset_importance_original.set_index('Feature'))

# Footer
st.markdown("""
    Source code and dataset available at: https://github.com/navinaamuthan/OmdenaKanoSmartFarmingAI
""")
