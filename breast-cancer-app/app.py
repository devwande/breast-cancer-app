import streamlit as st
import pandas as pd
import joblib

model = joblib.load("breast_cancer_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Breast Cancer Risk Predictor", layout="centered")

st.title("🧬 Breast Cancer Susceptibility Predictor")
st.markdown("This app predicts the likelihood of breast cancer based on input features.")

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

if st.button("Predict Susceptibility"):
    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]
    prediction_label = "Malignant (Higher Risk)" if prediction == 0 else "Benign (Lower Risk)"
    
    st.subheader("Prediction Result:")
    st.success(f"🩺 The model predicts: **{prediction_label}**")
