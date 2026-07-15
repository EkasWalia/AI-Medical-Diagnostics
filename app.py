import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model artifact
model = joblib.load('models/best_xgboost_model.pkl')

st.set_page_config(page_title="AI Medical Diagnostics", layout="centered")
st.title("🩺 Enterprise Disease Prediction Portal")
st.write("Input patient clinical metrics below to compute real-time diagnostic risk analysis.")

# Create clean user input fields arranged in columns
col1, col2 = st.columns(2)

with col1:
    glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
    bp = st.slider("Blood Pressure (mm Hg)", 0, 150, 70)
    bmi = st.slider("Body Mass Index (BMI)", 0.0, 60.0, 25.0)

with col2:
    insulin = st.slider("Insulin Level (μU/mL)", 0, 800, 80)
    age = st.slider("Patient Age (Years)", 1, 120, 30)
    pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

# Calculate the engineered feature on the fly
age_bmi_index = age * bmi

# Package data exactly how the model expects it
input_data = pd.DataFrame([[glucose, bp, insulin, bmi, pedigree, age, age_bmi_index]], 
                           columns=['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Age_BMI_Index'])

if st.button("Analyze Diagnostic Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk Detected: The patient has a **{probability*100:.2f}%** statistical probability of diabetes.")
    else:
        st.success(f"✅ Low Risk: The patient has a **{(1-probability)*100:.2f}%** structural probability of being healthy.")