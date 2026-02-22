import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model (Pipeline)
# -----------------------------
# model = pickle.load(open('model.pkl', 'rb'))

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# -----------------------------
# Title Section
# -----------------------------
st.title("🩺 Diabetes Prediction System")
st.markdown("An ML-powered web application to predict if a patient is diabetic based on clinical features.")

st.divider()

# -----------------------------
# Sidebar - About Section
# -----------------------------
st.sidebar.write("""
This model was trained using:
- 4 Machine Learning classifiers were trained and tested
- The best performing model was selected based on accuracy.
- StandardScaler
- 8 clinical features

The model predicts whether a patient is diabetic based on medical inputs.
""")

st.sidebar.write("**Model Accuracy:** 97.5%")  # Replace with your actual accuracy

st.sidebar.markdown("[GitHub Repository](https://github.com/GayathriRamu)")

# -----------------------------
# Input Section (2 Columns)
# -----------------------------
st.subheader("Enter Patient Details")
st.markdown("Please fill in the following clinical features to get a diabetes prediction. Default values are provided for demonstration purposes.")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of times pregnant (0–10)"
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=450.0,
        value=100.0,
        help="Plasma glucose concentration (0–450)"
    )

    bp = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=80.0,
        help="Diastolic blood pressure (0–200)"
    )

    body_type = st.selectbox(
        "Body Type",
        ["Lean", "Average", "Overweight", "Obese"],
        help="Select the body type that best describes you."
    )

    if body_type == "Lean":
        skin = 20
    elif body_type == "Average":
        skin = 50
    elif body_type == "Overweight":
        skin = 70
    else:
        skin = 100

with col2:

    insulin_condition = st.selectbox(
        "Do you have insulin resistance or high insulin levels?",
        ["No / Not Known", "Yes"]
    )

    if insulin_condition == "Yes":
        insulin = 180
    else:
        insulin = 80

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=50.0,
        value=25.0,
        help="Body Mass Index (0–50)"
    )

    family_history = st.selectbox(
        "Family History of Diabetes",
        ["None", "One Parent", "Both Parents"],
        help="Indicates whether close family members have diabetes."
    )

    # Map family history to DPF score
    if family_history == "Both Parents":
        dpf = 1.5   # Higher hereditary risk
    elif family_history == "One Parent":
        dpf = 0.5   # Moderate hereditary risk
    else:
        dpf = 0.2   # Lower hereditary risk

    
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30,
        help="Age in years (0–120)"
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    features = np.array([[preg, glucose, bp, skin,
                          insulin, bmi, dpf, age]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    st.divider()
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Patient is likely Diabetic")
    else:
        st.success("✅ Low Risk: Patient is likely Not Diabetic")

    st.write(f"**Probability of Diabetes:** {probability[0][1]*100:.2f}%")

st.divider()
st.caption("⚠️ This tool is for educational purposes only and not a medical diagnosis.")