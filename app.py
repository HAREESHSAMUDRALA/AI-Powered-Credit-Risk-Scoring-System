import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model & scaler
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names (must match training time)
feature_names = ["person_income", "person_age", "person_home_ownership", "loan_intent",
                 "loan_grade", "loan_amount", "loan_int_rate", "cb_person_default_on_file",
                 "credit_history_length", "person_emp_length", "loan_percent_income"]

# Streamlit App
st.title("AI-Powered Credit Risk Scoring System")
st.write("Enter loan details to predict credit risk:")

# User Inputs
income = st.number_input("Applicant's Income", min_value=0)
age = st.number_input("Applicant's Age", min_value=18, max_value=100)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "PERSONAL", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amount = st.number_input("Loan Amount", min_value=100)
loan_interest_rate = st.number_input("Loan Interest Rate", min_value=0.0)
credit_default = st.selectbox("Credit Bureau Default", ["Y", "N"])
credit_history_length = st.number_input("Credit History Length (Years)", min_value=0)
emp_length = st.number_input("Employment Length (Years)", min_value=0)
loan_percent_income = loan_amount / (income + 1e-5)  # Avoid division by zero

# Convert categorical variables to numeric
home_ownership_dict = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
loan_intent_dict = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "HOMEIMPROVEMENT": 3, "PERSONAL": 4, "DEBTCONSOLIDATION": 5}
loan_grade_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
credit_default_dict = {"Y": 1, "N": 0}

# Create feature array
features = np.array([
    income, age, home_ownership_dict[home_ownership], loan_intent_dict[loan_intent],
    loan_grade_dict[loan_grade], loan_amount, loan_interest_rate, 
    credit_default_dict[credit_default], credit_history_length, emp_length, loan_percent_income
]).reshape(1, -1)

# Convert to DataFrame (to maintain feature names)
features_df = pd.DataFrame(features, columns=feature_names)

if st.button("Predict Credit Risk"):
    try:
        # Scale the input features
        features_scaled = scaler.transform(features_df)  # Ensure correct feature order
        prediction = model.predict(features_scaled)  # Predict risk category
        risk_category = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.success(f"Predicted Credit Risk: {risk_category}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

