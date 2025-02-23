import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# Ensure feature names match model training
FEATURE_NAMES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length"
]

st.title("AI-Powered Credit Risk Scoring System")
st.write("Enter loan details to predict credit risk:")

# User Inputs
age = st.number_input("Applicant's Age", min_value=18)
income = st.number_input("Applicant's Income", min_value=0)
home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
employment_length = st.number_input("Employment Length (Years)", min_value=0)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amount = st.number_input("Loan Amount", min_value=0)
interest_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, format="%.2f")
credit_default = st.selectbox("Credit Bureau Default", ["Y", "N"])
credit_history_length = st.number_input("Credit History Length (Years)", min_value=0)

# Encode categorical variables (Match training format)
home_ownership_map = {"RENT": 0, "MORTGAGE": 1, "OWN": 2}
loan_intent_map = {"EDUCATION": 0, "MEDICAL": 1, "PERSONAL": 2, "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5}
loan_grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
credit_default_map = {"N": 0, "Y": 1}

# Create feature array with correct column names
input_features = np.array([
    age, income, home_ownership_map[home_ownership], employment_length,
    loan_intent_map[loan_intent], loan_grade_map[loan_grade], loan_amount,
    interest_rate, loan_amount/income, credit_default_map[credit_default], credit_history_length
]).reshape(1, -1)

# Convert to DataFrame with correct feature names
input_df = pd.DataFrame(input_features, columns=FEATURE_NAMES)

# Scale numerical features
input_df[FEATURE_NAMES] = scaler.transform(input_df[FEATURE_NAMES])

# Make prediction
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)
    risk_category = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.success(f"Predicted Credit Risk: {risk_category}")
