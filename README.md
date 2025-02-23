# AI-Powered Credit Risk Scoring System

## 📌 Project Overview
This project is an AI-powered credit risk scoring system that predicts whether a loan applicant is high-risk or low-risk. The model is trained on financial data using **Logistic Regression** and **XGBoost**, with explainability provided by **SHAP** and **LIME**. The system also includes a **Streamlit-based web app** for user interaction and risk assessment.

## 📂 Project Structure
```
├── AI-Powered Credit Risk Scoring System.ipynb  # Jupyter Notebook for data processing & model training
├── app.py                                      # Streamlit web application
├── credit_risk_dataset.csv                     # Credit risk dataset
├── credit_risk_model.pkl                       # Trained machine learning model
├── label_encoders.pkl                          # Encoded categorical values
├── scaler.pkl                                  # Scaler for feature normalization
├── main.py                                     # Alternative script for running models
├── README.md                                   # Project documentation
```

## 🚀 How to Run the Project
### **1. Install Dependencies**
Before running the project, install the required Python packages:
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is not available, install manually:)*
```bash
pip install pandas numpy scikit-learn xgboost shap lime streamlit joblib
```

### **2. Run the Streamlit App**
Execute the following command in the project directory:
```bash
streamlit run app.py
```
This will launch the **credit risk prediction app** in your browser.

### **3. Interact with the App**
- Enter applicant details such as **income, age, home ownership, loan intent, and credit history**.
- Click **"Predict Credit Risk"** to get an AI-driven risk assessment.

## 🛠 Model Training and Explainability
- The models are trained on **Logistic Regression** and **XGBoost**.
- Feature importance is analyzed using **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations).

## 🎯 Features
✅ **Machine Learning Models** - Logistic Regression & XGBoost for risk prediction.  
✅ **Explainability** - SHAP & LIME to explain model decisions.  
✅ **Data Preprocessing** - Handles missing values and categorical variables.  
✅ **Interactive Dashboard** - Built using **Streamlit** for user-friendly risk assessment.  
✅ **Scalability** - The trained model can be extended with new data.  

## 📌 Future Enhancements
- Add **real-time API integration** for live financial data.
- Improve feature engineering for better prediction accuracy.
- Expand explainability using **counterfactual explanations**.

---
**🔗 Developed by:** *Hareesh Samudrala*  
*For AI-powered fintech solutions 🚀*

