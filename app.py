import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set Page Config
st.set_page_config(page_title="Loan Approval AI", layout="centered")

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("best_model.pkl", "rb"))
        col_info = pickle.load(open("model_columns.pkl", "rb"))
        return model, col_info
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None

model, col_info = load_assets()

def calculate_emi(principal, annual_rate, months):
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / months
    numerator = principal * r * ((1 + r) ** months)
    denominator = ((1 + r) ** months) - 1
    return numerator / denominator

st.title("🏦 Banking Loan Prediction System")
st.write("Predict if a loan application will be **Approved** or **Rejected**.")

if model is None:
    st.warning("Model is not loaded.")
    st.stop()

with st.form("loan_prediction_form"):
    st.subheader("Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", 
            ["Software Engineer", "Doctor", "Retail Manager", "High School Teacher", 
             "Construction Worker", "Freelance Designer", "University Student", "CEO", "Retired (Pension)"])
        income = st.number_input("Annual Income (INR)", min_value=0, value=500000, step=10000)
        score = st.slider("Credit Score", 300, 850, value=700) 
        dti = st.slider("DTI Ratio", 0.0, 1.0, value=0.3)
        
    with col2:
        lender = st.selectbox("Lender", ["HDFC Bank", "State Bank of India", "Bajaj Finserv", "Kotak Mahindra Bank", "ICICI Bank", "Axis Bank"])
        purpose = st.selectbox("Loan Purpose", 
            ["Debt Consolidation", "Medical Emergency", "Home Renovation", "Education (Tuition/Fees)", 
             "Travel/Vacation", "Wedding Expenses", "Vehicle Purchase", "Consumer Durable/Tech"])
        approved_loan_amount = st.number_input("Requested Loan Amount (INR)", min_value=0, value=100000, step=5000)
        interest = st.number_input("Interest Rate (%)", min_value=0.0, value=8.5, step=0.1)
        months = st.number_input("Loan Tenure (Months)", min_value=1, value=12)

    submitted = st.form_submit_button("Predict Status")

if submitted:
    
    emi = calculate_emi(approved_loan_amount, interest, months)
    
   
    input_data = {
        'Age': age,
        'Occupation': occupation,
        'Annual Income (INR)': income,
        'Credit Score (300-850)': score,
        'Lender': lender,
        'Loan Purpose': purpose,
        'Approved Loan Amount (INR)': approved_loan_amount,
        'Months of Payment': months,
        'Interest Rate (%)': interest,
        'Monthly EMI (INR)': emi,
        'DTI Ratio': dti
    }
    
    input_df = pd.DataFrame([input_data])
    
    try:
        # Predict
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)
        
        st.divider()
        st.subheader("Prediction Result")
        
        st.write(f"**Calculated Monthly EMI:** ₹{emi:,.2f}")
        
        if prediction[0] == 1:
            st.success(" Status: APPROVED (Pass)")
            st.write(f"Confidence: **{np.max(prob)*100:.2f}%**")
            st.balloons()
        else:
            st.error(" Status: REJECTED")
            st.write(f"Confidence: **{np.max(prob)*100:.2f}%**")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")