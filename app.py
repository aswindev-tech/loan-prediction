import streamlit as st
import pickle
import pandas as pd
import numpy as np


@st.cache_resource
def load_assets():
    model = pickle.load(open("final_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, features

model, scaler, feature_cols = load_assets()


st.set_page_config(page_title="Loan Approval AI", layout="centered")
st.title("🏦 Banking Loan Prediction System")
st.write("Please enter the applicant details. All fields are currently reset to zero.")


with st.form("loan_prediction_form"):
    st.subheader("Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        age = st.number_input("Age", min_value=0, max_value=100, value=0)
        income = st.number_input("Annual Income (INR)", min_value=0, value=0)
        score = st.slider("Credit Score", 300, 850, value=300) 
        dti = st.slider("DTI Ratio", 0.0, 1.0, value=0.0)
        
    with col2:
        loan_amount = st.number_input("Requested Loan Amount (INR)", min_value=0, value=0)
        approved_loan_amount = st.number_input("Approved Loan Amount (INR)", min_value=0, value=0)
        interest = st.number_input("Interest Rate (%)", min_value=0.0, value=0.0)
        months = st.number_input("Months of Payment", min_value=0, value=0)

    st.write("---")
    
    
    occupation = st.text_input("Occupation", value="", placeholder="e.g. Software Engineer")
    purpose = st.text_input("Loan Purpose", value="", placeholder="e.g. Debt Consolidation")
    
    
    lender = st.selectbox("Lender", ["HDFC Bank", "State Bank of India", "Bajaj Finserv", "Kotak Mahindra Bank"])

    
    submitted = st.form_submit_button("Predict Status")


if submitted:
    
    data = {
        'Age': age,
        'Annual Income (INR)': income,
        'Credit Score (300-850)': score,
        'Approved Loan Amount (INR)': approved_loan_amount,
        'Interest Rate (%)': interest,
        'Months of Payment': months,
        'DTI Ratio': dti
    }
    
    input_df = pd.DataFrame([data])
    

    clean_occ = occupation.strip()
    clean_purpose = purpose.strip()
    
    input_df[f"Occupation_{clean_occ}"] = 1
    input_df[f"Lender_{lender}"] = 1
    input_df[f"Loan Purpose_{clean_purpose}"] = 1
    
    
    final_df = input_df.reindex(columns=feature_cols, fill_value=0)
    
    
    scaled_data = scaler.transform(final_df)
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)

    
    st.divider()
    st.subheader("Result")
    
    
    if prediction[0] == 1:
        st.success("Status: PASS / APPROVED")
        st.write(f"Confidence: **{np.max(prob)*100:.2f}%**")
        st.balloons()
    else:
        st.error(" Status: REJECTED")
        st.write(f"Confidence: **{np.max(prob)*100:.2f}%**")