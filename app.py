import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_resource
def load_assets():
    model = pickle.load(open("final_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, features

model, scaler, feature_cols = load_assets()

@st.cache_data
def load_data():
    return pd.read_csv('loan.csv')

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

st.set_page_config(page_title="Loan Approval AI", layout="centered")

# Sidebar for Navigation
page = st.sidebar.radio("Navigation", ["Loan Prediction", "Data Analysis"])

if page == "Loan Prediction":
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

elif page == "Data Analysis":
    st.title("📊 Data Analysis & Insights")
    st.write("Explore the dataset used to train the model. Note: Outliers are removed for clearer visualization.")
    
    df = load_data()
    
    # Preprocessing (Outlier Removal)
    df_clean = remove_outliers(df, 'Annual Income (INR)')
    df_clean = remove_outliers(df_clean, 'Approved Loan Amount (INR)')
    
    st.subheader("Loan Status Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_clean, x='Loan Status (Pass/Rejected)', palette='viridis', ax=ax1)
    ax1.set_xlabel('Loan Status')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)
    
    st.divider()
    
    st.subheader("Approved Loan Amount Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_clean, x='Approved Loan Amount (INR)', bins=30, kde=True, color='skyblue', ax=ax2)
    ax2.set_xlabel('Approved Loan Amount (INR)')
    ax2.set_ylabel('Frequency')
    st.pyplot(fig2)
    
    st.divider()
    
    st.subheader("Correlation Heatmap (Numerical Features)")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df_clean[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax3)
    st.pyplot(fig3)