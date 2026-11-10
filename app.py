import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "loan_approval_xgboost.pkl"

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.title("Loan Approval Prediction System")
st.write("Enter applicant details to get a machine-learning prediction.")

bundle = load_model()

if bundle is None:
    st.warning("The trained model was not found.")
    st.code("python training/train_model.py", language="bash")
    st.info("First place loan_data_new.csv inside the dataset folder, then run the training command.")
    st.stop()

with st.form("loan_application"):
    st.subheader("Applicant information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        income = st.number_input("Person Income", min_value=1.0, value=50000.0, step=1000.0)
        experience = st.number_input("Employee Experience (years)", min_value=0.0, value=2.0)
        home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    with col2:
        loan_amount = st.number_input("Loan Amount", min_value=1.0, value=10000.0, step=500.0)
        intent = st.selectbox(
            "Loan Intent",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        loan_percentage = st.number_input("Loan Percentage", min_value=0.0, value=20.0)
        credit_history = st.number_input("Credit History", min_value=0.0, value=5.0)
        credit_score = st.number_input("Credit Score", min_value=0.0, value=700.0)
        previous_loan = st.selectbox("Previous Loan", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Predict Loan Status", use_container_width=True)

if submitted:
    applicant = {
        "Age": age,
        "Gender": gender,
        "Education": education,
        "Person Income": income,
        "Employee Experience": experience,
        "Home Onwership": home,
        "Loan Amount": loan_amount,
        "Loan Intent": intent,
        "Loan percentage": loan_percentage,
        "Credit History": credit_history,
        "Credit Score": credit_score,
        "Previous Loan": previous_loan
    }

    data = pd.DataFrame([applicant])
    data["Loan_to_Income_Ratio"] = (
        data["Loan Amount"] / data["Person Income"].replace(0, np.nan)
    )
    data = data[bundle["feature_columns"]]

    pipeline = bundle["pipeline"]
    prediction = int(pipeline.predict(data)[0])
    probability = float(pipeline.predict_proba(data)[0][1]) * 100

    st.divider()
    if prediction == 1:
        st.success("Prediction: LOAN APPROVED")
    else:
        st.error("Prediction: LOAN REJECTED")

    st.metric("Estimated approval probability", f"{probability:.2f}%")
    st.caption("This is an educational model prediction, not an actual bank lending decision.")

st.divider()
st.caption("Loan Approval Prediction System | Built with Python, XGBoost and Streamlit")
