import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ABC Ltd. Customer Churn Predictor", layout="centered")

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("📉 ABC Ltd. — Customer Churn Predictor")
st.write("Enter a customer's details to predict their likelihood of churning.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

if st.button("Predict Churn Risk"):
    input_dict = {
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone, 'MultipleLines': multiple_lines,
        'InternetService': internet, 'OnlineSecurity': online_sec,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protect,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    # Align columns with training data (fill missing dummy cols with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — Probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Churn Risk — Probability: {probability:.1%}")

    st.progress(float(probability))
