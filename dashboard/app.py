import streamlit as st
import requests

API_URL = "http://localhost:30080/predict"

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="???",
    layout="wide"
)

st.title("??? Financial Fraud Detection System")
st.caption("Real-time transaction fraud prediction using XGBoost")

st.subheader("Transaction Details")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_id = st.number_input("Transaction ID", min_value=1, value=2987095)
        amount = st.number_input("Transaction Amount", min_value=0.0, value=150.0)
        card1 = st.number_input("Card ID", value=10000)
        addr1 = st.number_input("Address 1", value=100)
        addr2 = st.number_input("Address 2", value=87)

    with col2:
        tx_1h = st.number_input("Transactions in Last 1 Hour", min_value=0, value=2)
        tx_24h = st.number_input("Transactions in Last 24 Hours", min_value=0, value=5)
        amount_ratio = st.number_input("Amount vs Card Average", min_value=0.0, value=1.2)
        device_changed = st.selectbox("Device Changed", [0, 1])
        addr1_changed = st.selectbox("Address 1 Changed", [0, 1])
        addr2_changed = st.selectbox("Address 2 Changed", [0, 1])

    submitted = st.form_submit_button("Predict Transaction")

if submitted:
    payload = {
        "TransactionID": int(transaction_id),
        "TransactionAmt": amount,
        "card1": card1,
        "addr1": addr1,
        "addr2": addr2,
        "transactions_last_1h": tx_1h,
        "transactions_last_24h": tx_24h,
        "amount_vs_card_average": amount_ratio,
        "device_changed": device_changed,
        "addr1_changed": addr1_changed,
        "addr2_changed": addr2_changed
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Result")

        probability = result["fraud_probability"]
        st.metric("Fraud Probability", f"{probability * 100:.2f}%")
        st.progress(probability)

        if result["prediction"] == "Fraud":
            st.error("?? Fraudulent Transaction Detected")
        else:
            st.success("? Legitimate Transaction")

        st.json(result)

    except requests.exceptions.RequestException as error:
        st.error(f"API connection failed: {error}")
