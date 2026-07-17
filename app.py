import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Load model and encoder once at startup ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("churn_histgbm_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

numeric_cols = [
    "AGE", "TECH_COMFORT_SCORE", "DAYS_SINCE_LAST_ACTIVITY",
    "TOTAL_SESSIONS_2022", "GROSS_SESSION_LENGTH_2022", "ACTIVE_DAYS_2022",
    "ACTIVE_QUARTERS_2022", "AVG_SESSIONS_PER_ACTIVE_QUARTER", "AVG_SESSION_LENGTH_2022",
]
categorical_cols = ["INCOME_LEVEL", "EDUCATION", "DEVICE_TYPE"]
feature_order = numeric_cols + categorical_cols

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("Healthy Meals — Renewal / Churn Probability Predictor")
st.write(
    "Enter a customer's 2022 engagement metrics and demographics to predict the "
    "probability they renew their Healthy Meals subscription in 2023."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5)
    days_since_last_activity = st.number_input(
        "Days Since Last Activity (as of 2023-01-01)", min_value=0, value=30
    )
    total_sessions_2022 = st.number_input("Total Sessions (2022)", min_value=0, value=20)
    gross_session_length_2022 = st.number_input(
        "Gross Session Length, minutes (2022)", min_value=0.0, value=400.0
    )

with col2:
    active_days_2022 = st.number_input("Active Days (2022)", min_value=0, value=15)
    active_quarters_2022 = st.number_input("Active Quarters (2022)", min_value=0, max_value=4, value=3)
    avg_sessions_per_active_quarter = st.number_input(
        "Avg Sessions per Active Quarter", min_value=0.0, value=6.5
    )
    avg_session_length_2022 = st.number_input(
        "Avg Session Length per Session, minutes (2022)", min_value=0.0, value=20.0
    )

income_level = st.selectbox("Income Level", ["Low", "Medium", "High", "Very High"])
education = st.selectbox("Education", ["High School", "Graduate", "Post-Graduate", "Other"])
device_type = st.selectbox("Device Type", ["Desktop-only", "Mobile-only", "Multi-device"])

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("Predict"):

    row = pd.DataFrame([{
        "AGE": age,
        "TECH_COMFORT_SCORE": tech_comfort_score,
        "DAYS_SINCE_LAST_ACTIVITY": days_since_last_activity,
        "TOTAL_SESSIONS_2022": total_sessions_2022,
        "GROSS_SESSION_LENGTH_2022": gross_session_length_2022,
        "ACTIVE_DAYS_2022": active_days_2022,
        "ACTIVE_QUARTERS_2022": active_quarters_2022,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_sessions_per_active_quarter,
        "AVG_SESSION_LENGTH_2022": avg_session_length_2022,
        "INCOME_LEVEL": income_level,
        "EDUCATION": education,
        "DEVICE_TYPE": device_type,
    }])

    # Ordinal-encode the categoricals with the SAME fitted encoder used in training
    # (transform only — never fit_transform here)
    row[categorical_cols] = encoder.transform(row[categorical_cols])

    # Re-order columns to match the training feature matrix exactly
    row = row[feature_order]

    renewal_prob = model.predict_proba(row)[0][1]
    churn_prob = 1 - renewal_prob
    risk = "Low" if renewal_prob >= 0.6 else "Medium" if renewal_prob >= 0.4 else "High"

    st.metric("Renewal Probability", f"{renewal_prob:.1%}")
    st.metric("Churn Probability", f"{churn_prob:.1%}")

    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")