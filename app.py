import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Readmission Risk Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('xgb_model.pkl')
    encoders = joblib.load('encoders.pkl')
    explainer = shap.TreeExplainer(model)
    X_train = pd.read_csv('X_train.csv')
    return model, encoders, explainer, X_train

model, encoders, explainer, X_train = load_artifacts()

st.title("🏥 30-Day Hospital Readmission Risk Predictor")
st.markdown("""
This tool predicts whether a diabetic patient is at risk of being readmitted to the
hospital within **30 days** of discharge, using a machine learning model trained on
real (anonymized) records from **101,766 hospital encounters across 130 US hospitals**.

Built to help hospitals and compliance teams identify high-risk patients for early
intervention — reducing both readmission penalties and patient harm.
""")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Patient Age", 5, 95, 65)
    time_in_hospital = st.slider("Days in hospital (this stay)", 1, 14, 4)
    num_medications = st.slider("Number of medications prescribed", 1, 50, 15)
with col2:
    num_lab_procedures = st.slider("Number of lab procedures", 1, 120, 45)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
    number_inpatient = st.slider("Prior inpatient visits (past year)", 0, 15, 0)
with col3:
    number_outpatient = st.slider("Prior outpatient visits (past year)", 0, 15, 0)
    number_emergency = st.slider("Prior emergency visits (past year)", 0, 15, 0)
    diabetesMed = st.selectbox("On diabetes medication?", ["Yes", "No"])

if st.button("Predict Readmission Risk", type="primary"):
    # Build a feature row matching training data structure, using sensible defaults
    # for fields not exposed in this simplified demo UI
    row = X_train.iloc[0:1].copy()
    row['age_numeric'] = age
    row['time_in_hospital'] = time_in_hospital
    row['num_medications'] = num_medications
    row['num_lab_procedures'] = num_lab_procedures
    row['number_diagnoses'] = number_diagnoses
    row['number_inpatient'] = number_inpatient
    row['number_outpatient'] = number_outpatient
    row['number_emergency'] = number_emergency
    row['total_prior_visits'] = number_inpatient + number_outpatient + number_emergency
    row['diabetesMed'] = encoders['diabetesMed'].transform([diabetesMed])[0]

    risk_proba = model.predict_proba(row)[0, 1]

    st.divider()
    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        st.metric("Predicted Readmission Risk", f"{risk_proba*100:.1f}%")
        if risk_proba > 0.3:
            st.error("⚠️ HIGH RISK — recommend follow-up intervention")
        elif risk_proba > 0.15:
            st.warning("⚡ MODERATE RISK — consider monitoring")
        else:
            st.success("✅ LOW RISK")

    with rcol2:
        st.markdown("**Why did the model predict this? (SHAP explanation)**")
        shap_values = explainer.shap_values(row)
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                              data=row.iloc[0], feature_names=list(row.columns)),
            max_display=8, show=False
        )
        st.pyplot(fig)

st.divider()
st.markdown("**Global feature importance across all patients:**")
st.image("shap_summary.png", use_container_width=True)

st.caption("Built by Srija Reddy Annam | Model: XGBoost | Dataset: UCI Diabetes 130-US Hospitals (1999-2008)")
