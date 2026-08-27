# 30-Day Hospital Readmission Risk Prediction

Predicting which diabetic patients are at risk of hospital readmission within 30 days
of discharge — with an explainable, deployable machine learning model.

## Business Problem

Hospitals face significant financial penalties under CMS's Hospital Readmissions
Reduction Program when patients are readmitted within 30 days of discharge. Beyond
the financial cost, early readmission often signals a gap in discharge planning or
post-care follow-up that puts patient health at risk.

This project builds a model that flags high-risk patients **before** discharge, so
care teams and compliance staff can prioritize follow-up outreach where it matters
most — turning a reactive penalty problem into a proactive care opportunity.

## Dataset

- **Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999-2008)
- **Size:** 101,766 hospital encounters across 130 US hospitals and integrated delivery
  networks, 50 original clinical features
- After cleaning (removing deceased/hospice discharges and ID columns): **99,343 patient
  encounters** used for modeling

## Approach

1. **Data Cleaning** — removed non-predictive ID columns, excluded death/hospice
   discharges (cannot be readmitted), preserved meaningful "not tested" signals in lab
   result fields rather than discarding them.
2. **Feature Engineering** — converted age ranges to numeric midpoints, grouped 700+
   raw ICD diagnosis codes into 9 clinically meaningful categories (Circulatory,
   Diabetes, Respiratory, etc.), engineered a `total_prior_visits` feature combining
   outpatient/emergency/inpatient history as a proxy for care complexity.
3. **Class Imbalance Handling** — only 11.2% of encounters involve a 30-day
   readmission. Rather than ignoring this (which produces a model that just predicts
   "no" for everyone), the model uses weighted training (`scale_pos_weight`) to
   prioritize catching true at-risk cases, since a missed at-risk patient is
   costlier than a false alarm in a clinical context.
4. **Model** — Gradient-boosted trees (XGBoost), chosen for strong performance on
   structured clinical data and native support for explainability tooling.
5. **Explainability (SHAP)** — every prediction can be broken down into the specific
   factors that drove it, which is essential in healthcare and compliance settings
   where black-box decisions aren't acceptable.
6. **Interactive Demo** — a Streamlit app where a user can input patient
   characteristics and get a live risk score with a visual explanation of *why*.

## Results (on held-out test data, 19,869 patients never seen during training)

| Metric | Score | What it means |
|---|---|---|
| AUC-ROC | 0.672 | Model reliably separates higher-risk from lower-risk patients — consistent with published research on this same dataset |
| Recall (at-risk patients) | 0.575 | Catches 57.5% of true 30-day readmissions |
| Precision | 0.184 | Reflects a deliberate trade-off favoring catching more true cases over minimizing false alarms |

**Top predictive factors** (via SHAP): prior inpatient visit history, discharge
disposition, primary diagnosis category, total number of diagnoses, and patient age —
all clinically sensible drivers of readmission risk.

## Tech Stack

Python, pandas, XGBoost, scikit-learn, SHAP, Streamlit, Matplotlib

## Files

- `diabetic_data.csv` — raw source data
- `cleaned_data.csv`, `engineered_data.csv` — intermediate cleaned/engineered datasets
- `xgb_model.pkl` — trained model
- `app.py` — interactive Streamlit demo
- `shap_summary.png` — global feature importance visualization

## Running the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Srija Reddy Annam — MS Artificial Intelligence, Faulkner University
[LinkedIn](https://linkedin.com/in/srijareddyannam) | [GitHub](https://github.com/srijareddy-AI)
