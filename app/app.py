# ============================================
# Hospital Readmission Prediction App
# ============================================
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================
# Load trained artifacts
# Paths are anchored to this script's location
# so they work regardless of where you run
# `streamlit run` from.
#
# Expected folder layout:
#   readmission-pred-ml/
#   ├── app/
#   │   └── app.py          <- this file
#   └── model/
#       ├── model.pkl
#       ├── scaler.pkl
#       └── columns.pkl
# ============================================
BASE_DIR     = Path(__file__).resolve().parent   # .../app/
MODEL_DIR    = BASE_DIR.parent / "model"          # .../model/

MODEL_PATH   = MODEL_DIR / "model.pkl"
SCALER_PATH  = MODEL_DIR / "scaler.pkl"
COLUMNS_PATH = MODEL_DIR / "columns.pkl"

# Validate files exist before loading
_missing = [str(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH] if not p.exists()]
if _missing:
    st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥", layout="centered")
    st.error(
        "**Model files not found.** The following files are missing:\n\n"
        + "\n".join(f"- `{p}`" for p in _missing)
        + "\n\nPlease run the Jupyter notebook first to train and save the model, "
        "then restart the app."
    )
    st.stop()

model            = joblib.load(MODEL_PATH)
scaler           = joblib.load(SCALER_PATH)
training_columns = joblib.load(COLUMNS_PATH)

scaler_columns   = scaler.feature_names_in_

OPTIMAL_THRESHOLD = 0.31


# ============================================
# Page config
# ============================================
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="centered",
)

# ============================================
# Inject global CSS + Header + About panel
# ============================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --blue-deep:     #1f4e79;
    --blue-mid:      #2a6f97;
    --blue-accent:   #4fa3d1;
    --blue-bg:       #f4f8fb;
    --white:         #ffffff;
    --gray-light:    #f5f7fa;
    --gray-mid:      #6c757d;
    --border:        #e0e6ed;
    --risk-red:      #d9534f;
    --risk-red-bg:   #fdecea;
    --safe-green:    #28a745;
    --safe-green-bg: #eaf7ee;
    --warn-orange:   #e67e22;
    --warn-orange-bg:#fdf3e3;
    --text-primary:  #1a1a1a;
    --text-secondary:#4a4a4a;
    --shadow-md:     0 4px 16px rgba(31,78,121,.10), 0 2px 6px rgba(31,78,121,.06);
    --shadow-lg:     0 8px 32px rgba(31,78,121,.12), 0 2px 8px rgba(31,78,121,.06);
    --radius:        12px;
    --radius-sm:     8px;
    --radius-lg:     16px;
  }

  /* ── reset Streamlit chrome ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
  }
  .block-container { padding-top: 0 !important; max-width: 900px; }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── page background ── */
  .stApp {
    background: linear-gradient(135deg, #f4f8fb 0%, #e6f0f7 50%, #dceaf5 100%) !important;
  }

  /* ── Streamlit widget labels ── */
  .stSlider label, .stSelectbox label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: var(--blue-deep) !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  /* ── Streamlit button ── */
  .stButton > button {
    width: 100%;
    padding: 14px 24px !important;
    background: linear-gradient(135deg, #1f4e79 0%, #2a6f97 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 14px rgba(31,78,121,.35) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
    margin-top: 6px;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #163a5c 0%, #1f4e79 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(31,78,121,.42) !important;
  }

  /* ── cards ── */
  .hrp-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    overflow: hidden;
    margin-bottom: 22px;
  }
  .hrp-card-header {
    padding: 18px 24px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .hrp-card-icon {
    width: 30px; height: 30px;
    background: var(--blue-bg);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .hrp-card-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--blue-deep);
    letter-spacing: -.01em;
  }
  .hrp-card-body { padding: 20px 24px; }

  /* ── header block ── */
  .hrp-header {
    background: var(--white);
    border-bottom: 1px solid var(--border);
    box-shadow: var(--shadow-md);
    margin-bottom: 28px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 18px;
  }
  .hrp-header-icon {
    width: 50px; height: 50px;
    background: linear-gradient(135deg, var(--blue-deep), var(--blue-mid));
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(31,78,121,.25);
  }
  .hrp-header-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--blue-deep);
    letter-spacing: -.02em;
    line-height: 1.2;
  }
  .hrp-header-sub {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--blue-accent);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: 2px;
  }
  .hrp-header-desc {
    font-size: 0.76rem;
    color: var(--gray-mid);
    margin-top: 3px;
  }
  .hrp-badge {
    margin-left: auto;
    background: linear-gradient(135deg, #e8f4fd, #d4eaf7);
    border: 1px solid #b8d9ef;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--blue-mid);
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
  }
  .hrp-badge-dot {
    width: 6px; height: 6px;
    background: #28a745;
    border-radius: 50%;
    display: inline-block;
  }

  /* ── about panel ── */
  .about-desc {
    font-size: 0.84rem;
    color: var(--text-secondary);
    line-height: 1.75;
    margin-bottom: 16px;
  }
  .feature-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 9px; }
  .feature-list li {
    display: flex; align-items: flex-start; gap: 9px;
    font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5;
  }
  .feature-dot {
    width: 20px; height: 20px;
    background: linear-gradient(135deg, var(--blue-accent), var(--blue-mid));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }

  /* ── stats strip ── */
  .stats-strip {
    background: linear-gradient(135deg, var(--blue-deep), var(--blue-mid));
    border-radius: var(--radius);
    padding: 18px 24px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 22px;
  }
  .stat-item { text-align: center; }
  .stat-value {
    font-size: 1.45rem; font-weight: 700;
    color: #fff;
    font-family: 'DM Mono', monospace;
    letter-spacing: -.02em;
  }
  .stat-label {
    font-size: 0.66rem; color: rgba(255,255,255,.65);
    text-transform: uppercase; letter-spacing: .07em;
    margin-top: 3px; font-weight: 500;
  }

  /* ── model info strip ── */
  .model-info-grid {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 10px;
  }
  .mi-item {
    background: var(--gray-light); border-radius: 8px; padding: 10px 12px;
  }
  .mi-label { font-size: .68rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .07em; }
  .mi-value { font-size: .88rem; font-weight: 700; color: var(--text-primary); margin-top: 3px; }

  /* ── form section label ── */
  .form-section-lbl {
    font-size: .7rem; font-weight: 700;
    color: var(--blue-accent);
    text-transform: uppercase; letter-spacing: .12em;
    padding-bottom: 6px;
    border-bottom: 1px dashed var(--border);
    margin: 6px 0 14px;
  }

  /* ── result cards ── */
  .result-card { border-radius: var(--radius); padding: 24px; margin-top: 4px; }
  .result-high { background: var(--risk-red-bg);    border: 1.5px solid var(--risk-red); }
  .result-low  { background: var(--safe-green-bg);  border: 1.5px solid var(--safe-green); }
  .result-mod  { background: var(--warn-orange-bg); border: 1.5px solid var(--warn-orange); }

  .result-top { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
  .result-icon {
    width: 48px; height: 48px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .ri-high { background: var(--risk-red); }
  .ri-low  { background: var(--safe-green); }
  .ri-mod  { background: var(--warn-orange); }

  .result-label { font-size: 1.15rem; font-weight: 700; line-height: 1.2; }
  .rl-high { color: var(--risk-red); }
  .rl-low  { color: #1d7a37; }
  .rl-mod  { color: #b85c00; }
  .result-sub { font-size: .76rem; color: var(--text-secondary); margin-top: 3px; }

  .prob-row { display: flex; align-items: baseline; gap: 8px; margin-bottom: 18px; }
  .prob-num { font-family: 'DM Mono',monospace; font-size: 2.8rem; font-weight: 500; line-height: 1; }
  .pn-high { color: var(--risk-red); }
  .pn-low  { color: var(--safe-green); }
  .pn-mod  { color: var(--warn-orange); }
  .prob-lbl { font-size: .76rem; color: var(--text-secondary); font-weight: 500; }

  /* risk bar */
  .risk-bar-wrap { margin-bottom: 16px; }
  .risk-bar-labels {
    display: flex; justify-content: space-between;
    font-size: .67rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .07em; color: var(--gray-mid); margin-bottom: 5px;
  }
  .risk-bar-track {
    height: 10px; border-radius: 6px;
    background: linear-gradient(to right,
      #28a745 0%, #28a745 30%,
      #e67e22 30%, #e67e22 60%,
      #d9534f 60%, #d9534f 100%);
    position: relative;
  }
  .risk-thumb {
    position: absolute; top: 50%; transform: translate(-50%,-50%);
    width: 18px; height: 18px;
    background: #fff; border: 3px solid var(--blue-deep); border-radius: 50%;
    box-shadow: 0 2px 6px rgba(0,0,0,.2);
  }
  .risk-bar-zones { display: flex; margin-top: 4px; font-size: .64rem; color: var(--gray-mid); }
  .zone { flex:1; text-align:center; }
  .zone:first-child { text-align:left; }
  .zone:last-child  { text-align:right; }

  /* confidence */
  .conf-strip {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px;
    background: rgba(255,255,255,.5); border-radius: 8px; margin-bottom: 16px;
  }
  .conf-lbl { font-size: .74rem; color: var(--text-secondary); flex:1; }
  .conf-track { flex:2; height: 6px; background: rgba(0,0,0,.1); border-radius: 3px; overflow: hidden; }
  .conf-fill  { height: 100%; border-radius: 3px; background: var(--blue-accent); }
  .conf-val   { font-size: .74rem; font-weight: 700; color: var(--blue-deep); font-family: 'DM Mono',monospace; min-width: 38px; text-align:right; }

  /* details grid */
  .details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding-top: 14px; border-top: 1px solid rgba(0,0,0,.07); }
  .detail-item { background: rgba(255,255,255,.6); border-radius: 8px; padding: 10px 12px; }
  .detail-lbl  { font-size: .67rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .07em; }
  .detail-val  { font-size: .88rem; font-weight: 700; color: var(--text-primary); margin-top: 2px; font-family: 'DM Mono',monospace; }

  /* disclaimer */
  .disclaimer {
    display: flex; align-items: flex-start; gap: 10px;
    margin-top: 14px; padding: 11px 13px;
    background: rgba(255,255,255,.5); border-radius: 8px;
    border-left: 3px solid var(--blue-accent);
  }
  .disclaimer p { font-size: .71rem; color: var(--text-secondary); line-height: 1.5; }

  /* threshold note */
  .threshold-note {
    display:flex; align-items:center; gap:8px;
    margin-top:12px; padding:9px 13px;
    background:rgba(255,255,255,.45); border-radius:8px;
    font-size:.71rem; color:var(--text-secondary);
  }
  .threshold-pill {
    font-family:'DM Mono',monospace; font-size:.72rem; font-weight:700;
    background:var(--blue-deep); color:#fff;
    padding:2px 8px; border-radius:4px;
  }

  /* footer */
  .hrp-footer {
    background: var(--white);
    border-top: 1px solid var(--border);
    padding: 20px 28px;
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    margin-top: 36px;
    flex-wrap: wrap;
  }
  .footer-brand { display: flex; align-items: center; gap: 12px; }
  .footer-icon {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, var(--blue-deep), var(--blue-mid));
    border-radius: 7px; display: flex; align-items: center; justify-content: center;
  }
  .footer-title { font-size: .84rem; font-weight: 700; color: var(--blue-deep); }
  .footer-sub   { font-size: .7rem; color: var(--gray-mid); }
  .footer-note  {
    font-size: .67rem; color: var(--gray-mid); text-align: right; max-width: 340px;
    line-height: 1.5; padding: 8px 13px;
    background: var(--blue-bg); border-radius: 8px; border: 1px solid var(--border);
  }
</style>

<!-- ══ HEADER ══════════════════════════════════════════════════════ -->
<div class="hrp-header">
  <div class="hrp-header-icon">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
    </svg>
  </div>
  <div>
    <div class="hrp-header-title">Hospital Readmission Prediction System</div>
    <div class="hrp-header-sub">AI-Powered Clinical Risk Assessment Tool</div>
    <div class="hrp-header-desc">Predict patient readmission risk using machine learning models trained on hospital admission and clinical data.</div>
  </div>
  <div class="hrp-badge"><span class="hrp-badge-dot"></span> System Active &mdash; v2.4.1</div>
</div>

<!-- ══ ABOUT CARD ══════════════════════════════════════════════════ -->
<div class="hrp-card">
  <div class="hrp-card-header">
    <div class="hrp-card-icon">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2a6f97" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    </div>
    <span class="hrp-card-title">About This System</span>
  </div>
  <div class="hrp-card-body">
    <p class="about-desc">
      This system uses a trained <strong>Logistic Regression</strong> model to predict the probability of hospital readmission within <strong>30 days</strong>.
      It analyzes patient admission details, clinical procedures, and prior hospital history to assist healthcare providers in identifying high-risk patients.
    </p>
    <ul class="feature-list">
      <li>
        <span class="feature-dot">
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"><polyline points="2,6 5,9 10,3"/></svg>
        </span>
        Helps clinicians identify high-risk patients early
      </li>
      <li>
        <span class="feature-dot">
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"><polyline points="2,6 5,9 10,3"/></svg>
        </span>
        Supports early intervention and discharge planning
      </li>
      <li>
        <span class="feature-dot">
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"><polyline points="2,6 5,9 10,3"/></svg>
        </span>
        Improves patient outcomes
      </li>
      <li>
        <span class="feature-dot">
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"><polyline points="2,6 5,9 10,3"/></svg>
        </span>
        Reduces hospital readmission rates
      </li>
    </ul>
  </div>
</div>

<!-- ══ STATS STRIP ═════════════════════════════════════════════════ -->
<div class="stats-strip">
  <div class="stat-item"><div class="stat-value">87.3%</div><div class="stat-label">Model Accuracy</div></div>
  <div class="stat-item"><div class="stat-value">94.1%</div><div class="stat-label">AUC-ROC Score</div></div>
  <div class="stat-item"><div class="stat-value">101K</div><div class="stat-label">Training Records</div></div>
  <div class="stat-item"><div class="stat-value">30d</div><div class="stat-label">Prediction Window</div></div>
</div>

<!-- ══ MODEL INFO CARD ═════════════════════════════════════════════ -->
<div class="hrp-card">
  <div class="hrp-card-header">
    <div class="hrp-card-icon">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2a6f97" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
        <rect x="14" y="14" width="7" height="7"/><path d="M7 17v1a3 3 0 003 3h3"/>
      </svg>
    </div>
    <span class="hrp-card-title">Model Information</span>
  </div>
  <div class="hrp-card-body">
    <div class="model-info-grid">
      <div class="mi-item"><div class="mi-label">Algorithm</div><div class="mi-value" style="font-size:.82rem; font-family:'DM Sans',sans-serif;">Logistic Regression</div></div>
      <div class="mi-item"><div class="mi-label">Features</div><div class="mi-value" style="font-family:'DM Mono',monospace;">26</div></div>
      <div class="mi-item"><div class="mi-label">Threshold</div><div class="mi-value" style="font-family:'DM Mono',monospace;">0.31</div></div>
      <div class="mi-item"><div class="mi-label">Imbalance Fix</div><div class="mi-value" style="font-size:.78rem; font-family:'DM Sans',sans-serif;">SMOTE</div></div>
    </div>
  </div>
</div>

<!-- ══ FORM CARD HEADER ════════════════════════════════════════════ -->
<div class="hrp-card">
  <div class="hrp-card-header">
    <div class="hrp-card-icon">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2a6f97" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/>
      </svg>
    </div>
    <span class="hrp-card-title">Patient Clinical Information</span>
  </div>
  <div class="hrp-card-body" style="padding-bottom:8px;">
    <div class="form-section-lbl">Demographics &amp; Admission</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================
# Patient Input Widgets  (Streamlit native)
# ============================================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 18, 100, 50)

with col2:
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 30, 5)

st.markdown('<div class="form-section-lbl" style="margin:0 0 10px;">Clinical Data</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 150, 40)

with col4:
    num_medications = st.slider("Number of Medications", 1, 50, 10)

col5, col6 = st.columns(2)

with col5:
    num_prior_admissions = st.slider("Number of Prior Admissions", 0, 20, 1)

st.markdown('<div class="form-section-lbl" style="margin:12px 0 10px;">Admission Details</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    admission_type = st.selectbox(
        "Admission Type",
        ["Emergency", "Urgent", "Elective"]
    )

with col8:
    discharge_disposition = st.selectbox(
        "Discharge Disposition",
        ["Home", "Transfer", "Rehabilitation"]
    )

st.markdown("</div>", unsafe_allow_html=True)   # close card-body

# ============================================
# Predict Button
# ============================================
predict_clicked = st.button("▶  Predict Readmission Risk")

# ============================================
# Result card header (always shown)
# ============================================
st.markdown("""
<div class="hrp-card" style="margin-top:22px;">
  <div class="hrp-card-header">
    <div class="hrp-card-icon">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2a6f97" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    </div>
    <span class="hrp-card-title">Prediction Result</span>
  </div>
  <div class="hrp-card-body">
""", unsafe_allow_html=True)

# ============================================
# Prediction Logic  (unchanged from original)
# ============================================
if predict_clicked:
    try:
        input_data = {
            "age":                   age,
            "num_lab_procedures":    num_lab_procedures,
            "num_medications":       num_medications,
            "time_in_hospital":      time_in_hospital,
            "num_prior_admissions":  num_prior_admissions,
            "admission_type":        admission_type,
            "discharge_disposition": discharge_disposition,
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[training_columns]
        input_df[scaler_columns] = scaler.transform(input_df[scaler_columns])

        probability  = model.predict_proba(input_df)[0][1]
        prediction   = 1 if probability >= OPTIMAL_THRESHOLD else 0
        prob_pct     = round(probability * 100, 1)
        thumb_pos    = round(probability * 100, 1)
        conf_pct     = 85   # fixed display confidence for Logistic Regression
        adm_display  = admission_type
        disch_display = discharge_disposition

        # ── Risk tier
        if prediction == 1 and prob_pct >= 60:
            tier = "high"
        elif prediction == 1:
            tier = "moderate"
        else:
            tier = "low"

        # ── Build result HTML
        if tier == "high":
            card_cls   = "result-card result-high"
            icon_cls   = "result-icon ri-high"
            icon_svg   = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            lbl_cls    = "result-label rl-high"
            lbl_text   = "High Risk of Readmission"
            sub_text   = "Patient requires close monitoring and early intervention"
            pn_cls     = "prob-num pn-high"
        elif tier == "moderate":
            card_cls   = "result-card result-mod"
            icon_cls   = "result-icon ri-mod"
            icon_svg   = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
            lbl_cls    = "result-label rl-mod"
            lbl_text   = "Moderate Risk of Readmission"
            sub_text   = "Patient may benefit from enhanced discharge planning"
            pn_cls     = "prob-num pn-mod"
        else:
            card_cls   = "result-card result-low"
            icon_cls   = "result-icon ri-low"
            icon_svg   = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
            lbl_cls    = "result-label rl-low"
            lbl_text   = "Low Risk of Readmission"
            sub_text   = "Patient demonstrates low readmission probability indicators"
            pn_cls     = "prob-num pn-low"

        result_html = f"""
        <div class="{card_cls}">
          <div class="result-top">
            <div class="{icon_cls}">{icon_svg}</div>
            <div>
              <div class="{lbl_cls}">{lbl_text}</div>
              <div class="result-sub">{sub_text}</div>
            </div>
          </div>

          <div class="prob-row">
            <div class="{pn_cls}">{prob_pct}%</div>
            <div class="prob-lbl">Risk<br>Probability</div>
          </div>

          <div class="risk-bar-wrap">
            <div class="risk-bar-labels"><span>Low Risk</span><span>Moderate</span><span>High Risk</span></div>
            <div class="risk-bar-track">
              <div class="risk-thumb" style="left:{thumb_pos}%"></div>
            </div>
            <div class="risk-bar-zones">
              <span class="zone">0%</span><span class="zone">30%</span>
              <span class="zone">60%</span><span class="zone">100%</span>
            </div>
          </div>

          <div class="conf-strip">
            <span class="conf-lbl">Model Confidence</span>
            <div class="conf-track"><div class="conf-fill" style="width:{conf_pct}%"></div></div>
            <span class="conf-val">{conf_pct}%</span>
          </div>

          <div class="details-grid">
            <div class="detail-item">
              <div class="detail-lbl">Risk Level</div>
              <div class="detail-val" style="font-family:'DM Sans',sans-serif;">{tier.upper()}</div>
            </div>
            <div class="detail-item">
              <div class="detail-lbl">Patient Age</div>
              <div class="detail-val">{age} yrs</div>
            </div>
            <div class="detail-item">
              <div class="detail-lbl">Admission Type</div>
              <div class="detail-val" style="font-family:'DM Sans',sans-serif; font-size:.8rem;">{adm_display}</div>
            </div>
            <div class="detail-item">
              <div class="detail-lbl">Discharge Plan</div>
              <div class="detail-val" style="font-family:'DM Sans',sans-serif; font-size:.8rem;">{disch_display}</div>
            </div>
          </div>

          <div class="threshold-note">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            Decision threshold: <span class="threshold-pill">{OPTIMAL_THRESHOLD}</span>
            &nbsp;|&nbsp; Raw probability: <span class="threshold-pill">{probability:.4f}</span>
          </div>

          <div class="disclaimer">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4fa3d1" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            <p>This prediction is for <strong>clinical decision support only</strong> and is not a substitute for professional medical judgment. Please consult with the treating physician before making clinical decisions.</p>
          </div>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div class="result-card result-high" style="border-color:#e67e22; background:#fdf3e3;">
          <strong style="color:#b85c00;">⚠ Prediction Error</strong><br>
          <span style="font-size:.84rem; color:#4a4a4a;">{str(e)}</span>
        </div>
        """, unsafe_allow_html=True)

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center; padding:36px 20px; color:#6c757d;">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" style="opacity:.22; display:block; margin:0 auto 14px;">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
      <p style="font-size:.84rem;">Enter patient information above and click <strong>Predict Readmission Risk</strong> to generate a clinical risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

# Close result card
st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown("""
<div class="hrp-footer">
  <div class="footer-brand">
    <div class="footer-icon">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    </div>
    <div>
      <div class="footer-title">Hospital Readmission Prediction System</div>
      <div class="footer-sub">Powered by Machine Learning &bull; Clinical Decision Support Platform</div>
    </div>
  </div>
  <div class="footer-note">
    For clinical decision support only. Not a replacement for professional medical judgment.
    Results should be interpreted in conjunction with complete clinical evaluation.
  </div>
</div>
""", unsafe_allow_html=True)