import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import base64
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* fallback bg — overridden below by SVG injection */
    .stApp {
        background: #eef2ff;
    }

    /* ── Responsive: stack all columns on mobile ── */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        div[data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .hero-banner {
            padding: 1.6rem 1.2rem !important;
            border-radius: 14px !important;
            margin-bottom: 0.8rem !important;
            min-height: unset !important;
        }
        .hero-banner h1 { font-size: clamp(1.4rem, 5vw, 2rem) !important; }
        .hero-banner p  { font-size: clamp(0.95rem, 3vw, 1.1rem) !important; }
        .about-card {
            min-height: unset !important;
            padding: 1.2rem 1.4rem !important;
        }
        div[data-testid="stForm"] {
            padding: 1.2rem 1rem !important;
            border-radius: 14px !important;
        }
        .result-card-churn, .result-card-safe {
            padding: 1.5rem 1rem !important;
        }
        .result-card-churn h2, .result-card-safe h2 { font-size: 1.5rem !important; }
        .result-card-churn p,  .result-card-safe p  { font-size: 1rem !important; }
        .metric-box .val { font-size: 1.3rem !important; }
        .metric-box .lbl { font-size: 0.85rem !important; }
    }

    @media (max-width: 480px) {
        .about-avatar { width: 54px !important; height: 54px !important; font-size: 1.6rem !important; }
        .about-info h3 { font-size: 1.1rem !important; }
        .about-info .role { font-size: 0.92rem !important; }
        .about-link { font-size: 0.85rem !important; padding: 0.28rem 0.65rem !important; }
        .section-title { font-size: 0.92rem !important; }
        .prob-badge { font-size: 1.4rem !important; }
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 20px;
        padding: 2.8rem 2.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(79, 70, 229, 0.25);
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }
    .hero-banner h1 {
        color: white;
        font-size: clamp(1.6rem, 3vw, 2.6rem);
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.92);
        font-size: clamp(1rem, 1.8vw, 1.25rem);
        margin: 0;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    /* ── Section titles ── */
    .section-title {
        color: #4f46e5;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.4rem 0;
    }

    /* ── Form container ── */
    div[data-testid="stForm"] {
        background: white;
        border-radius: 20px;
        padding: 2rem 2.5rem !important;
        box-shadow: 0 4px 24px rgba(79, 70, 229, 0.08);
        border: 1px solid #e8eaf6;
    }

    /* ── Labels ── */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* ── Selectbox ── */
    div[data-testid="stSelectbox"] > div > div {
        background: #f8faff !important;
        border: 1.5px solid #c7d2fe !important;
        border-radius: 10px !important;
        color: #1e1b4b !important;
        font-size: 1.05rem !important;
    }

    /* ── Number inputs ── */
    div[data-testid="stNumberInput"] input {
        background: #f8faff !important;
        border: 1.5px solid #c7d2fe !important;
        border-radius: 10px !important;
        color: #1e1b4b !important;
        font-weight: 500;
        font-size: 1.05rem !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
    }

    /* ── Slider track ── */
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    }

    /* ── Submit button ── */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        padding: 0.85rem 0 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(79,70,229,0.35) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
        margin-top: 0.8rem;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(79,70,229,0.45) !important;
    }

    /* ── Result cards ── */
    .result-card-churn {
        background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(239,68,68,0.3);
    }
    .result-card-safe {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(5,150,105,0.3);
    }
    .result-card-churn h2, .result-card-safe h2 {
        color: white;
        font-size: clamp(1.4rem, 2.5vw, 2.1rem);
        font-weight: 700;
        margin: 0.4rem 0 0.25rem 0;
    }
    .result-card-churn p, .result-card-safe p {
        color: rgba(255,255,255,0.92);
        font-size: 1.1rem;
        margin: 0;
    }
    .prob-badge {
        display: inline-block;
        background: rgba(255,255,255,0.28);
        border-radius: 50px;
        padding: 0.4rem 1.4rem;
        font-size: 1.75rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.6rem;
    }

    /* ── Metric boxes ── */
    .metric-box {
        background: white;
        border: 1.5px solid #e0e7ff;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(79,70,229,0.07);
        margin-bottom: 0.5rem;
    }
    .metric-box .val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-box .lbl {
        font-size: 0.92rem;
        color: #6b7280;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* ── About card ── */
    .about-card {
        background: white;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        box-shadow: 0 4px 24px rgba(79,70,229,0.08);
        border: 1px solid #e8eaf6;
        display: flex;
        align-items: center;
        gap: 1.4rem;
        min-height: 170px;
        box-sizing: border-box;
    }
    .about-avatar {
        width: 68px;
        height: 68px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.9rem;
        flex-shrink: 0;
        box-shadow: 0 4px 16px rgba(79,70,229,0.3);
    }
    .about-info h3 {
        margin: 0 0 0.2rem 0;
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e1b4b;
    }
    .about-info .role {
        color: #4f46e5;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.65rem;
    }
    .about-links {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .about-link {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #f0f4ff;
        border: 1.5px solid #c7d2fe;
        border-radius: 8px;
        padding: 0.35rem 0.9rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #4f46e5;
        text-decoration: none;
        transition: all 0.2s;
    }
    .about-link:hover {
        background: #4f46e5;
        color: white;
        border-color: #4f46e5;
    }
    .about-badge {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        border-radius: 6px;
        padding: 0.2rem 0.65rem;
        font-size: 0.88rem;
        font-weight: 700;
        margin-left: 0.4rem;
        vertical-align: middle;
    }

    /* ── Bottom padding so content isn't clipped ── */
    .main .block-container {
        padding-bottom: 3rem !important;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── ANN background: tiled SVG of neural-network nodes + connections ────────────
@st.cache_data
def _build_ann_bg() -> str:
    inp = [(40, 80),  (40, 150), (40, 220)]
    hid = [(150, 50), (150, 115), (150, 180), (150, 245)]
    out = [(260, 100), (260, 200)]
    el = []
    for x1, y1 in inp:
        for x2, y2 in hid:
            el.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
                      f"stroke='rgba(79,70,229,0.09)' stroke-width='1'/>")
    for x1, y1 in hid:
        for x2, y2 in out:
            el.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
                      f"stroke='rgba(124,58,237,0.09)' stroke-width='1'/>")
    for x, y in inp:
        el.append(f"<circle cx='{x}' cy='{y}' r='6' fill='rgba(79,70,229,0.22)' "
                  f"stroke='rgba(79,70,229,0.35)' stroke-width='1.5'/>")
    for x, y in hid:
        el.append(f"<circle cx='{x}' cy='{y}' r='6' fill='rgba(124,58,237,0.22)' "
                  f"stroke='rgba(124,58,237,0.35)' stroke-width='1.5'/>")
    for x, y in out:
        el.append(f"<circle cx='{x}' cy='{y}' r='7' fill='rgba(79,70,229,0.28)' "
                  f"stroke='rgba(79,70,229,0.42)' stroke-width='1.5'/>")
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'>"
           + "".join(el) + "</svg>")
    return base64.b64encode(svg.encode()).decode()


_ann_bg = _build_ann_bg()
st.markdown(
    f'<style>.stApp{{background-color:#eef2ff!important;'
    f'background-image:url("data:image/svg+xml;base64,{_ann_bg}")!important;'
    f'background-repeat:repeat!important;background-size:300px 300px!important;}}</style>',
    unsafe_allow_html=True
)


# ── Load model & preprocessors ─────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model('model.h5', compile=False)
    with open('label_encoder_gender.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    return m, le, ohe, sc

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()


# ── Hero banner + About Me ─────────────────────────────────────────────────────
col_hero, col_about = st.columns([3, 2], gap="large")

with col_hero:
    st.markdown("""
    <div class="hero-banner">
        <h1>🔮 Customer Churn Predictor</h1>
        <p>Deep Learning ANN Churn Classification</p>
    </div>
    """, unsafe_allow_html=True)

with col_about:
    st.markdown("""
    <div class="about-card">
        <div class="about-avatar">👨‍💻</div>
        <div class="about-info">
            <h3>Lohith M <span class="about-badge">3+ YOE</span></h3>
            <div class="role">🤖 AI / ML Engineer</div>
            <div class="about-links">
                <a class="about-link" href="https://www.linkedin.com/in/lohithmothu" target="_blank">🔗 LinkedIn</a>
                <a class="about-link" href="https://github.com/krishna-lohith" target="_blank">🐙 GitHub</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    with col2:
        gender = st.selectbox('⚧ Gender', label_encoder_gender.classes_)
    with col3:
        age = st.slider('🎂 Age', 18, 100, value=35, step=1)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">💳 Financial Details</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        credit_score = st.number_input('📊 Credit Score', min_value=350, max_value=1000, value=650, step=50)
    with col5:
        balance = st.number_input('💰 Balance ($)', min_value=0.0, value=50000.0, step=5000.0)
    with col6:
        estimated_salary = st.number_input('💵 Estimated Salary ($)', min_value=0.0, value=60000.0, step=1000.0)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏦 Account Details</div>', unsafe_allow_html=True)
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        tenure = st.slider('📅 Tenure (yrs)', 0, 10, value=5)
    with col8:
        num_of_products = st.slider('📦 Products', 1, 4, value=2)
    with col9:
        has_cr_card = st.selectbox('💳 Credit Card', ['Yes', 'No'])
    with col10:
        is_active_member = st.selectbox('✅ Active Member', ['Yes', 'No'])

    submitted = st.form_submit_button("🔮  Predict Churn", use_container_width=True)


# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    try:
        input_data = pd.DataFrame({
            'CreditScore':       [credit_score],
            'Gender':            [label_encoder_gender.transform([gender])[0]],
            'Age':               [age],
            'Tenure':            [tenure],
            'Balance':           [balance],
            'NumOfProducts':     [num_of_products],
            'HasCrCard':         [1 if has_cr_card == 'Yes' else 0],
            'IsActiveMember':    [1 if is_active_member == 'Yes' else 0],
            'EstimatedSalary':   [estimated_salary],
        })

        geo_raw = onehot_encoder_geo.transform([[geography]])
        geo_array = geo_raw.toarray() if hasattr(geo_raw, 'toarray') else np.array(geo_raw)
        geo_df = pd.DataFrame(
            geo_array,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )

        input_df = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)
        prediction_proba = float(
            model.predict(scaler.transform(input_df), verbose=0)[0][0]
        )
        risk_pct      = prediction_proba * 100
        retention_pct = (1 - prediction_proba) * 100

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction_proba > 0.5:
            st.markdown(f"""
            <div class="result-card-churn">
                <div style="font-size:3rem">🚨</div>
                <div class="prob-badge">{risk_pct:.1f}% Risk</div>
                <h2>High Churn Risk Detected</h2>
                <p>This customer is <strong>likely to leave</strong>. Consider a targeted retention strategy.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-safe">
                <div style="font-size:3rem">✅</div>
                <div class="prob-badge">{retention_pct:.1f}% Retention</div>
                <h2>Customer Likely to Stay</h2>
                <p>This customer is <strong>not likely to churn</strong>. Keep up the great service!</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{risk_pct:.1f}%</div>
                <div class="lbl">Churn Probability</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{retention_pct:.1f}%</div>
                <div class="lbl">Retention Probability</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            risk_label = "🔴 High" if prediction_proba > 0.7 else (
                         "🟡 Medium" if prediction_proba > 0.4 else "🟢 Low")
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{risk_label}</div>
                <div class="lbl">Risk Level</div>
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}. Please check that all model files are present.")
