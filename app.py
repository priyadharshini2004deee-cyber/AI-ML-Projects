import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bank Term Deposit Subscription Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 1500px;
}
.main-header {
    background: linear-gradient(135deg, #ffffff, #eff6ff);
    border: 1px solid #dbeafe;
    border-radius: 24px;
    padding: 22px 24px 18px 24px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    margin-bottom: 16px;
}
.main-title {
    font-size: 2.45rem;
    font-weight: 900;
    color: #0f172a;
    text-align: center;
    line-height: 1.25;
    margin: 0;
    white-space: normal !important;
    word-break: break-word;
}
.main-subtitle {
    text-align: center;
    color: #475569;
    font-size: 1rem;
    margin-top: 10px;
}
.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0f172a;
    margin: 8px 0 12px 0;
}
.sub-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    margin-bottom: 10px;
}
.kpi-card {
    background: linear-gradient(135deg, #ffffff, #f8fbff);
    border: 1px solid #dbeafe;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}
.kpi-label {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 6px;
}
.kpi-value {
    color: #0f172a;
    font-size: 2rem;
    font-weight: 900;
}
.kpi-note {
    color: #2563eb;
    font-size: 0.83rem;
    margin-top: 6px;
}
.small-note {
    color: #64748b;
    font-size: 0.9rem;
}
.assistant-box {
    background: linear-gradient(135deg, #f8fafc, #eff6ff);
    border: 1px solid #bfdbfe;
    border-radius: 18px;
    padding: 18px;
    margin-top: 10px;
}
.voice-box {
    background: linear-gradient(135deg, #fefce8, #fff7ed);
    border: 1px solid #fde68a;
    border-radius: 16px;
    padding: 14px;
    margin-top: 12px;
}
div[data-baseweb="select"] > div {
    border-radius: 12px !important;
}
.stNumberInput input, .stTextInput input {
    border-radius: 12px !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    height: 44px;
    padding: 0 14px;
}
.stTabs [aria-selected="true"] {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    border-color: #bfdbfe !important;
}
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "bank-full.csv")
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

# =========================
# HELPERS
# =========================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("bank-full.csv not found in current folder.")
    return pd.read_csv(DATA_PATH, sep=";")

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "y" in df.columns:
        df["y_clean"] = df["y"].astype(str).str.lower().str.strip()
        df["y_num"] = df["y_clean"].map({"yes": 1, "no": 0})
        df["y_label"] = df["y_clean"].map({"yes": "Subscribed", "no": "Not Subscribed"})

    if {"balance", "age"}.issubset(df.columns):
        df["balance_age"] = df["balance"] * df["age"]

    if {"campaign", "balance"}.issubset(df.columns):
        df["campaign_balance"] = df["campaign"] * df["balance"]

    if "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"],
            bins=[17, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        )
    return df

def safe_mode(series, fallback):
    try:
        m = series.dropna().mode()
        return m.iloc[0] if len(m) else fallback
    except Exception:
        return fallback

def call_api(payload: dict, api_url: str):
    if not REQUESTS_AVAILABLE:
        return None, "requests package not installed. Run: pip install requests"
    try:
        response = requests.post(api_url, json=payload, timeout=20)
        if response.status_code != 200:
            return None, f"API Error {response.status_code}: {response.text}"
        return response.json(), None
    except Exception as e:
        return None, str(e)

def lead_segment_from_prob(prob: float) -> str:
    if prob >= 0.60:
        return "High Potential"
    if prob >= 0.30:
        return "Medium Potential"
    return "Low Potential"

def build_assistant_response(question: str, payload: dict, prob: float, prediction: str, segment: str):
    q = question.lower().strip()

    age = payload.get("age", 0)
    balance = payload.get("balance", 0)
    campaign = payload.get("campaign", 0)
    previous = payload.get("previous", 0)
    poutcome = payload.get("poutcome", "unknown")
    housing = payload.get("housing", "no")
    loan = payload.get("loan", "no")
    education = payload.get("education", "")
    job = payload.get("job", "")
    marital = payload.get("marital", "")
    month = payload.get("month", "")
    contact = payload.get("contact", "")

    positives = []
    risks = []

    if balance >= 3000:
        positives.append("strong account balance")
    else:
        risks.append("lower balance profile")

    if campaign <= 2:
        positives.append("healthy campaign contact count")
    else:
        risks.append("campaign contact count is relatively high")

    if previous > 0 and poutcome == "success":
        positives.append("successful previous campaign history")
    elif previous == 0:
        risks.append("no previous campaign history")

    if housing == "yes":
        risks.append("housing loan commitment")
    if loan == "yes":
        risks.append("personal loan burden")

    if education == "tertiary":
        positives.append("higher education profile")
    if contact == "cellular":
        positives.append("mobile contact channel")
    if marital == "single":
        positives.append("single customer segment")

    if "why" in q and ("low" in q or "subscribe" in q or "potential" in q):
        return (
            f"The model predicts **{prediction}** with **{prob*100:.2f}%** probability because "
            f"the profile currently looks **{segment.lower()}**. "
            f"Positive signals: {', '.join(positives) if positives else 'limited strong signals'}. "
            f"Risk factors: {', '.join(risks) if risks else 'few major risk signals'}."
        )

    if "improve" in q or "increase" in q or "how" in q:
        tips = []
        if campaign > 2:
            tips.append("reduce campaign contacts closer to 1 or 2")
        if balance < 3000:
            tips.append("target higher-balance or premium-fit customers")
        if previous == 0:
            tips.append("build trust with an introductory message before aggressive selling")
        if poutcome != "success":
            tips.append("re-target leads with better follow-up timing and tailored messaging")
        if loan == "yes" or housing == "yes":
            tips.append("position the deposit as safe and stable rather than promotional")
        if month not in ["mar", "apr", "sep", "oct", "dec"]:
            tips.append("test stronger-contact months or better outreach timing")
        return "To improve subscription chance, try this: " + "; ".join(tips if tips else ["maintain strong current profile and personalize the offer"])

    if "summary" in q or "profile" in q:
        return (
            f"This customer is a **{age}-year-old {job}** with **{education} education**, "
            f"**{marital}** marital status, and **balance {balance:,.0f}**. "
            f"Current model result: **{prediction}** with **{prob*100:.2f}%** probability, "
            f"segment = **{segment}**."
        )

    if "sales pitch" in q or "pitch" in q:
        return (
            "Suggested sales pitch: "
            "“Based on your profile, we can offer a stable term deposit option that helps protect your money while giving structured returns. "
            "This product is suitable for customers who value safety, planning, and predictable growth.”"
        )

    if "factor" in q:
        return (
            f"Key factors affecting this prediction include balance, campaign contact count, previous campaign history, "
            f"loan commitments, and customer profile signals. Current probability is **{prob*100:.2f}%**."
        )

    if "target" in q or "worth" in q:
        return (
            f"This customer is **{segment}**. Based on the current probability of **{prob*100:.2f}%**, "
            f"{'the lead is worth targeting with priority follow-up.' if prob >= 0.50 else 'the lead should be handled with a softer or lower-cost approach.'}"
        )

    if "strategy" in q:
        return (
            "Recommended strategy: use personalized communication, keep campaign pressure low, "
            "highlight safety and returns, and prioritize timing based on prior engagement."
        )

    if "what if" in q:
        return (
            "What-if insight: increasing balance, reducing campaign pressure, and improving previous outcome signals "
            "can materially improve conversion probability. Use the What-If Lab tab to compare scenarios."
        )

    return (
        f"The customer is currently classified as **{prediction}** with **{prob*100:.2f}%** probability. "
        f"Ask things like: **why is this lead low potential?**, **how to improve subscription chance?**, "
        f"**summarize this customer profile**, **give a sales pitch**, **what factors affect prediction?**, "
        f"**is this customer worth targeting?**, or **what strategy should bank use?**"
    )

def get_recommendations(prob, balance, campaign, previous, poutcome, housing, loan):
    recs = []

    if prob >= 0.60:
        recs.append("High-potential lead. Immediate personalized follow-up is recommended.")
    elif prob >= 0.30:
        recs.append("Medium-potential lead. Use targeted communication and optimized timing.")
    else:
        recs.append("Low-potential lead. Avoid repeated outreach and use a softer trust-building approach.")

    if balance >= 3000:
        recs.append("Balance profile is strong. Premium term deposit messaging may work well.")
    else:
        recs.append("Balance is modest. Emphasize safety, benefits, and gradual trust-building.")

    if campaign > 3:
        recs.append("Campaign contact frequency is high. Reduce pressure and improve call quality.")
    else:
        recs.append("Campaign contact count is acceptable.")

    if previous > 0 and poutcome == "success":
        recs.append("Previous campaign success found. This lead should be prioritized.")
    elif previous == 0:
        recs.append("No previous campaign history. Begin with educational communication.")

    if housing == "yes" or loan == "yes":
        recs.append("Existing loan commitments may affect decision. Use value-focused messaging.")

    return recs

def explain_prediction(prob, payload):
    reasons = []

    if payload["balance"] >= 3000:
        reasons.append("Higher balance is a positive sign.")
    else:
        reasons.append("Low or moderate balance may reduce subscription intent.")

    if payload["campaign"] <= 2:
        reasons.append("Low campaign contact count avoids over-contact pressure.")
    else:
        reasons.append("High campaign count can reduce response quality.")

    if payload["previous"] > 0 and payload["poutcome"] == "success":
        reasons.append("Successful previous campaign outcome strongly helps conversion.")
    elif payload["previous"] == 0:
        reasons.append("No previous campaign history gives limited trust signals.")

    if payload["loan"] == "yes" or payload["housing"] == "yes":
        reasons.append("Loan commitments can reduce deposit willingness.")

    if payload["contact"] == "cellular":
        reasons.append("Cellular contact often supports better reachability.")

    if prob >= 0.60:
        heading = "Why this looks strong"
    elif prob >= 0.30:
        heading = "Why this looks moderate"
    else:
        heading = "Why this looks weak"

    return heading, reasons

def build_voice_text(prob_percent, prediction, segment):
    return (
        f"This customer has a subscription probability of {prob_percent:.1f} percent. "
        f"The predicted class is {prediction}. "
        f"The lead segment is {segment}."
    )

def render_autoplay_audio(text: str):
    safe_text = json.dumps(text)
    html = f"""
    <script>
    const text = {safe_text};
    const msg = new SpeechSynthesisUtterance(text);
    msg.rate = 1;
    msg.pitch = 1;
    msg.volume = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(html, height=0)

# =========================
# LOAD DATA
# =========================
try:
    raw_df = load_data()
    df = prepare_data(raw_df)
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-header">
    <div class="main-title">🏦 Bank Term Deposit Subscription Prediction</div>
    <div class="main-subtitle">
        Top-tier dashboard with analytics, live API prediction, mini AI-style assistant,
        voice result output, what-if simulation, and premium business insights
        <br><br><b>Developed by Priyadharshini</b>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR AUTHOR
# =========================
st.sidebar.markdown("## 👩‍💻 Project Author")
st.sidebar.info("**Priyadharshini**\n\nData Science & Machine Learning Project")

# =========================
# TOP BAR
# =========================
t1, t2 = st.columns([3, 2])
with t1:
    st.markdown('<div class="section-title">🔗 Backend API Connection</div>', unsafe_allow_html=True)
with t2:
    api_url = st.text_input("Prediction API URL", value=DEFAULT_API_URL)

# =========================
# FILTERS
# =========================
st.markdown('<div class="section-title">🎛️ Smart Dataset Filters</div>', unsafe_allow_html=True)
with st.expander("Open Filters", expanded=True):
    f1, f2, f3, f4 = st.columns(4)

    job_options = sorted(df["job"].dropna().unique().tolist()) if "job" in df.columns else []
    marital_options = sorted(df["marital"].dropna().unique().tolist()) if "marital" in df.columns else []
    education_options = sorted(df["education"].dropna().unique().tolist()) if "education" in df.columns else []
    month_options = sorted(df["month"].dropna().unique().tolist()) if "month" in df.columns else []

    with f1:
        selected_job = st.selectbox("Job", ["All"] + job_options, index=0)
    with f2:
        selected_marital = st.selectbox("Marital", ["All"] + marital_options, index=0)
    with f3:
        selected_education = st.selectbox("Education", ["All"] + education_options, index=0)
    with f4:
        selected_month = st.selectbox("Month", ["All"] + month_options, index=0)

    f5, f6 = st.columns(2)
    with f5:
        bal_min = int(df["balance"].min()) if "balance" in df.columns else 0
        bal_max = int(df["balance"].max()) if "balance" in df.columns else 100000
        balance_range = st.slider("Balance Range", bal_min, bal_max, (bal_min, bal_max))
    with f6:
        age_min = int(df["age"].min()) if "age" in df.columns else 18
        age_max = int(df["age"].max()) if "age" in df.columns else 95
        age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

filtered_df = df.copy()
if selected_job != "All":
    filtered_df = filtered_df[filtered_df["job"] == selected_job]
if selected_marital != "All":
    filtered_df = filtered_df[filtered_df["marital"] == selected_marital]
if selected_education != "All":
    filtered_df = filtered_df[filtered_df["education"] == selected_education]
if selected_month != "All":
    filtered_df = filtered_df[filtered_df["month"] == selected_month]

filtered_df = filtered_df[
    (filtered_df["balance"] >= balance_range[0]) &
    (filtered_df["balance"] <= balance_range[1]) &
    (filtered_df["age"] >= age_range[0]) &
    (filtered_df["age"] <= age_range[1])
]

# =========================
# KPI CARDS
# =========================
st.markdown('<div class="section-title">📌 Key Performance Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
total_customers = len(filtered_df)
avg_balance = filtered_df["balance"].mean() if len(filtered_df) else 0
avg_age = filtered_df["age"].mean() if len(filtered_df) else 0
sub_rate = filtered_df["y_num"].mean() * 100 if "y_num" in filtered_df.columns and len(filtered_df) else 0

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Customers</div>
        <div class="kpi-value">{total_customers:,}</div>
        <div class="kpi-note">Filtered records</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Balance</div>
        <div class="kpi-value">{avg_balance:,.0f}</div>
        <div class="kpi-note">Financial profile</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Age</div>
        <div class="kpi-value">{avg_age:.1f}</div>
        <div class="kpi-note">Customer age mix</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Subscription Rate</div>
        <div class="kpi-value">{sub_rate:.2f}%</div>
        <div class="kpi-note">Observed conversion rate</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📈 Advanced Analytics",
    "🎯 Live Prediction",
    "🤖 AI Assistant",
    "🔄 What-If Lab",
    "💡 Recommendations",
    "📘 API Guide"
])

# =========================
# TAB 1 - OVERVIEW
# =========================
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        if "y_label" in filtered_df.columns and len(filtered_df):
            fig = px.pie(filtered_df, names="y_label", hole=0.55, title="Subscription Distribution")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if all(col in filtered_df.columns for col in ["job", "y_num"]) and len(filtered_df):
            tmp = (
                filtered_df.groupby("job", as_index=False)["y_num"]
                .mean()
                .sort_values("y_num", ascending=False)
                .head(10)
            )
            tmp["y_num"] *= 100
            fig = px.bar(tmp, x="job", y="y_num", text_auto=".1f", title="Top Job-wise Subscription Rate (%)")
            fig.update_layout(height=430, yaxis_title="Rate %", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if all(col in filtered_df.columns for col in ["month", "y_num"]) and len(filtered_df):
            month_df = filtered_df.groupby("month", as_index=False)["y_num"].mean()
            month_df["y_num"] *= 100
            fig = px.line(month_df, x="month", y="y_num", markers=True, title="Month-wise Conversion Trend (%)")
            fig.update_layout(height=430, yaxis_title="Rate %", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if all(col in filtered_df.columns for col in ["balance", "y_label"]) and len(filtered_df):
            fig = px.histogram(
                filtered_df, x="balance", color="y_label", nbins=35,
                barmode="overlay", title="Balance Distribution by Subscription Class"
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2 - ADVANCED ANALYTICS
# =========================
with tab2:
    a1, a2 = st.columns(2)

    with a1:
        if all(col in filtered_df.columns for col in ["education", "marital", "y_num"]) and len(filtered_df):
            pivot = filtered_df.pivot_table(
                index="education", columns="marital", values="y_num", aggfunc="mean"
            ).fillna(0) * 100
            fig = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index)))
            fig.update_layout(title="Education vs Marital Conversion Heatmap (%)", height=430)
            st.plotly_chart(fig, use_container_width=True)

    with a2:
        if all(col in filtered_df.columns for col in ["campaign", "y_num"]) and len(filtered_df):
            camp = filtered_df.groupby("campaign", as_index=False)["y_num"].mean()
            camp["y_num"] *= 100
            fig = px.bar(camp, x="campaign", y="y_num", text_auto=".1f", title="Campaign Contacts vs Conversion Rate (%)")
            fig.update_layout(height=430, yaxis_title="Rate %")
            st.plotly_chart(fig, use_container_width=True)

    b1, b2 = st.columns(2)

    with b1:
        if all(col in filtered_df.columns for col in ["poutcome", "y_label"]) and len(filtered_df):
            fig = px.sunburst(filtered_df, path=["poutcome", "y_label"], title="Previous Outcome → Subscription Flow")
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)

    with b2:
        if all(col in filtered_df.columns for col in ["age_band", "y_label"]) and len(filtered_df):
            age_df = filtered_df.groupby(["age_band", "y_label"], as_index=False).size()
            fig = px.bar(age_df, x="age_band", y="size", color="y_label", barmode="group", title="Age Band vs Subscription Class")
            fig.update_layout(height=460, yaxis_title="Customers", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📋 Attractive Customer Table</div>', unsafe_allow_html=True)
    if len(filtered_df):
        show_cols = [c for c in [
            "age", "job", "marital", "education", "balance",
            "housing", "loan", "contact", "campaign",
            "pdays", "previous", "poutcome", "y_label"
        ] if c in filtered_df.columns]
        table_df = filtered_df[show_cols].copy()
        st.dataframe(table_df.head(30), use_container_width=True, height=430)
        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Filtered Customers CSV", data=csv_bytes, file_name="filtered_customers.csv", mime="text/csv")

# =========================
# TAB 3 - LIVE PREDICTION
# =========================
with tab3:
    st.markdown('<div class="section-title">🎯 Live Customer Prediction</div>', unsafe_allow_html=True)

    default_job = safe_mode(df["job"], "admin.")
    default_marital = safe_mode(df["marital"], "single")
    default_education = safe_mode(df["education"], "secondary")
    default_contact = safe_mode(df["contact"], "cellular")
    default_month = safe_mode(df["month"], "may")
    default_poutcome = safe_mode(df["poutcome"], "unknown")

    c1, c2, c3 = st.columns(3)

    job_list = sorted(df["job"].dropna().unique().tolist())
    marital_list = sorted(df["marital"].dropna().unique().tolist())
    education_list = sorted(df["education"].dropna().unique().tolist())
    contact_list = sorted(df["contact"].dropna().unique().tolist())
    month_list = sorted(df["month"].dropna().unique().tolist())
    poutcome_list = sorted(df["poutcome"].dropna().unique().tolist())

    with c1:
        age = st.slider("Age", 18, 95, 35)
        job = st.selectbox("Job", job_list, index=job_list.index(default_job) if default_job in job_list else 0)
        marital = st.selectbox("Marital", marital_list, index=marital_list.index(default_marital) if default_marital in marital_list else 0)
        education = st.selectbox("Education", education_list, index=education_list.index(default_education) if default_education in education_list else 0)
        default = st.selectbox("Default", ["no", "yes"], index=0)
        balance = st.number_input("Balance", min_value=-5000, max_value=100000, value=1500, step=100)

    with c2:
        housing = st.selectbox("Housing Loan", ["no", "yes"], index=1)
        loan = st.selectbox("Personal Loan", ["no", "yes"], index=0)
        contact = st.selectbox("Contact Type", contact_list, index=contact_list.index(default_contact) if default_contact in contact_list else 0)
        day = st.slider("Last Contact Day", 1, 31, 10)
        month = st.selectbox("Last Contact Month", month_list, index=month_list.index(default_month) if default_month in month_list else 0)

    with c3:
        campaign = st.number_input("Current Campaign Contacts", min_value=1, max_value=50, value=1, step=1)
        pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=999, value=0, step=1)
        previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0, step=1)
        poutcome = st.selectbox("Previous Campaign Outcome", poutcome_list, index=poutcome_list.index(default_poutcome) if default_poutcome in poutcome_list else 0)

    payload = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": float(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome
    }

    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    if "latest_payload" not in st.session_state:
        st.session_state.latest_payload = None

    if st.button("🔮 Predict Subscription", use_container_width=True, key="predict_btn_main"):
        result, error = call_api(payload, api_url)
        if error:
            st.error(f"Prediction failed: {error}")
        else:
            st.session_state.latest_prediction = result
            st.session_state.latest_payload = payload

    latest = st.session_state.latest_prediction
    latest_payload = st.session_state.latest_payload

    if latest is not None and latest_payload is not None:
        prob = float(latest.get("probability", 0))
        prob_percent = float(latest.get("probability_percent", prob * 100))
        prediction = latest.get("prediction", "Unknown")
        segment = latest.get("lead_segment", lead_segment_from_prob(prob))

        r1, r2, r3 = st.columns(3)
        r1.metric("Subscription Probability", f"{prob_percent:.2f}%")
        r2.metric("Predicted Class", prediction)
        r3.metric("Lead Segment", segment)

        g1, g2 = st.columns([2, 1])

        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_percent,
                title={"text": "Probability Gauge"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            gauge.update_layout(height=350)
            st.plotly_chart(gauge, use_container_width=True)

        with g2:
            st.markdown("#### ⚡ Live Insight")
            if prob >= 0.60:
                st.success("Customer is highly likely to subscribe.")
            elif prob >= 0.30:
                st.info("Customer has moderate subscription potential.")
            else:
                st.warning("Customer is currently unlikely to subscribe.")
            st.progress(int(max(0, min(100, prob_percent))))
            st.caption("Live conversion confidence meter")

        heading, reasons = explain_prediction(prob, latest_payload)
        st.markdown(f"### 🧠 {heading}")
        for i, reason in enumerate(reasons, 1):
            st.write(f"{i}. {reason}")

        st.markdown("### 💬 Smart Recommendations")
        recs = get_recommendations(
            prob,
            latest_payload["balance"],
            latest_payload["campaign"],
            latest_payload["previous"],
            latest_payload["poutcome"],
            latest_payload["housing"],
            latest_payload["loan"]
        )
        for i, rec in enumerate(recs, 1):
            st.write(f"{i}. {rec}")

        st.markdown('<div class="voice-box">🎤 <b>Voice Result Output</b><br>Click the button below to make the browser speak the prediction result.</div>', unsafe_allow_html=True)

        if st.button("🔊 Speak Prediction Result", key="speak_prediction_btn"):
            speech_text = build_voice_text(prob_percent, prediction, segment)
            render_autoplay_audio(speech_text)

# =========================
# TAB 4 - AI ASSISTANT
# =========================
with tab4:
    st.markdown('<div class="section-title">🤖 Mini AI-Style Assistant</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="assistant-box">
        Ask smart questions like:
        <br><br>
        • why is this lead low potential?
        <br>
        • how to improve subscription chance?
        <br>
        • summarize this customer profile
        <br>
        • give a sales pitch
        <br>
        • what factors affect prediction?
        <br>
        • is this customer worth targeting?
        <br>
        • what strategy should bank use?
        <br>
        • what if balance increases?
    </div>
    """, unsafe_allow_html=True)

    question_options = [
        "Why is this lead low potential?",
        "How to improve subscription chance?",
        "Summarize this customer profile",
        "Give a sales pitch",
        "What factors affect prediction?",
        "Is this customer worth targeting?",
        "What strategy should bank use?",
        "What if balance increases?"
    ]

    selected_q = st.selectbox("Choose a smart question", question_options, key="assistant_select_q")
    custom_q = st.text_input("Or type your own question", key="assistant_custom_q")
    final_q = custom_q.strip() if custom_q.strip() else selected_q

    if st.button("Ask Assistant", key="assistant_btn_main"):
        latest = st.session_state.get("latest_prediction")
        latest_payload = st.session_state.get("latest_payload")

        if latest is None or latest_payload is None:
            st.warning("First run a prediction in the Live Prediction tab.")
        else:
            prob = float(latest.get("probability", 0))
            prediction = latest.get("prediction", "Unknown")
            segment = latest.get("lead_segment", lead_segment_from_prob(prob))
            answer = build_assistant_response(final_q, latest_payload, prob, prediction, segment)
            st.markdown("### Assistant Response")
            st.info(answer)

# =========================
# TAB 5 - WHAT IF
# =========================
with tab5:
    st.markdown('<div class="section-title">🔄 What-If Scenario Lab</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Base Scenario")
        base_balance = st.number_input("Base Balance", 0, 100000, 1000, key="base_balance")
        base_campaign = st.number_input("Base Campaign", 1, 50, 3, key="base_campaign")
        base_previous = st.number_input("Base Previous", 0, 50, 0, key="base_previous")
        base_poutcome = st.selectbox("Base Poutcome", sorted(df["poutcome"].dropna().unique().tolist()), key="base_poutcome")

    with right:
        st.markdown("#### Improved Scenario")
        improved_balance = st.number_input("Improved Balance", 0, 100000, 5000, key="improved_balance")
        improved_campaign = st.number_input("Improved Campaign", 1, 50, 1, key="improved_campaign")
        improved_previous = st.number_input("Improved Previous", 0, 50, 2, key="improved_previous")
        pout_values = sorted(df["poutcome"].dropna().unique().tolist())
        improved_poutcome = st.selectbox("Improved Poutcome", pout_values, index=min(1, len(pout_values)-1), key="improved_poutcome")

    if st.button("Compare Scenarios", use_container_width=True, key="compare_scenarios_btn"):
        common = {
            "age": 35,
            "job": default_job,
            "marital": default_marital,
            "education": default_education,
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": default_contact,
            "day": 15,
            "month": default_month,
            "pdays": 5
        }

        base_payload = {
            **common,
            "balance": float(base_balance),
            "campaign": int(base_campaign),
            "previous": int(base_previous),
            "poutcome": base_poutcome
        }
        improved_payload = {
            **common,
            "balance": float(improved_balance),
            "campaign": int(improved_campaign),
            "previous": int(improved_previous),
            "poutcome": improved_poutcome
        }

        base_result, base_error = call_api(base_payload, api_url)
        improved_result, improved_error = call_api(improved_payload, api_url)

        if base_error:
            st.error(f"Base scenario failed: {base_error}")
        elif improved_error:
            st.error(f"Improved scenario failed: {improved_error}")
        else:
            comp = pd.DataFrame({
                "Scenario": ["Base", "Improved"],
                "Probability (%)": [
                    base_result.get("probability_percent", 0),
                    improved_result.get("probability_percent", 0)
                ]
            })
            fig = px.bar(comp, x="Scenario", y="Probability (%)", text_auto=".2f", title="Scenario Probability Comparison")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 6 - RECOMMENDATIONS
# =========================
with tab6:
    st.markdown('<div class="section-title">💡 Strategic Business Recommendations</div>', unsafe_allow_html=True)

    st.info("""
    - Focus on higher-balance customers and successful previous campaign profiles.
    - Reduce excessive campaign contact pressure.
    - Use targeted messaging for medium-potential customers.
    - Promote trust for customers with no previous history or active loans.
    """)

    c1, c2 = st.columns(2)

    with c1:
        if all(col in filtered_df.columns for col in ["job", "y_num"]) and len(filtered_df):
            top_job = (
                filtered_df.groupby("job", as_index=False)["y_num"]
                .mean()
                .sort_values("y_num", ascending=False)
                .head(10)
            )
            top_job["y_num"] *= 100
            fig = px.bar(top_job, x="job", y="y_num", text_auto=".1f", title="Best Converting Job Segments (%)")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if all(col in filtered_df.columns for col in ["month", "y_num"]) and len(filtered_df):
            top_month = (
                filtered_df.groupby("month", as_index=False)["y_num"]
                .mean()
                .sort_values("y_num", ascending=False)
            )
            top_month["y_num"] *= 100
            fig = px.bar(top_month, x="month", y="y_num", text_auto=".1f", title="Best Converting Months (%)")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 7 - API GUIDE
# =========================
with tab7:
    st.markdown('<div class="section-title">📘 API Guide</div>', unsafe_allow_html=True)

    st.write("This dashboard does not fetch predictions directly from the model.")
    st.write("It sends input data to the backend FastAPI service and receives the prediction result.")

    st.markdown("#### Run API")
    st.code("python -m uvicorn api:app --reload", language="bash")

    st.markdown("#### Open API Docs")
    st.code("http://127.0.0.1:8000/docs", language="text")

    st.markdown("#### Test JSON")
    st.code(
"""{
  "age": 35,
  "job": "management",
  "marital": "single",
  "education": "tertiary",
  "default": "no",
  "balance": 5000,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "sep",
  "campaign": 1,
  "pdays": 5,
  "previous": 2,
  "poutcome": "success"
}""",
        language="json"
    )

    st.markdown("#### Run Streamlit")
    st.code("streamlit run app.py", language="bash")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<div class='small-note'>Built with Streamlit frontend + FastAPI backend for bank term deposit subscription prediction<br><b>Author:</b> Priyadharshini</div>",
    unsafe_allow_html=True
)
