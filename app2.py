"""
AQI Predictor Dashboard
WHO Dark Mode + Glass UI + Animated Gauge
Cloud Deployment Ready
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# =========================
# Config
# =========================

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

st.set_page_config(
    page_title="AQI Monitoring Portal",
    page_icon="🌍",
    layout="wide"
)

# =========================
# WHO Dark Theme + Glass UI
# =========================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: #E2E8F0;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 30px;
}

/* Headings */
h1, h2, h3 {
    color: #F1F5F9;
}

/* Subtext */
.subtitle {
    color: #94A3B8;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Helper Functions
# =========================

def get_aqi_color(aqi):
    if aqi <= 50:
        return "#22C55E"
    elif aqi <= 100:
        return "#EAB308"
    elif aqi <= 150:
        return "#F97316"
    elif aqi <= 200:
        return "#EF4444"
    elif aqi <= 300:
        return "#A855F7"
    else:
        return "#7F1D1D"


def load_model_registry():
    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]
        return db["final_model_registry"].find_one()
    except:
        return None
    finally:
        client.close()


def load_predictions():
    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]
        df = pd.DataFrame(list(db["predictions"].find()))
        if not df.empty:
            df = df.drop(columns=['_id'], errors='ignore')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values("timestamp")
        return df
    except:
        return pd.DataFrame()
    finally:
        client.close()


def load_latest_aqi():
    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]
        return db["engineered_data_final"].find_one(sort=[("timestamp", -1)])
    except:
        return None
    finally:
        client.close()

# =========================
# Load Data
# =========================

registry = load_model_registry()
predictions_df = load_predictions()
latest_data = load_latest_aqi()

# =========================
# HEADER
# =========================

st.markdown("""
<div style="text-align:center;">
    <h1>🌍Air Quality Monitoring Portal</h1>
    <p class="subtitle">
    Environmental Intelligence System – Gujrat, Pakistan <br>
    72-Hour Forecast & Predictive Risk Assessment
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# AQI GAUGE SECTION
# =========================

if not predictions_df.empty:
    current_aqi = int(predictions_df.iloc[0]['predicted_aqi'])
elif latest_data:
    current_aqi = int(latest_data['aqi'])
else:
    current_aqi = 0

color = get_aqi_color(current_aqi)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=current_aqi,
    number={'font': {'size': 60}},
    gauge={
        'axis': {'range': [None, 400]},
        'bar': {'color': color},
        'steps': [
            {'range': [0, 50], 'color': "#22C55E"},
            {'range': [50, 100], 'color': "#EAB308"},
            {'range': [100, 150], 'color': "#F97316"},
            {'range': [150, 200], 'color': "#EF4444"},
            {'range': [200, 300], 'color': "#A855F7"},
            {'range': [300, 400], 'color': "#7F1D1D"},
        ],
    }
))

fig.update_layout(
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    font={'color': "#F1F5F9"}
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="text-align:center;">
    <p class="subtitle">Current Estimated Air Quality Index</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FORECAST SECTION
# =========================

if not predictions_df.empty:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### 72 Hour Predictive Trend Analysis")

    fig = px.line(
        predictions_df,
        x='timestamp',
        y='predicted_aqi'
    )

    fig.update_traces(
        line=dict(width=3, color="#38BDF8")
    )

    fig.update_layout(
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MODEL PERFORMANCE
# =========================

if registry:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### Model Evaluation & Selection")

    models_data = registry['models']

    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
        'R² Score': [
            models_data['random_forest']['r2'],
            models_data['xgboost']['r2'],
            models_data['lightgbm']['r2']
        ]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['R² Score'],
        marker_color="#60A5FA"
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        yaxis_title="R² Score"
    )

    st.plotly_chart(fig, use_container_width=True)

    best_model = comparison_df.loc[
        comparison_df['R² Score'].idxmax(), 'Model'
    ]

    st.markdown(f"""
    <div style='
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 12px;
        margin-top: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    '>
        <strong>Selected Production Model:</strong> {best_model}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption(f"System Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")