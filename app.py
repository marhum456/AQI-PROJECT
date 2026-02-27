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

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: #F8FAFC;  /* Brighter default text */
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 30px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #FFFFFF;
}

/* Subtitle - FIXED (Brighter Color) */
.subtitle {
    color: #E2E8F0;   /* Much brighter */
    font-size: 16px;
    font-weight: 400;
}

/* Markdown text */
p, span, label {
    color: #F1F5F9 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #E2E8F0 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
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
    <h1>Automated AQI Prediction System Gujrat, Pakistan – 72-Hour Forecast </h1>
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
    # DAILY AQI SUMMARY (PER DAY)
    # =========================

st.markdown("### 📅 3-Day AQI Daily Breakdown")

    # Create date column
predictions_df['date'] = predictions_df['timestamp'].dt.date

    # Group by each day
daily_stats = (
        predictions_df
        .groupby('date')['predicted_aqi']
        .agg(['max', 'min', 'mean'])
        .reset_index()
    )

    # Round average
daily_stats['mean'] = daily_stats['mean'].round(2)

    # Display each day in separate glass-style rows
for i, row in daily_stats.iterrows():

        st.markdown(f"#### Day {i+1} — {row['date']}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Maximum AQI", int(row['max']))
        col2.metric("Minimum AQI", int(row['min']))
        col3.metric("Average AQI", row['mean'])

        st.markdown("---")
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

    # Create full comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
        'R² Score': [
            models_data['random_forest']['r2'],
            models_data['xgboost']['r2'],
            models_data['lightgbm']['r2']
        ],
        'MAE': [
            models_data['random_forest']['mae'],
            models_data['xgboost']['mae'],
            models_data['lightgbm']['mae']
        ],
        'RMSE': [
            models_data['random_forest']['rmse'],
            models_data['xgboost']['rmse'],
            models_data['lightgbm']['rmse']
        ]
    })

    # =========================
    # 1️⃣ SHOW TABLE WITH VALUES
    # =========================
    st.dataframe(comparison_df, use_container_width=True)

    # =========================
    # 2️⃣ BAR CHART WITH VALUES DISPLAYED
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['R² Score'],
        text=[f"{val:.3f}" for val in comparison_df['R² Score']],
        textposition='outside',
        marker_color="#60A5FA"
    ))

    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        yaxis_title="R² Score",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 3️⃣ KPI STYLE DISPLAY FOR BEST MODEL
    # =========================
    best_index = comparison_df['R² Score'].idxmax()
    best_model = comparison_df.loc[best_index, 'Model']

    st.markdown("### 🏆 Selected Production Model")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", best_model)
    col2.metric("R² Score", f"{comparison_df.loc[best_index, 'R² Score']:.4f}")
    col3.metric("RMSE", f"{comparison_df.loc[best_index, 'RMSE']:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================

from datetime import datetime
import pytz

pakistan_tz = pytz.timezone("Asia/Karachi")
local_time = datetime.now(pakistan_tz)


st.caption(f"System Last Updated: {local_time.strftime('%Y-%m-%d %H:%M:%S')} PKT")