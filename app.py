"""
AQI Predictor Dashboard
Simple Streamlit dashboard showing current AQI and 3-day forecast
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import subprocess
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Page config
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="wide"
)



def get_aqi_category(aqi):
    """Get AQI category and color."""
    if aqi <= 50:
        return "Good", "#00E400"
    elif aqi <= 100:
        return "Moderate", "#FFFF00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"


def refresh_predictions():
    """Run prediction pipeline to generate fresh predictions."""
    try:
        with st.spinner("Generating fresh predictions... This may take a minute."):
            result = subprocess.run(
                ["python", "src/pipeline/predict_next_3_days.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                st.success("✅ Predictions updated successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        st.error("❌ Prediction timed out. Please try again.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def collect_fresh_data():
    """Run data collection pipeline to fetch latest AQI data."""
    try:
        with st.spinner("Collecting fresh AQI data from API... This may take a moment."):
            result = subprocess.run(
                ["python", "src/pipeline/new_data_with_features.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                st.success("✅ Fresh data collected successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        st.error("❌ Data collection timed out. Please try again.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")





def load_model_registry():
    """Load model comparison from MongoDB (using .env URI)."""
    
    try:
        # Connect using environment variable
        client = MongoClient(MONGO_URI)
        
        # Select database (change to your DB name)
        db = client["aqi_project"]
        
        # Select collection
        collection = db["final_model_registry"]
        
        # Fetch registry document
        registry = collection.find_one()
        
        return registry
    
    except Exception as e:
        print(f"Error loading model registry: {e}")
        return None
    
    finally:
        client.close()

def load_predictions():
    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]   # change to your DB name
        collection = db["predictions"]
        
        predictions = list(collection.find())
        
        df = pd.DataFrame(predictions)
        
        if not df.empty:
            df = df.drop(columns=['_id'], errors='ignore')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values("timestamp")
        
        return df
    
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return pd.DataFrame()
    
    finally:
        client.close()


def load_latest_aqi():
    """Load latest actual AQI from features (direct MongoClient)."""
    
    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]   # change to your DB name
        collection = db["engineered_data_final"]
        
        # Get latest record sorted by timestamp (descending)
        latest = collection.find_one(sort=[("timestamp", -1)])
        
        return latest
    
    except Exception as e:
        print(f"Error loading latest AQI: {e}")
        return None
    
    finally:
        client.close()

# Main Dashboard
st.title("🌍 AQI Predictor Dashboard")
st.markdown("Real-time Air Quality Index prediction for Gujrat")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    This dashboard shows:
    - Current AQI
    - 3-day forecast
    - Model performance comparison
    
    **Best Model:** Random Forest
    **Accuracy:** R² = 0.77
    """
)

st.sidebar.markdown("### 🤖 Automation")
st.sidebar.caption(
    """
    - **Data Collection**: Every hour
    - **Model Training**: Daily at 2 AM UTC
    - **Predictions**: Daily at 3 AM UTC
    
    *Note: OpenMeteo API data has 1-2 hour delay.  
    Current AQI shows model prediction for accuracy.*
    """
)

# Refresh predictions button
st.sidebar.markdown("---")
st.sidebar.header("Actions")

if st.sidebar.button("📡 Collect Fresh Data", use_container_width=True):
    collect_fresh_data()
st.sidebar.caption("Fetch latest AQI data (runs locally)")

st.sidebar.markdown("")  # Spacing

if st.sidebar.button("🔄 Regenerate Predictions", use_container_width=True):
    refresh_predictions()
st.sidebar.caption("Regenerate 72-hour forecast (runs locally)")

# Load data
registry = load_model_registry()
predictions_df = load_predictions()
latest_data = load_latest_aqi()

# Row 1: Current AQI
st.header("📊 Current AQI")

col1, col2, col3 = st.columns(3)

# Use first prediction as "current" AQI since it's more up-to-date
if not predictions_df.empty:
    # Get the first (earliest) prediction - this is the most current estimate
    first_prediction = predictions_df.iloc[0]
    current_aqi = int(first_prediction['predicted_aqi'])
    current_timestamp = first_prediction['timestamp']
    category, color = get_aqi_category(current_aqi)
    
    with col1:
        st.metric(
            label="Current AQI (Estimated)",
            value=current_aqi,
            delta=category
        )
        # Show when predictions were generated
        if 'prediction_date' in first_prediction:
            pred_date = pd.to_datetime(first_prediction['prediction_date'])
            # Convert to Pakistan time (UTC+5)
            from datetime import timedelta
            pred_date_pk = pred_date + timedelta(hours=5)
            st.caption(f"📊 Predictions generated: {pred_date_pk.strftime('%b %d, %H:%M')} PKT")
        else:
            st.caption("📊 Based on latest model prediction")
    
    with col2:
        st.metric(
            label="Location",
            value="Gujrat, Pakistan"
        )
    
    with col3:
        if registry:
            st.metric(
                label="Model Used",
                value=registry['best_model'].upper()
            )
elif latest_data:
    # Fallback to actual data if no predictions available
    current_aqi = int(latest_data['aqi'])
    category, color = get_aqi_category(current_aqi)
    latest_timestamp = pd.to_datetime(latest_data['timestamp'])
    
    with col1:
        st.metric(
            label="Current AQI (Measured)",
            value=current_aqi,
            delta=category
        )
        st.caption(f"Measured: {latest_timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        st.metric(
            label="Location",
            value="Gujrat, Pakistan"
        )
    
    with col3:
        if registry:
            st.metric(
                label="Model Used",
                value=registry['best_model'].upper()
            )

# Row 2: 3-Day Forecast
st.header("📈 3-Day Forecast")

if not predictions_df.empty:
    # Create forecast chart
    fig = px.line(
        predictions_df,
        x='timestamp',
        y='predicted_aqi',
        title='Next 72 Hours AQI Prediction',
        labels={'predicted_aqi': 'Predicted AQI', 'timestamp': 'Date & Time'}
    )
    
    # Add AQI category zones
    fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily summary
    st.subheader("Daily Summary")
    
    predictions_df['date'] = predictions_df['timestamp'].dt.date
    daily_summary = predictions_df.groupby('date')['predicted_aqi'].agg(['mean', 'min', 'max']).reset_index()
    daily_summary.columns = ['Date', 'Avg AQI', 'Min AQI', 'Max AQI']
    daily_summary['Avg AQI'] = daily_summary['Avg AQI'].round(0).astype(int)
    daily_summary['Min AQI'] = daily_summary['Min AQI'].round(0).astype(int)
    daily_summary['Max AQI'] = daily_summary['Max AQI'].round(0).astype(int)
    
    st.dataframe(daily_summary, use_container_width=True, hide_index=True)
    
    # Show overall AQI range
    overall_min = predictions_df['predicted_aqi'].min()
    overall_max = predictions_df['predicted_aqi'].max()
    st.caption(f"📊 Overall AQI Range: {int(overall_min)}-{int(overall_max)}")

else:
    st.warning("No predictions available. Run prediction pipeline first.")

# Row 3: Model Comparison
st.header("🤖 Model Comparison")

if registry:
    models_data = registry['models']
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
        'R² Score': [
            models_data['random_forest']['r2'],
            models_data['xgboost']['r2'],
            models_data['lightgbm']['r2']
        ],
        'RMSE': [
            models_data['random_forest']['rmse'],
            models_data['xgboost']['rmse'],
            models_data['lightgbm']['rmse']
        ],
        'MAE': [
            models_data['random_forest']['mae'],
            models_data['xgboost']['mae'],
            models_data['lightgbm']['mae']
        ]
    })
    
    # Highlight best model
    best_model_idx = comparison_df['R² Score'].idxmax()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            comparison_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                              .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.success(f"**Best Model:** {comparison_df.loc[best_model_idx, 'Model']}")
        st.info(f"**R² Score:** {comparison_df.loc[best_model_idx, 'R² Score']}")
        st.info(f"**RMSE:** {comparison_df.loc[best_model_idx, 'RMSE']}")
        st.info(f"**MAE:** {comparison_df.loc[best_model_idx, 'MAE']}")
    
    # Model performance chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='R² Score',
        x=comparison_df['Model'],
        y=comparison_df['R² Score'],
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison (R² Score)',
        yaxis_title='R² Score',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No model registry found. Run training pipeline first.")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")