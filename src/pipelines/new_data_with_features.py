import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import pytz

import streamlit as st
from pymongo import MongoClient

# =========================
# MongoDB Setup (Streamlit Cloud)
# =========================
MONGO_URI = st.secrets["MONGO_URI"]  # <- use Streamlit secrets
client = MongoClient(MONGO_URI)
db = client["aqi_project"]
engineered_collection = db["engineered_data_final"]

# Pakistan timezone
tz = pytz.timezone("Asia/Karachi")


# =========================
# Fetch latest data (single hour)
# =========================
def fetch_latest_weather_data():
    """Fetch only the latest 1 hour of AQI and weather data in Pakistan time."""
    print("Fetching latest weather data from OpenMeteo API...")

    # Current time in Pakistan (rounded to full hour)
    end_time = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=1)

    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")

    latitude = 32.5731
    longitude = 74.1005

    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url = "https://api.open-meteo.com/v1/forecast"

    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi"
    }

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi"
    }

    try:
        response_aq = requests.get(air_url, params=air_params, timeout=120)
        response_aq.raise_for_status()
        data_aq = response_aq.json()

        response_weather = requests.get(weather_url, params=weather_params, timeout=120)
        response_weather.raise_for_status()
        data_weather = response_weather.json()

        df_aq = pd.DataFrame(data_aq['hourly'])
        df_weather = pd.DataFrame(data_weather['hourly'])

# coordinates 
        latitude = "32.5731"
        longitude = "74.1005"

# Define city manually (since API doesn't return it)
        city = "Gujrat"

        df = pd.merge(df_aq, df_weather, on='time', how='inner')
        df = df.rename(columns={'time': 'timestamp'})

        # Add metadata
        df["city"] = city
        df["latitude"] = latitude
        df["longitude"] = longitude

        # Convert to timezone-aware datetime (if not already)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(tz)

        # Keep only the latest hour
        df = df[df['timestamp'] == end_time]

        if df.empty:
            print("No data available for the current hour yet.")
            return None

        print(f"Fetched data for: {df['timestamp'].iloc[0]}\n")
        return df


    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# =========================
# AQI calculation
# =========================
def calculate_aqi(row):
    """Calculate AQI from PM2.5 and PM10."""
    def sub_index(c, breakpoints):
        for bp in breakpoints:
            c_low, c_high, i_low, i_high = bp
            if c_low <= c <= c_high:
                return ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
        return breakpoints[-1][3]

    pm25_bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
               (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500,301,500)]
    pm10_bp = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
               (255,354,151,200),(355,424,201,300),(425,604,301,500)]

    return max(sub_index(row['pm2_5'], pm25_bp), sub_index(row['pm10'], pm10_bp))


# =========================
# Feature engineering
# =========================
def engineer_features(df, mongo):
    """Engineer full feature set for modeling (lags, rolling, interactions, cyclical)."""
    print("Engineering features...")

    # Ensure timestamp is tz-aware
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(tz)


    df['aqi'] = df.apply(calculate_aqi, axis=1)

    # Time features
    hour = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    day_of_week = df['timestamp'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    month = df['timestamp'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Lag features
    collection = mongo.get_collection("engineered_data_final")
    recent = list(collection.find().sort("timestamp", -1).limit(24))
    recent_df = pd.DataFrame(recent).sort_values("timestamp") if recent else pd.DataFrame()
    df['aqi_lag_1'] = recent_df['aqi'].iloc[-1] if len(recent_df) >= 1 else df['aqi']
    df['aqi_lag_3'] = recent_df['aqi'].iloc[-3] if len(recent_df) >= 3 else df['aqi']
    df['aqi_lag_6'] = recent_df['aqi'].iloc[-6] if len(recent_df) >= 6 else df['aqi']
    df['aqi_lag_24'] = recent_df['aqi'].iloc[0] if len(recent_df) >= 24 else df['aqi']

    # Rolling statistics
    aqi_series = recent_df['aqi'] if len(recent_df) > 0 else df['aqi']
    df['aqi_roll_mean_6'] = aqi_series.tail(6).mean()
    df['aqi_roll_std_6'] = aqi_series.tail(6).std()
    df['aqi_roll_mean_24'] = aqi_series.tail(24).mean()
    df['aqi_roll_std_24'] = aqi_series.tail(24).std()

    # Momentum
    df['aqi_change_1h'] = df['aqi'] - df['aqi_lag_1']

    # Interaction features
    df['temp_humidity'] = df['temperature_2m'] * df['relative_humidity_2m']
    # Wind decomposition (important for pollution dispersion)
    if 'wind_direction_10m' in df.columns:
        df['wind_x'] = df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))
        df['wind_y'] = df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))
    else:
        df['wind_x'] = df['wind_speed_10m'] * df['hour_sin']
        df['wind_y'] = df['wind_speed_10m'] * df['hour_cos']

    print(f"Engineered features for timestamp: {df['timestamp'].iloc[0]}")
    print(f"Calculated AQI: {df['aqi'].iloc[0]:.0f}\n")
    return df


# =========================
# Select final reduced feature set
# =========================
def select_final_features(df):
    metadata_cols = ["timestamp", "city", "latitude", "longitude"]
    base_pollutants = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    weather_features = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cloud_cover"]
    time_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    lag_features = ["aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_24"]
    rolling_features = ["aqi_roll_mean_6", "aqi_roll_std_6", "aqi_roll_mean_24", "aqi_roll_std_24"]
    change_features = ["aqi_change_1h"]
    interaction_features = ["temp_humidity", "wind_x", "wind_y"]

    feature_cols = base_pollutants + weather_features + time_features + lag_features + rolling_features + change_features + interaction_features
    feature_cols = [col for col in feature_cols if col in df.columns]
    target_col = "aqi"
    all_cols = metadata_cols + feature_cols + [target_col]
    available_cols = [col for col in all_cols if col in df.columns]

    print(f"\nTotal Features Selected (Reduced): {len(feature_cols)}")
    return df[available_cols], feature_cols


# =========================
# Store in MongoDB
# =========================
def store_to_mongodb(df, mongo):
    print("Storing data to MongoDB...")
    collection = mongo.get_collection("engineered_data_final")
    record = df.to_dict('records')[0]


    # Check duplicates in MongoDB
    existing = collection.find_one({"timestamp": record['timestamp']})
    if existing:
        print(f"Data for {record['timestamp']} already exists. Skipping.\n")
        return False

    collection.insert_one(record)
    print(f"Stored 1 record to MongoDB")
    print(f"Timestamp: {record['timestamp']}")
    print(f"AQI: {record['aqi']:.0f}\n")
    return True

# =========================
# Main pipeline
# =========================
def main():
    print("\nHourly AQI Data Collection & Feature Engineering Pipeline\n")

    df = fetch_latest_weather_data()
    if df is None or df.empty:
        print("No data fetched. Exiting.")
        return

    # Avoid duplicate timestamps
    existing_timestamps = set(rec['timestamp'] for rec in engineered_collection.find({}, {'timestamp':1}))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[~df['timestamp'].isin(existing_timestamps)]

    if df.empty:
        print("Latest timestamp already exists. Skipping insert.")
        return

    # Engineer features
    df = engineer_features(df, db)

    # Select reduced feature set
    df, feature_cols = select_final_features(df)

    # Drop NaNs if any
    df = df.dropna().reset_index(drop=True)

    # Store
    store_to_mongodb(df, db)


if __name__ == "__main__":
    main()