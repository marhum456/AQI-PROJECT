"""
Hourly Data Collection and Feature Engineering
Fetches latest data from OpenMeteo API, calculates AQI, engineers features, and stores in MongoDB
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.database.mongo_db import MongoDB


def fetch_latest_weather_data():
    """Fetch latest 1 hour of data from OpenMeteo API."""
    
    print("Fetching latest weather data from OpenMeteo API...")
    
    # Islamabad coordinates
    latitude = 32.5731
    longitude = 74.1005
    
    # Get current time and 1 hour ago
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    # Format dates for API
    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    
    # OpenMeteo API URLs - FIXED to use working endpoints
    # Using the same endpoints that work for historical data collection
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    
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
        # Fetch air quality data
        print(f"Fetching from: {air_url}")
        response_aq = requests.get(air_url, params=air_params, timeout=120)
        response_aq.raise_for_status()
        data_aq = response_aq.json()
        
        # Fetch weather data
        print(f"Fetching from: {weather_url}")
        response_weather = requests.get(weather_url, params=weather_params, timeout=120)
        response_weather.raise_for_status()
        data_weather = response_weather.json()
        
        # Combine data
        df_aq = pd.DataFrame(data_aq['hourly'])
        df_weather = pd.DataFrame(data_weather['hourly'])
        
        # Merge on timestamp
        df = pd.merge(df_aq, df_weather, on='time', how='inner')
        df = df.rename(columns={'time': 'timestamp'})
        
        # Get only the latest hour
        df = df.tail(1)
        
        print(f"Fetched data for: {df['timestamp'].iloc[0]}\n")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def calculate_aqi(row):
    """Calculate AQI from pollutant concentrations."""
    
    def calculate_sub_index(concentration, breakpoints):
        """Calculate sub-index for a pollutant."""
        for i in range(len(breakpoints) - 1):
            c_low, c_high, i_low, i_high = breakpoints[i]
            if c_low <= concentration <= c_high:
                return ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
        return breakpoints[-1][3]
    
    # AQI breakpoints (concentration_low, concentration_high, index_low, index_high)
    pm25_bp = [
        (0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500, 301, 500)
    ]
    
    pm10_bp = [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)
    ]
    
    # Calculate sub-indices
    pm25_aqi = calculate_sub_index(row['pm2_5'], pm25_bp)
    pm10_aqi = calculate_sub_index(row['pm10'], pm10_bp)
    
    # Return maximum (worst) AQI
    return max(pm25_aqi, pm10_aqi)


def engineer_features(df, mongo):
    """Engineer time-based and lag features."""
    
    print("Engineering features...")
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate AQI
    df['aqi'] = df.apply(calculate_aqi, axis=1)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Get lag features from MongoDB (last 24 hours)
    collection = mongo.get_collection("aqi_features")
    recent_data = list(collection.find().sort("timestamp", -1).limit(24))
    
    if len(recent_data) > 0:
        recent_df = pd.DataFrame(recent_data)
        df['aqi_lag_1'] = recent_df['aqi'].iloc[0]  # 1 hour ago
        df['aqi_lag_24'] = recent_df['aqi'].iloc[-1] if len(recent_df) >= 24 else recent_df['aqi'].iloc[0]
    else:
        # First run - use current AQI as lag
        df['aqi_lag_1'] = df['aqi']
        df['aqi_lag_24'] = df['aqi']
    
    print(f"Engineered features for timestamp: {df['timestamp'].iloc[0]}")
    print(f"Calculated AQI: {df['aqi'].iloc[0]:.0f}\n")
    
    return df


def store_to_mongodb(df, mongo):
    """Store engineered features to MongoDB."""
    
    print("Storing data to MongoDB...")
    
    collection = mongo.get_collection("aqi_features")
    
    # Convert to dictionary
    record = df.to_dict('records')[0]
    
    # Check if timestamp already exists (avoid duplicates)
    existing = collection.find_one({"timestamp": record['timestamp']})
    
    if existing:
        print(f"Data for {record['timestamp']} already exists. Skipping.\n")
        return False
    
    # Insert new record
    collection.insert_one(record)
    
    print(f"Stored 1 record to MongoDB")
    print(f"Timestamp: {record['timestamp']}")
    print(f"AQI: {record['aqi']:.0f}\n")
    
    return True


def main():
    """Main hourly data collection pipeline."""
    
    print("Hourly Data Collection Pipeline\n")
    
    # Fetch latest data
    df = fetch_latest_weather_data()
    
    if df is None or df.empty:
        print("Failed to fetch data. Exiting.")
        return
    
    # Connect to MongoDB
    mongo = MongoDB()
    if not mongo.connect():
        print("Failed to connect to MongoDB. Exiting.")
        return
    
    # Engineer features
    df = engineer_features(df, mongo)
    
    # Store to MongoDB
    success = store_to_mongodb(df, mongo)
    
    mongo.close()
    
    if success:
        print("Data collection complete!")
    else:
        print("Data collection skipped (duplicate).")


if __name__ == "__main__":
    main()