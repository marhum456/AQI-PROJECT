"""
Creating time-based, lag, and interaction features for AQI prediction
(Fetch from MongoDB and Save back to MongoDB)
"""

import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv


# =========================
# MongoDB Setup
# =========================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["aqi_project"]

input_collection = db["processed_data"]
output_collection = db["engineered_data_final"]


def create_time_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Cyclical encoding (keep only this)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def create_lag_features(df):
    """Create limited lag and rolling features (reduced to prevent overfitting)."""
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # =============================
    # Important Lag Features Only
    # =============================
    lag_hours = [1, 3, 6, 24]   # Removed 2,12,48,72,168

    for lag in lag_hours:
        df[f'aqi_lag_{lag}'] = df['aqi'].shift(lag)

    # =============================
    # Small Rolling Windows Only
    # =============================
    rolling_windows = [6, 24]   # Removed 12,48,72,168

    for window in rolling_windows:
        df[f'aqi_roll_mean_{window}'] = df['aqi'].rolling(window).mean()
        df[f'aqi_roll_std_{window}'] = df['aqi'].rolling(window).std()

    # =============================
    # Keep Only 1 Momentum Feature
    # =============================
    df['aqi_change_1h'] = df['aqi'] - df['aqi'].shift(1)

    return df

def create_interaction_features(df):
    df = df.copy()

    # Keep only most meaningful interactions
    df['temp_humidity'] = df['temperature_2m'] * df['relative_humidity_2m']

    # Wind decomposition (important for pollution dispersion)
    df['wind_x'] = df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))
    df['wind_y'] = df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))

    return df

def select_final_features(df):
    """
    Select final reduced feature set for modeling
    (Controlled features to prevent overfitting)
    """

    # =========================
    # Metadata (not for model)
    # =========================
    metadata_cols = ["timestamp", "city", "latitude", "longitude"]

    # =========================
    # Base pollutant features
    # =========================
    base_pollutants = [
        "pm10", "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    # =========================
    # Weather features
    # =========================
    weather_features = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "cloud_cover"
    ]

    # =========================
    # Reduced Time Features
    # (Only cyclical encoding)
    # =========================
    time_features = [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos"
    ]

    # =========================
    # Controlled Lag Features
    # (Only small important lags)
    # =========================
    lag_features = [
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24"
    ]

    # =========================
    # Small Rolling Windows
    # =========================
    rolling_features = [
        "aqi_roll_mean_6",
        "aqi_roll_std_6",
        "aqi_roll_mean_24",
        "aqi_roll_std_24"
    ]

    # =========================
    # Momentum Feature
    # =========================
    change_features = [
        "aqi_change_1h"
    ]

    # =========================
    # Minimal Interaction Features
    # =========================
    interaction_features = [
        "temp_humidity",
        "wind_x",
        "wind_y"
    ]

    # =========================
    # Combine selected features
    # =========================
    feature_cols = (
        base_pollutants
        + weather_features
        + time_features
        + lag_features
        + rolling_features
        + change_features
        + interaction_features
    )

    # Keep only available columns
    feature_cols = [col for col in feature_cols if col in df.columns]

    target_col = "aqi"

    all_cols = metadata_cols + feature_cols + [target_col]
    available_cols = [col for col in all_cols if col in df.columns]

    print(f"\nTotal Features Selected (Reduced): {len(feature_cols)}")

    return df[available_cols], feature_cols

def main():
    """Main function to engineer reduced features and save to MongoDB."""

    print("\nFeature Engineering Pipeline (Reduced Version)")
    print("-" * 60)

    # =========================
    # Load data from MongoDB
    # =========================
    print("Fetching data from MongoDB (processed_data)...")

    cursor = input_collection.find({})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        print("No data found in processed_data collection.")
        return

    # Remove MongoDB internal ID
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    # Ensure timestamp sorted (CRITICAL for lags)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} rows")
    print(f"Initial columns: {len(df.columns)}")

    # =========================
    # Feature Engineering
    # =========================
    print("\nCreating time features...")
    df = create_time_features(df)

    print("Creating lag features...")
    df = create_lag_features(df)

    print("Creating interaction features...")
    df = create_interaction_features(df)

    # =========================
    # Select Reduced Features
    # =========================
    print("\nSelecting reduced feature set...")
    df_final, feature_cols = select_final_features(df)

    # =========================
    # Remove NaN rows (from lag/rolling)
    # =========================
    initial_rows = len(df_final)
    df_final = df_final.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(df_final)

    print(f"Removed {removed_rows} rows due to lag/rolling NaNs")
    print(f"Remaining rows: {len(df_final)}")

    # =========================
    # Save to MongoDB
    # =========================
    print("\nSaving engineered data to MongoDB (engineered_data2)...")

    records = df_final.to_dict(orient="records")

    # Optional reset (safe overwrite)
    output_collection.delete_many({})
    if records:
        output_collection.insert_many(records)

    print("\nData saved successfully!")
    print(f"Final rows: {len(df_final)}")
    print(f"Total columns stored: {len(df_final.columns)}")
    print(f"Model features count: {len(feature_cols)}")

    print("\nFinal Feature List:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"{i:2d}. {feat}")

    print("\n✔ Reduced feature engineering complete.")
    print("Now retrain models and check Train vs Test R² gap.")
    

if __name__ == "__main__":
    main()