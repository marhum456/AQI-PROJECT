
"""
Predict next 3 days (72 hours) AQI
Loads best model from MongoDB registry and generates forecast
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
from pymongo import MongoClient
from dotenv import load_dotenv




import os
import sys
import pickle
from pymongo import MongoClient
from bson import ObjectId
from gridfs import GridFS
from dotenv import load_dotenv


def load_best_model_from_registry():
    """
    Load latest best model from MongoDB GridFS using registry metadata
    """

    print("Loading best model from MongoDB registry...")

    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")

    if not MONGO_URI:
        print("ERROR: MONGO_URI not found")
        sys.exit(1)

    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]

        collection = db["final_model_registry"]
        fs = GridFS(db)

        # get latest model
        registry = collection.find_one({"is_latest": True})

        if not registry:
            print("No latest model found")
            return None, None

        best_model_name = registry["best_model"]
        model_gridfs_id = registry["model_gridfs_id"]

        model_r2 = registry.get("models", {}).get(best_model_name, {}).get("r2", "N/A")

        print(f"Best model: {best_model_name} (R² = {model_r2})")

        # Load model from GridFS
        model_bytes = fs.get(ObjectId(model_gridfs_id)).read()
        model = pickle.loads(model_bytes)

        client.close()

        print("Model loaded successfully\n")

        return model, best_model_name

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    

def get_latest_data_from_mongodb(limit=24):
    """
    Load the latest AQI feature records from MongoDB 'aqi_features' collection.
    Used for generating lag, rolling, and interaction features for forecasting.

    Args:
        limit (int): Number of latest records to fetch (default 24).

    Returns:
        pd.DataFrame: Latest AQI records sorted by timestamp ascending.
    """
    print("Loading latest data from MongoDB...")

    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in .env file")
        sys.exit(1)

    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]
        collection = db["engineered_data_final"]

        # Fetch latest 'limit' records sorted descending by timestamp
        latest_data = list(collection.find().sort("timestamp", -1).limit(limit))
    except Exception as e:
        print(f"ERROR connecting to MongoDB or fetching data: {e}")
        sys.exit(1)
    finally:
        try:
            client.close()
        except:
            pass

    if not latest_data:
        print(f"ERROR: No data found in 'aqi_features' collection!")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(latest_data)

    # Drop MongoDB internal ID
    if "_id" in df.columns:
        df = df.drop("_id", axis=1)

    # Ensure timestamps are datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort ascending so earliest is first
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} recent records\n")
    return df

import pandas as pd
import numpy as np
from datetime import timedelta
import pytz




    

def generate_forecast_features_sequential(latest_df, model, hours_ahead=72):
    """
    Generate features for the next 'hours_ahead' hours sequentially,
    updating lag and rolling features after each prediction.
    Timestamps are converted to Pakistan Time (PKT) to match local date.
    """
    tz = pytz.timezone("Asia/Karachi")
    # Ensure timestamps are timezone-aware and in PKT
    latest_df['timestamp'] = pd.to_datetime(latest_df['timestamp'])
    if latest_df['timestamp'].dt.tz is None:
        latest_df['timestamp'] = latest_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)
    else:
        latest_df['timestamp'] = latest_df['timestamp'].dt.tz_convert(tz)

    # Sort ascending
    latest_df = latest_df.sort_values("timestamp").reset_index(drop=True)

    forecast_data = []

    # Start forecasting from the next hour after the last known timestamp
    last_timestamp = latest_df['timestamp'].max()
    last_timestamp = last_timestamp.ceil('H')  # round up to next full hour

    # Prepare a working DataFrame to update lags sequentially
    working_df = latest_df.copy()

    for i in range(hours_ahead):
        ts = last_timestamp + timedelta(hours=i+1)

        # =====================
        # Time features
        # =====================
        hour = ts.hour
        day_of_week = ts.dayofweek
        month = ts.month

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # =====================
        # Average weather/pollutants
        # =====================
        avg_pm10 = latest_df['pm10'].mean()
        avg_pm25 = latest_df['pm2_5'].mean()
        avg_co = latest_df['carbon_monoxide'].mean()
        avg_no2 = latest_df['nitrogen_dioxide'].mean()
        avg_so2 = latest_df['sulphur_dioxide'].mean()
        avg_o3 = latest_df['ozone'].mean()
        avg_temp = latest_df['temperature_2m'].mean()
        avg_humidity = latest_df['relative_humidity_2m'].mean()
        avg_wind_speed = latest_df['wind_speed_10m'].mean()
        avg_precip = latest_df['precipitation'].mean()
        avg_cloud = latest_df['cloud_cover'].mean()
       

        # =====================
        # Lag features (from working_df)
        # =====================
        aqi_lag_1 = working_df['aqi'].iloc[-1]
        aqi_lag_3 = working_df['aqi'].iloc[-3] if len(working_df) >= 3 else working_df['aqi'].iloc[-1]
        aqi_lag_6 = working_df['aqi'].iloc[-6] if len(working_df) >= 6 else working_df['aqi'].iloc[-1]
        aqi_lag_24 = working_df['aqi'].iloc[-24] if len(working_df) >= 24 else working_df['aqi'].iloc[-1]

        # =====================
        # Rolling features
        # =====================
        aqi_roll_mean_6 = working_df['aqi'].tail(6).mean()
        aqi_roll_std_6 = working_df['aqi'].tail(6).std()
        aqi_roll_mean_24 = working_df['aqi'].tail(24).mean()
        aqi_roll_std_24 = working_df['aqi'].tail(24).std()

        # =====================
        # Change
        # =====================
        aqi_change_1h = aqi_lag_1 - working_df['aqi'].iloc[-2] if len(working_df) >= 2 else 0

        # =====================
        # Interaction features
        # =====================
        temp_humidity = avg_temp * avg_humidity
        if 'wind_direction_10m' in latest_df.columns:
    # Use average wind direction from latest data
            avg_wind_dir = latest_df['wind_direction_10m'].mean()
            wind_dir_rad = np.radians(avg_wind_dir)
            wind_x = avg_wind_speed * np.cos(wind_dir_rad)
            wind_y = avg_wind_speed * np.sin(wind_dir_rad)
        else:
    # Fallback if wind direction not available
            wind_x = avg_wind_speed * hour_sin  # simple proxy
            wind_y = avg_wind_speed * hour_cos  # simple proxy

        # =====================
        # Create row for prediction
        # =====================
        row = {
            'timestamp': ts,
            'pm10': avg_pm10,
            'pm2_5': avg_pm25,
            'carbon_monoxide': avg_co,
            'nitrogen_dioxide': avg_no2,
            'sulphur_dioxide': avg_so2,
            'ozone': avg_o3,
            'temperature_2m': avg_temp,
            'relative_humidity_2m': avg_humidity,
            'wind_speed_10m': avg_wind_speed,
            'precipitation': avg_precip,
            'cloud_cover': avg_cloud,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'aqi_lag_1': aqi_lag_1,
            'aqi_lag_3': aqi_lag_3,
            'aqi_lag_6': aqi_lag_6,
            'aqi_lag_24': aqi_lag_24,
            'aqi_roll_mean_6': aqi_roll_mean_6,
            'aqi_roll_std_6': aqi_roll_std_6,
            'aqi_roll_mean_24': aqi_roll_mean_24,
            'aqi_roll_std_24': aqi_roll_std_24,
            'aqi_change_1h': aqi_change_1h,
            'temp_humidity': temp_humidity,
            'wind_x': wind_x,
            'wind_y': wind_y
        }

        # Predict AQI
        features = [col for col in row.keys() if col != 'timestamp']
        row['aqi'] = model.predict(pd.DataFrame([row])[features])[0]

        # Append to forecast and update working_df
        forecast_data.append(row)
        working_df = pd.concat([working_df, pd.DataFrame([row])], ignore_index=True)

    forecast_df = pd.DataFrame(forecast_data)

    # Ensure all forecast timestamps are in PKT
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp']).dt.tz_convert(tz)

    return forecast_df

def make_predictions(model, forecast_df):
    """Make AQI predictions using 29 trained features."""
    
    features = [
        # Base pollutants (6)
        "pm10", "pm2_5", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide", "ozone",

        # Weather (5)
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "precipitation", "cloud_cover",

        # Time cyclical (6)
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",

        # Lag features (4)
        "aqi_lag_1", "aqi_lag_3",
        "aqi_lag_6", "aqi_lag_24",

        # Rolling features (4)
        "aqi_roll_mean_6", "aqi_roll_std_6",
        "aqi_roll_mean_24", "aqi_roll_std_24",

        # Change feature (1)
        "aqi_change_1h",

        # Interaction features (3)
        "temp_humidity",
        "wind_x", "wind_y"
    ]
    
    X = forecast_df[features]
    predictions = model.predict(X)
    
    forecast_df["predicted_aqi"] = predictions.round().astype(int)
    
    return forecast_df


def save_predictions_to_mongodb(forecast_df, model_name):
    """Save predictions to MongoDB."""
    
    print("Saving predictions to MongoDB...")

    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")

    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in .env file")
        sys.exit(1)

    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)

        # Use your database
        db = client["aqi_project"]

        # Use predictions collection
        collection = db["predictions"]

        # Clear old predictions
        collection.delete_many({})

        # Prepare prediction documents
        predictions = []
        prediction_date = datetime.utcnow().isoformat()

        for _, row in forecast_df.iterrows():
            predictions.append({
                "timestamp": row["timestamp"].isoformat(),
                "predicted_aqi": int(row["predicted_aqi"]),
                "model_used": model_name,
                "prediction_date": prediction_date
            })

        if predictions:
            collection.insert_many(predictions)

        print(f"Saved {len(predictions)} predictions\n")

    except Exception as e:
        print(f"ERROR saving predictions: {e}")
        sys.exit(1)

    finally:
        try:
            client.close()
        except:
            pass


def display_summary(forecast_df):
    """Display prediction summary."""
    
    print("Prediction Summary:\n")
    
    # Group by day
    forecast_df['date'] = forecast_df['timestamp'].dt.date
    daily_avg = forecast_df.groupby('date')['predicted_aqi'].agg(['mean', 'min', 'max'])
    
    for date, row in daily_avg.iterrows():
        print(f"{date}:")
        print(f"  Avg AQI: {row['mean']:.0f}")
        print(f"  Range: {row['min']:.0f} - {row['max']:.0f}\n")


def main():
    """Main prediction pipeline."""
    
    print("3-Day AQI Prediction Pipeline\n")
    
    # Load best model
    model, model_name = load_best_model_from_registry()
    if model is None:
        return
    
    # Get latest data
    latest_df = get_latest_data_from_mongodb()
    
    # Generate forecast features
    forecast_df = generate_forecast_features_sequential(latest_df, model, hours_ahead=72)
    
    # Make predictions
    forecast_df = make_predictions(model, forecast_df)
    
    # Display summary
    display_summary(forecast_df)
    
    # Save to MongoDB
    save_predictions_to_mongodb(forecast_df, model_name)
    
    print("Prediction complete!")


if __name__ == "__main__":
    main()
