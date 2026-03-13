"""
Data Collector Script for AQI Prediction Project
Fetches last 90 days AQI + Weather data
Stores into MongoDB
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

# =========================
# Load Environment Variables
# =========================
load_dotenv()

CITY = os.getenv("CITY_NAME", "Gujrat")
LAT = float(os.getenv("CITY_LATITUDE", "32.5731"))
LON = float(os.getenv("CITY_LONGITUDE", "74.1005"))

MONGO_URI = os.getenv("MONGO_URI")

# =========================
# Date Range (Last 90 Days)
# =========================
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=90)

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

# =========================
# API URLs
# =========================
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_PARAMS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
]

WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover"
]

# =========================
# MongoDB Setup
# =========================
client = MongoClient(MONGO_URI)
db = client["aqi_project"]

raw_collection = db["raw_data"]
weather_collection = db["weather_data"]

# Optional: Clear old data
raw_collection.delete_many({})
weather_collection.delete_many({})


# =========================
# API Fetch Function
# =========================
def fetch_api(url, params):
    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json().get("hourly", {})

        if "time" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


# =========================
# Main Function
# =========================
def main():

    print(f"\nFetching data for {CITY} ({LAT}, {LON})")
    print(f"Date range: {start_str} to {end_str}")

    # Fetch Air Quality Data
    air_df = fetch_api(AIR_URL, {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": ",".join(AIR_PARAMS),
        "timezone": "Asia/Karachi"
    })

    # Fetch Weather Data
    weather_df = fetch_api(WEATHER_URL, {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": ",".join(WEATHER_PARAMS),
        "timezone": "Asia/Karachi"
    })

    if air_df.empty or weather_df.empty:
        print("Failed to fetch API data.")
        return

    # Merge datasets
    df = pd.merge(air_df, weather_df, on="timestamp", how="inner")

    # Add metadata
    df["city"] = CITY
    df["latitude"] = LAT
    df["longitude"] = LON

    # Drop duplicates & missing PM values
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.dropna(subset=["pm2_5", "pm10"], inplace=True)

    print(f"Rows collected: {len(df)}")

    # Insert into MongoDB
    records = df.to_dict(orient="records")
    raw_collection.insert_many(records)

    print("Data inserted into MongoDB successfully!")
    print(f"Collection: raw_data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()