"""
AQI Calculation Script for MongoDB
Fetches raw AQI + weather data from MongoDB,
Calculates AQI and category using EPA standards,
Stores processed data into a new MongoDB collection
"""

import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

# =========================
# Load Environment Variables
# =========================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
CITY = os.getenv("CITY_NAME", "Gujrat")
LAT = float(os.getenv("CITY_LATITUDE", "32.5731"))
LON = float(os.getenv("CITY_LONGITUDE", "74.1005"))

# =========================
# MongoDB Setup
# =========================
client = MongoClient(MONGO_URI)
db = client["aqi_project"]

raw_collection = db["raw_data"]
processed_collection = db["processed_data"]

# =========================
# EPA AQI Breakpoints
# =========================
PM25_BREAKPOINTS = [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                    (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
PM10_BREAKPOINTS = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
                    (255,354,151,200),(355,424,201,300),(425,604,301,500)]
O3_BREAKPOINTS = [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),(106,200,201,300)]
NO2_BREAKPOINTS = [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,2049,301,500)]
SO2_BREAKPOINTS = [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300),(605,1004,301,500)]
CO_BREAKPOINTS = [(0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500)]

# =========================
# AQI Calculation Functions
# =========================
def calculate_aqi_single(concentration, breakpoints):
    if pd.isna(concentration) or concentration < 0:
        return np.nan
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            return round(((i_high - i_low)/(c_high - c_low)) * (concentration - c_low) + i_low)
    return 500

def calculate_aqi_pm25(pm25): return calculate_aqi_single(pm25, PM25_BREAKPOINTS)
def calculate_aqi_pm10(pm10): return calculate_aqi_single(pm10, PM10_BREAKPOINTS)
def calculate_aqi_o3(o3): return calculate_aqi_single(o3, O3_BREAKPOINTS)
def calculate_aqi_no2(no2): return calculate_aqi_single(no2, NO2_BREAKPOINTS)
def calculate_aqi_so2(so2): return calculate_aqi_single(so2, SO2_BREAKPOINTS)
def calculate_aqi_co(co):
    co_ppm = co / 1145.0 if not pd.isna(co) else np.nan
    return calculate_aqi_single(co_ppm, CO_BREAKPOINTS)

def get_aqi_category(aqi):
    if pd.isna(aqi): return "Unknown"
    elif aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

# =========================
# Main Function
# =========================
def main():
    print(f"\nFetching raw data for {CITY} from MongoDB...")
    
    # Fetch all documents from raw_data
    cursor = raw_collection.find({})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("No data found in raw_data collection.")
        return
    
    # Remove MongoDB internal column
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    
    # Drop duplicates & missing PM values
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.dropna(subset=["pm2_5", "pm10"], inplace=True)
    
    print(f"Rows fetched: {len(df)}")
    
    # Calculate AQI for each pollutant
    df['aqi_pm25'] = df['pm2_5'].apply(calculate_aqi_pm25)
    df['aqi_pm10'] = df['pm10'].apply(calculate_aqi_pm10)
    df['aqi_o3'] = df['ozone'].apply(calculate_aqi_o3)
    df['aqi_no2'] = df['nitrogen_dioxide'].apply(calculate_aqi_no2)
    df['aqi_so2'] = df['sulphur_dioxide'].apply(calculate_aqi_so2)
    df['aqi_co'] = df['carbon_monoxide'].apply(calculate_aqi_co)
    
    # Calculate overall AQI
    aqi_columns = ['aqi_pm25','aqi_pm10','aqi_o3','aqi_no2','aqi_so2','aqi_co']
    df['aqi'] = df[aqi_columns].max(axis=1)
    df['aqi_category'] = df['aqi'].apply(get_aqi_category)
    
    # Insert processed data into MongoDB
    records = df.to_dict(orient="records")
    processed_collection.delete_many({})  # optional: clear old data
    processed_collection.insert_many(records)
    
    print(f"Processed data inserted into MongoDB successfully!")
    print(f"Collection: processed_data")
    print(f"Rows: {len(df)}")
    print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    main()