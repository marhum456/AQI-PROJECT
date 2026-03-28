"""
Train models daily and register best model to MongoDB
Reads features from MongoDB, trains 3 models, selects best, saves metadata
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import pickle



import os


from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from gridfs import GridFS

def load_features_from_mongodb():
    """Load engineered AQI features from MongoDB collection 'engineered_data_final'."""
    
    print("Loading features from MongoDB...")

    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")

    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in .env file")
        sys.exit(1)

    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)

        # Use your database name explicitly
        db = client["aqi_project"]

        # Use correct collection
        collection = db["engineered_data_final"]

        # Fetch data
        data = list(collection.find())

    except Exception as e:
        print(f"ERROR connecting to MongoDB: {e}")
        sys.exit(1)

    finally:
        # Ensure client closes safely
        try:
            client.close()
        except:
            pass

    if not data:
        print("ERROR: No data found in 'engineered_data_final' collection!")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Remove MongoDB internal ID
    if "_id" in df.columns:
        df = df.drop("_id", axis=1)

    print(f"Loaded {len(df)} rows from 'engineered_data_final'\n")

    return df



# ===============================
# Data Cleaning + Missing Filling
# ===============================

def clean_and_fill_data(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    df = df.set_index("timestamp")

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h"
    )

    df = df.reindex(full_range)
    df.index.name = "timestamp"

    print("Missing rows added:", df.isna().any(axis=1).sum())

    df = df.ffill()   # only past

    df = df.reset_index()

    print("After filling:", df.shape)

    return df

def prepare_data(df):
    """Prepare features and target using final engineered features (NO LEAKAGE)."""

    # Sort first
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Split FIRST (before any fill)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df  = df.iloc[split_index:].copy()

    # Fill separately (no cross-boundary leakage)
    train_df = train_df.ffill()
    test_df  = test_df.ffill()

    # Final Feature Groups
    base_pollutants = [
        "pm10", "pm2_5", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide", "ozone"
    ]

    weather_features = [
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "precipitation", "cloud_cover"
    ]

    time_features = [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos"
    ]

    lag_features = [
        "aqi_lag_1", "aqi_lag_3",
        "aqi_lag_6", "aqi_lag_24"
    ]

    rolling_features = [
        "aqi_roll_mean_6", "aqi_roll_std_6",
        "aqi_roll_mean_24", "aqi_roll_std_24"
    ]

    change_features = ["aqi_change_1h"]

    interaction_features = [
        "temp_humidity",
        "wind_x", "wind_y"
    ]

    features = (
        base_pollutants +
        weather_features +
        time_features +
        lag_features +
        rolling_features +
        change_features +
        interaction_features
    )

    features = [col for col in features if col in df.columns]

    # Split features/target AFTER safe processing
    X_train = train_df[features]
    y_train = train_df["aqi"]

    X_test = test_df[features]
    y_test = test_df["aqi"]

    print(f"Total Features Used: {len(features)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    return X_train, X_test, y_train, y_test, features

# ===============================
# Train Models (Fixed Parameters)
# ===============================

def train_models(X_train, y_train):

    print("Training models...\n")

    rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=4,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1
    )

    lgb_model.fit(X_train, y_train)

    return {
        "random_forest": rf,
        "xgboost": xgb_model,
        "lightgbm": lgb_model
    }



def evaluate_models(models, X_test, y_test):
    """Evaluate all models."""
    
    print("Evaluating models...\n")
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4)
        }
        
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE:  {mae:.2f}\n")
    
    return results


def select_best_model(results):
    """Select model with highest R²."""
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    return best_model[0]


def save_best_model_to_mongodb(models, best_model_name):
    """Save the best model directly to MongoDB GridFS."""

    print("Saving best model to MongoDB...")

    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")

    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in .env file")
        sys.exit(1)

    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]
        fs = GridFS(db)

        # Serialize model
        model_bytes = pickle.dumps(models[best_model_name])

        # Save to GridFS
        file_id = fs.put(
            model_bytes,
            filename=f"best_model_{best_model_name}.pkl",
            model_name=best_model_name,
            saved_at=datetime.utcnow()
        )

        print(f"Model saved to MongoDB with file_id: {file_id}\n")
        client.close()

        return file_id

    except Exception as e:
        print(f"ERROR saving model to MongoDB: {e}")
        sys.exit(1)


def register_model_to_mongodb(results, best_model_name, file_id):
    """Save model metadata to MongoDB and maintain training history."""

    print("Registering model to MongoDB...")

    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")

    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in .env file")
        sys.exit(1)

    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]
        collection = db["final_model_registry"]

        # Create model registry document
        registry = {
            "version": "v1.0",
            "trained_date": datetime.utcnow().isoformat(),
            "models": results,
            "best_model": best_model_name,
            "model_gridfs_id": str(file_id),
            "is_latest": True,
            "is_baseline": False
        }

        # Mark all existing records as not latest
        collection.update_many({}, {"$set": {"is_latest": False}})

        # Insert new registry record
        collection.insert_one(registry)

        # Keep baseline + last 5 daily runs
        all_records = list(collection.find().sort("trained_date", -1))

        baseline_models = [r for r in all_records if r.get("is_baseline", False)]
        daily_models = [r for r in all_records if not r.get("is_baseline", False)]

        if len(daily_models) > 5:
            ids_to_delete = [rec["_id"] for rec in daily_models[5:]]
            collection.delete_many({"_id": {"$in": ids_to_delete}})
            print(f"Kept last 5 daily runs, deleted {len(ids_to_delete)} older records")

        print(f"Registered: {best_model_name} (R² = {results[best_model_name]['r2']})")
        print(f"Total models: {len(baseline_models)} baseline + {min(len(daily_models), 5)} daily\n")

        client.close()

    except Exception as e:
        print(f"ERROR registering model: {e}")
        sys.exit(1)


def main():
    """Main training pipeline."""
    
    print("Daily Model Training Pipeline\n")
    
    # Load features from MongoDB
    df = load_features_from_mongodb()
    
    #df = clean_and_fill_data(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    # Prepare data
    X_train, X_test, y_train, y_test, features = prepare_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Select best model
    best_model = select_best_model(results)
    print(f"Best Model: {best_model.replace('_', ' ').title()} (R² = {results[best_model]['r2']})\n")
    
    file_id = save_best_model_to_mongodb(models, best_model)
    
    register_model_to_mongodb(results, best_model, file_id)

    
    print("Training complete!")


if __name__ == "__main__":
    main()