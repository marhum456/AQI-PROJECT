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




import os


from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb


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


def prepare_data(df):
    """Prepare features and target using final engineered features."""
    
    # Final Feature Groups (same as select_final_features)
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

    # Combine all feature columns
    features = (
        base_pollutants +
        weather_features +
        time_features +
        lag_features +
        rolling_features +
        change_features +
        interaction_features
    )

    # Keep only available columns (safe guard)
    features = [col for col in features if col in df.columns]

    X = df[features]
    y = df["aqi"]

    # Time-series split (no shuffle!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=False
    )

    print(f"Total Features Used: {len(features)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    return X_train, X_test, y_train, y_test, features

def train_models(X_train, y_train):
    """Train all 3 models."""
    
    print("Training models...\n")
    
    # Random Forest
    print("[1/3] Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=150, max_depth=8, min_samples_split=15,
        min_samples_leaf=8, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # XGBoost
    print("[2/3] XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.6,reg_alpha=3.0,reg_lambda=5.0,gamma=1.0, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # LightGBM
    print("[3/3] LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.02,min_child_samples=30,
        num_leaves=7, subsample=0.7,colsample_bytree=0.6,reg_alpha=3.0,reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    print()
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


def save_best_model(models, best_model_name):
    """Save only the best model to disk."""
    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(exist_ok=True)
    
    filename = f"best_model_{best_model_name}.pkl"
    joblib.dump(models[best_model_name], models_dir / filename)
    
    print(f"Best model saved: {filename}\n")


def register_model_to_mongodb(results, best_model_name):
    """Save model metadata to MongoDB and maintain training history."""
    
    print("Registering model to MongoDB...")

    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")

    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in .env file")
        sys.exit(1)

    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)

        # Use default database from URI
        db = client["aqi_project"]

        # Use your collection
        collection = db["final_model_registry"]

        # Create model registry document
        registry = {
            "version": "v1.0",
            "trained_date": datetime.now().isoformat(),
            "models": results,
            "best_model": best_model_name,
            "best_model_path": f"models/best_model_{best_model_name}.pkl",
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
    
    # Prepare data
    X_train, X_test, y_train, y_test, features = prepare_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Select best model
    best_model = select_best_model(results)
    print(f"Best Model: {best_model.replace('_', ' ').title()} (R² = {results[best_model]['r2']})\n")
    
    # Save only the best model to disk
    save_best_model(models, best_model)
    
    # Register to MongoDB
    register_model_to_mongodb(results, best_model)
    
    print("Training complete!")


if __name__ == "__main__":
    main()