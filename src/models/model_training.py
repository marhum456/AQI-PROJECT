"""
Training Random Forest, XGBoost, and LightGBM models for AQI prediction
(Fetch features from engineered_data2 and save models to MongoDB using GridFS)
"""

import os
import pickle
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from gridfs import GridFS
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb


# =========================
# MongoDB Setup
# =========================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["aqi_project"]

# ✅ Updated collections
data_collection = db["engineered_data_final"]
model_registers = db["model_registers"]

# ✅ Initialize GridFS
fs = GridFS(db)


# =========================
# Load Data from MongoDB
# =========================
def load_data():
    print("Fetching engineered_data from MongoDB...")
    
    cursor = data_collection.find({})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        raise ValueError("No data found in engineered_data collection.")
    
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


# =========================
# Prepare features & target
# =========================
def prepare_data(df):

    target_col = "aqi"

    exclude_cols = [
        "timestamp", "city", "latitude", "longitude", target_col
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].dtype != "object"
    ]

    print(f"Using {len(feature_cols)} features")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[target_col]

    data_clean = pd.concat([X, y], axis=1).dropna()
    X = data_clean[feature_cols]
    y = data_clean[target_col]

    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_cols


# =========================
# Train models
# =========================
def train_models(X_train, y_train):

    print("\nTraining models...")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=8,              # Reduced depth
            min_samples_split=15,     # Stronger split control
            min_samples_leaf=8,       # Prevent tiny leaf nodes
            max_features="sqrt",      # Reduce variance
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=120,        # Reduced
            max_depth=3,             # Very shallow
            learning_rate=0.02,      # Slower learning
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=3.0,           # Strong L1
            reg_lambda=5.0,          # Strong L2
            gamma=1.0,               # Split penalty
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
        n_estimators=120,
        max_depth=3,
        num_leaves=7,            # Very small tree
        learning_rate=0.02,
        min_child_samples=30,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=3.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name}")

    return models


# =========================
# Evaluate models
# =========================
def evaluate_models(models, X_test, y_test):

    print("\nEvaluating models...")
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results[name] = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "R2": round(r2_score(y_test, y_pred), 4)
        }

        print(f"\n{name}")
        print(f"  R²   : {results[name]['R2']}")
        print(f"  RMSE : {results[name]['RMSE']}")
        print(f"  MAE  : {results[name]['MAE']}")
        print("-" * 40)

    return results


# =========================
# Select best model
# =========================
def select_best_model(results):
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    return best_model[0]


# =========================
# Save models to MongoDB (GridFS Version)
# =========================
def save_models_to_mongo(models, results, feature_cols):

    print("\nSaving models to MongoDB using GridFS...")

    # Clear previous metadata (NOT GridFS files)
    model_registers.delete_many({})

    for name, model in models.items():

        # Serialize model
        model_bytes = pickle.dumps(model)

        # Store in GridFS
        file_id = fs.put(
            model_bytes,
            filename=f"{name}.pkl"
        )

        # Save metadata in model_register
        doc = {
            "model_name": name,
            "gridfs_file_id": file_id,
            "metrics": results[name],
            "features": feature_cols,
            "created_at": datetime.utcnow()
        }

        model_registers.insert_one(doc)
        print(f"  ✓ {name} saved (GridFS ID: {file_id})")

    print("\nAll models saved successfully using GridFS!")


# =========================
# Main training pipeline
# =========================
def main():

    print("Training AQI Prediction Models (engineered_data2)")
    print("-" * 60)

    df = load_data()

    X_train, X_test, y_train, y_test, features = prepare_data(df)

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    best_model = select_best_model(results)

    print("\n" + "=" * 60)
    print(f"🏆 Best Model: {best_model}")
    print(f"R² Score: {results[best_model]['R2']}")
    print("=" * 60)

    save_models_to_mongo(models, results, features)

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()