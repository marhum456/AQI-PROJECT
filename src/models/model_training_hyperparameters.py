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
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# =========================
# MongoDB Setup
# =========================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["aqi_project"]

# ✅ Updated collections
data_collection = db["engineered_data_final"]
model_registers = db["model_registers_hyperparameters"]

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

    df = df.interpolate(method="time")
    df = df.bfill().ffill()

    df = df.reset_index()

    print("After filling:", df.shape)

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

    print("Training models with hyperparameter tuning...\n")

    # ------------------- Random Forest -------------------
    print("[1/3] Random Forest")

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    rf_params = {
        "n_estimators":[100,200,300],
        "max_depth":[5,8,12,None],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,4,8]
    }

    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=10,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    rf_search.fit(X_train, y_train)

    print("\nBest Random Forest Parameters:")
    print(rf_search.best_params_)
    print("Best CV Score:", rf_search.best_score_)


    # ------------------- XGBoost -------------------
    print("\n[2/3] XGBoost")

    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    xgb_params = {
        "n_estimators":[100,150,200],
        "max_depth":[3,5,7],
        "learning_rate":[0.01,0.05,0.1],
        "subsample":[0.7,0.8,1.0],
        "colsample_bytree":[0.6,0.8,1.0]
    }

    xgb_search = RandomizedSearchCV(
        xgb_model,
        xgb_params,
        n_iter=10,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    xgb_search.fit(X_train, y_train)

    print("\nBest XGBoost Parameters:")
    print(xgb_search.best_params_)
    print("Best CV Score:", xgb_search.best_score_)


    # ------------------- LightGBM -------------------
    print("\n[3/3] LightGBM")

    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

    lgb_params = {
        "n_estimators":[100,200,300],
        "max_depth":[3,5,7,-1],
        "learning_rate":[0.01,0.05,0.1],
        "num_leaves":[7,15,31],
        "subsample":[0.7,0.8,1.0],
        "colsample_bytree":[0.6,0.8,1.0]
    }

    lgb_search = RandomizedSearchCV(
        lgb_model,
        lgb_params,
        n_iter=10,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    lgb_search.fit(X_train, y_train)

    print("\nBest LightGBM Parameters:")
    print(lgb_search.best_params_)
    print("Best CV Score:", lgb_search.best_score_)


  # Collect best models
    # Collect best models
    models = {
        "RandomForest": rf_search.best_estimator_,
        "XGBoost": xgb_search.best_estimator_,
        "LightGBM": lgb_search.best_estimator_
    }

    # Fit best models on training data
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name}")

    # ✅ Return dictionary of trained models
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

    df = clean_and_fill_data(df)


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