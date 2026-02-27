### **Automated AQI Prediction System Gujrat, Pakistan – 72-Hour Forecast**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Technology Stack](#technology-stack) 
3. [Data Collection](#data-collection)  
4. [MongoDB as Feature Store](#mongodb-as-feature-store)  
5. [AQI Calculation](#aqi-calculation)  
6. [Feature Engineering](#feature-engineering)  
7. [Model Training](#model-training)  
8. [Feature Importance](#feature-importance)  
9. [Pipelines](#pipelines)  
10. [Web Application Dashboard](#web-application-dashboard)  
11. [Key Challenges and Solutions](#key-challenges-and-solutions)   
12. [How to Run](#how-to-run)  

---

## Introduction
Air pollution is a critical environmental issue in Pakistan. This project builds an automated machine learning system to predict AQI in Gujrat for up to 72 hours, helping individuals plan activities safely and enabling policymakers to act promptly.  

**Objectives:**  
- Predict AQI with R² ≥ 0.75.  
- Automate hourly data collection.  
- Establish daily model training and forecasting pipelines.  
- Deploy an interactive, cloud-ready dashboard.  

**Scope:**  
- Location: Gujrat, Pakistan (LAT=32.5731, LON=74.1005)  
- Historical data: 3 months  
- AQI: EPA standards  
- Features: 29 engineered features  
- Models: Random Forest, XGBoost, LightGBM  
- Deployment: Automated via GitHub Actions  
- Integration: MongoDB + Streamlit dashboard  

---

## Technology Stack
- **Python**: Core language  
- **Pandas / NumPy**: Data processing  
- **scikit-learn, XGBoost, LightGBM**: Machine learning  
- **SHAP**: Feature importance  
- **MongoDB**: Feature store & model registry  
- **Streamlit / Plotly**: Dashboard & visualization  
- **GitHub Actions / Airflow**: CI/CD automation

---


## Data Collection
| API         | Access                       | Historical Data | Selected |
|------------|-------------------------------|----------------|---------|
| AQICN      | DNS blocked, VPN needed       | 2 months       | No      |
| OpenWeather| DNS blocked, VPN needed       | 2 months       | No      |
| OpenMeteo  | No restrictions               | 90+ days       | Yes     |

**Reason for OpenMeteo:** Free, no API key, no DNS blocking, sufficient historical data.  

---

## MongoDB as Feature Store
Raw AQI and weather data is collected hourly via OpenMeteo and stored in MongoDB.  
- JSON responses are parsed, cleaned, and structured into tables.  
- Timestamps and location metadata are included for time-series alignment.  
- Ensures reproducible feature engineering, automated training, and consistent predictions.  

---

## AQI Calculation
**Single pollutant AQI:**  
The AQI for an individual pollutant is calculated using linear interpolation between breakpoint concentrations:

Formula:

AQI_pollutant = ((I_high − I_low) / (C_high − C_low)) × (C_measured − C_low) + I_low

Where:

C_measured = Measured concentration of the pollutant

C_low, C_high = Breakpoint concentration range in which the measured value falls

I_low, I_high = AQI index values corresponding to that breakpoint range

**Overall AQI:**  
The final AQI value is determined by taking the maximum AQI value among all considered pollutants:

AQI_overall = max(AQI_PM2.5, AQI_PM10, AQI_O3, AQI_NO2, AQI_SO2, AQI_CO)

This means the pollutant with the highest AQI value defines the overall air quality category.

**AQI Categories:**

| AQI Range | Category                       |
|-----------|--------------------------------|
| 0–50      | Good                           |
| 51–100    | Moderate                       |
| 101–150   | Unhealthy for Sensitive Groups |
| 151–200   | Unhealthy                      |
| 201–300   | Very Unhealthy                 |
| >300      | Hazardous                      |

---

## Feature Engineering
29 features are generated to capture temporal patterns and pollutant relationships:  

| Feature Category | Features |
|-----------------|----------|
| Pollutants (6)   | pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone |
| Weather (5)      | temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, cloud_cover |
| Time (6)         | hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos |
| Lag (4)          | aqi_lag_1, aqi_lag_3, aqi_lag_6, aqi_lag_24 |
| Rolling (4)      | aqi_roll_mean_6, aqi_roll_std_6, aqi_roll_mean_24, aqi_roll_std_24 |
| Momentum (1)     | aqi_change_1h |
| Interaction (3)  | temp_humidity, wind_x, wind_y |

Cleaned datasets are saved back to MongoDB for modeling.

---

## Model Training
**Process:**  
- Source: `engineered_data_final` in MongoDB  
- Models: Random Forest, XGBoost, LightGBM  
- Split: 80% training, 20% testing  
- Input: Pollutants, weather, time, lag, rolling, momentum, interaction features  
- Output: AQI  

**Hyperparameters:**  
- Random Forest: 150 trees, max depth 8, min samples per leaf 8  
- XGBoost: depth 3, learning rate 0.02, L1/L2 regularization  
- LightGBM: 120 iterations, num_leaves=7, learning rate 0.02, subsample 0.7  

**Model Comparison Results:**

| Model         | R² Score | MAE   | RMSE  |
|---------------|----------|-------|-------|
| Random Forest | 0.729    | 14.91 | 28.99 |
| XGBoost       | 0.686    | 19.79 | 31.2 |
| LightGBM      | 0.692   | 19.52 | 30.91 |

**Best Model:** Random Forest (highest R², lowest MAE & RMSE)  

---

## Feature Importance
**Top 10 features using SHAP:**

| Rank | Feature       | Mean Absolute SHAP |
|------|---------------|------------------|
| 1    | aqi_lag_1     | 10.24            |
| 2    | aqi_roll_mean_6 | 5.63           |
| 3    | ozone         | 5.11             |
| 4    | aqi_lag_24    | 4.93             |
| 5    | pm2_5         | 4.38             |
| 6    | aqi_change_1h | 4.04             |
| 7    | pm10          | 4.01             |
| 8    | sulphur_dioxide | 2.45           |
| 9    | aqi_roll_mean_24 | 2.28          |
| 10   | nitrogen_dioxide | 2.00          |

**Key Insights:**  
- Recent AQI dominates predictions.  
- Primary pollutants (ozone, pm2.5, pm10, SO2, NO2) are critical.  
- Momentum and rolling features improve short-term predictions.  

---

## Pipelines
**Hourly Pipeline:**  
- Collects, processes, and stores AQI/weather data hourly  
- Converts timestamps to PKT  
- Computes AQI and engineered features  
- Stores in `engineered_data_final`  

**Daily Model Training & Registration:**  
- Retrains models daily (RF, XGBoost, LightGBM)  
- Best model selected by R²  
- Metadata stored in `final_model_registry`  

**72-Hour Prediction Pipeline:**  
- Loads latest best model  
- Generates sequential 72-hour forecast features  
- Updates lag, rolling, and interaction features dynamically  
- Stores predictions in MongoDB  

**GitHub Actions / CI-CD:**  
- Hourly feature pipeline & daily training automated  
- Dependencies installed, pipelines run, Streamlit dashboard deployed  
- Logging, retries, and monitoring for reliability  

---

## Web Application Dashboard
- Streamlit-based interface  
- WHO Dark Mode + Glass UI  
- Features:  
  - Animated AQI gauge  
  - 72-hour forecast line chart  
  - Model evaluation panel  
- Cloud-ready and responsive  

---

## View Real-Time Deployed Dashboard on:
https://aqi-project-gujrat.streamlit.app/

---
  

## How to Run the Project


1. **Clone the repository:**
    ```bash
    git clone https://github.com/marhum456/AQI-PROJECT.git
    cd AQI-PROJECT
    ```

2. **Setup Virtual Environment & Install Dependencies:**
    ```bash
        python -m venv venv

        source venv/bin/activate   # Linux/macOS
        venv\Scripts\activate      # Windows
        pip install -r requirements.txt
    ```

3. **Create .env in project root:**

    ```bash
    MONGO_URI=<your_mongodb_uri>
    CITY_LATITUDE=32.5731
    CITY_LONGITUDE=74.1005
    ```
4. **Create .env in project root:**

    ```bash
    MONGO_URI=<your_mongodb_uri>
    CITY_LATITUDE=32.5731
    CITY_LONGITUDE=74.1005
    ```
5. **run streamlit dashboard:**
    ```bash
    streamlit run app.py
    ```
    Open browser at: http://localhost:8501

---
   


