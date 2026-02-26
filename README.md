# Automated AQI Prediction System Gujrat, Pakistan – 72-Hour Forecast 
# Automated AQI Prediction System – Gujrat, Pakistan

**72-Hour Forecast & Environmental Intelligence**  

**Author:** Muhammad Arhum  
**Program:** 10P Shine Internship Program  
**Duration:** Dec 2025 – Feb 2026  

---

## Overview
This project implements an automated system to predict Air Quality Index (AQI) for Gujrat, Pakistan, up to 72 hours in advance. The system combines hourly weather and pollutant data collection, machine learning-based AQI forecasting, and a real-time interactive dashboard.

Key features:
- Hourly data ingestion and feature engineering  
- Predictive ML models: Random Forest, XGBoost, LightGBM  
- 72-hour AQI forecasts  
- Streamlit dashboard with WHO Dark Mode + Glass UI  
- CI/CD automation with GitHub Actions  

---

## Data Collection
- **APIs Used:** OpenMeteo (selected for free access and no DNS restrictions)  
- **Historical Data:** 90+ days  
- Hourly data includes pollutants (PM2.5, PM10, O3, NO2, SO2, CO) and weather features (temperature, humidity, wind, precipitation, cloud cover)  
- Data stored in **MongoDB** for reproducible feature engineering and model training  

---

## AQI Calculation
- Individual pollutant AQI calculated using EPA linear breakpoints  
- Overall AQI = maximum of all pollutant AQIs  
- AQI categorized as Good, Moderate, Unhealthy (and variants), Hazardous  

---

## Feature Engineering
- Total 29 features generated, including pollutants, weather, temporal, lag, rolling, momentum, and interaction features  

| Feature Category | Features |
|-----------------|----------|
| Pollutants (6) | pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone |
| Weather (5) | temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, cloud_cover |
| Time (6) | hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos |
| Lag (4) | aqi_lag_1, aqi_lag_3, aqi_lag_6, aqi_lag_24 |
| Rolling (4) | aqi_roll_mean_6, aqi_roll_std_6, aqi_roll_mean_24, aqi_roll_std_24 |
| Momentum (1) | aqi_change_1h |
| Interaction (3) | temp_humidity, wind_x, wind_y |

Cleaned datasets are saved back into MongoDB for modeling.

---

## Modeling
**Models Trained:** Random Forest, XGBoost, LightGBM  
**Best Model:** Random Forest  

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Random Forest | 0.775 | 11.65 | 23.32 |
| XGBoost | 0.698 | 17.20 | 27.03 |
| LightGBM | 0.718 | 16.86 | 26.11 |

- Random Forest chosen for production due to highest R² and lowest error metrics.  
- Models evaluated using R², RMSE, and MAE, and serialized with metadata in MongoDB.  

---

## Pipelines
**Hourly Pipeline**
- Collects, processes, and stores AQI/weather data hourly  
- Converts timestamps to PKT  
- Computes AQI and engineered features  
- Stores processed features in `engineered_data_final` collection  

**Daily Model Training & Registration**
- Retrains models daily using latest engineered features  
- Three models trained (RF, XGBoost, LightGBM)  
- Best model selected based on R²  
- Metadata stored in `final_model_registry`  

**72-Hour Prediction Pipeline**
- Loads latest best model from MongoDB  
- Sequentially generates 72-hour forecast features  
- Updates lag, rolling, and interaction features dynamically  
- Stores predicted AQI in MongoDB with timestamps and metadata  

**GitHub Actions & CI/CD**
- Hourly feature pipeline and daily model training automated  
- CI/CD deploys updated Streamlit dashboard to cloud  
- Monitoring, logging, and retries ensure production reliability  

---

## Web Dashboard
- Streamlit-based, responsive interface  
- WHO Dark Mode + Glass UI  
- Features:
  - Animated AQI gauge  
  - 72-hour forecast interactive line chart  
  - Model evaluation panel (R² comparison, selected model)  
  - Automated cloud-ready updates  

---
## Web Dashboard Link:
https://aqi-project-gujrat.streamlit.app/

---

## Contact:
muhammadarhum277@gmail.com  
