# AQI Prediction System — Faisalabad, Pakistan

A real-time Air Quality Index prediction system that forecasts PM2.5 concentrations for the next 3 days using machine learning, with an interactive Streamlit dashboard.

## Overview

This project collects hourly weather and pollution data from the OpenWeather API, engineers features from historical patterns, and trains a Multi-Output Random Forest model to predict air quality 24, 48, and 72 hours ahead. Predictions are converted to standard EPA AQI (0–500 scale) and displayed on a live dashboard with health recommendations.

## Features

- **Real-time data collection** — Hourly weather and pollution data via OpenWeather API
- **Automated pipelines** — GitHub Actions for hourly data collection and daily model retraining
- **Multi-Output prediction** — Single model predicts 24h, 48h, and 72h simultaneously
- **Interactive dashboard** — Streamlit app with current conditions, 3-day forecast, and health alerts
- **Cloud database** — MongoDB Atlas for data storage and model persistence

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Dashboard | Streamlit, Plotly |
| Database | MongoDB Atlas |
| API | OpenWeather API |
| CI/CD | GitHub Actions |

## Project Structure

```
├── app.py                      # Streamlit dashboard
├── config.py                   # Configuration (reads from env variables)
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_fetcher.py         # OpenWeather API data collection
│   ├── database.py             # MongoDB operations
│   ├── feature_engineering.py  # Feature creation pipeline
│   ├── train_model.py          # Model training and evaluation
│   ├── backfill_historical.py  # Historical data backfill
│   ├── shap_explainer.py       # SHAP model explainability
│   └── eda_analysis.py         # Exploratory data analysis
├── .github/workflows/
│   ├── feature_pipeline.yml    # Hourly data collection
│   └── training_pipeline.yml   # Daily model retraining
├── notebooks/
│   └── SHAP_Analysis.ipynb     # SHAP analysis notebook
└── models/                     # Trained model files (.pkl)
```

## How It Works

1. **Data Collection** — The feature pipeline runs every hour, fetching temperature, humidity, wind speed, pressure, and pollutant concentrations (PM2.5, PM10, NO2, O3, CO, SO2) from the OpenWeather API.

2. **Feature Engineering** — Raw data is transformed into 36 features including time features (hour, day, weekend), lag features (values from 1–24 hours ago), rolling averages (3h, 6h, 12h, 24h windows), and rate-of-change features.

3. **Model Training** — Three models (Ridge Regression, Random Forest, XGBoost) are compared using TimeSeriesSplit cross-validation. The model with the lowest RMSE is selected automatically.

4. **Prediction** — The model predicts PM2.5 concentration, which is converted to standard AQI using EPA breakpoint formulas.

5. **Dashboard** — Current conditions and the 3-day forecast are displayed on a Streamlit web app with color-coded severity levels and health recommendations.

## Setup

### Prerequisites

- Python 3.10+
- MongoDB Atlas account
- OpenWeather API key

### Installation

```bash
# Clone the repository
git clone https://github.com/rzfaheem/aqi-predictor.git
cd aqi-predictor

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENWEATHER_API_KEY=your_api_key_here
MONGODB_CONNECTION_STRING=your_mongodb_connection_string_here
```

### Run Locally

```bash
# Run the dashboard
streamlit run app.py

# Run data collection manually
python src/data_fetcher.py

# Run feature engineering
python src/feature_engineering.py

# Train the model
python src/train_model.py
```

## Deployment

The dashboard is deployed on **Streamlit Community Cloud** and updates automatically when changes are pushed to the main branch. API keys are configured as secrets in both Streamlit Cloud and GitHub Actions.

## Key Design Decisions

- **PM2.5 over AQI as target** — The OpenWeather API returns only 5 discrete AQI categories (1–5). With 97% of values being the same category in Faisalabad, the model couldn't learn. PM2.5 provides continuous values suitable for regression.
- **TimeSeriesSplit validation** — Prevents data leakage by always training on past data and testing on future data.
- **Multi-Output model** — One model predicts all three horizons together, leveraging correlations between them.

## License

This project was developed as part of an internship at 10Pearls.
