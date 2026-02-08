"""
Configuration file for AQI Predictor Project
============================================

This file reads API keys from environment variables for security.

LOCAL SETUP:
1. Create a .env file in the project root with:
   OPENWEATHER_API_KEY=your_api_key
   MONGODB_CONNECTION_STRING=your_connection_string

GITHUB ACTIONS:
   Secrets are configured in GitHub Settings → Secrets → Actions
"""

import os

# Try to load from .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# ============================================================
# OPENWEATHER API CONFIGURATION
# ============================================================
# Get your free API key from: https://openweathermap.org/api

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# ============================================================
# CITY CONFIGURATION
# ============================================================
# We're predicting AQI for Faisalabad, Pakistan

CITY_NAME = "Faisalabad"
CITY_COUNTRY = "Pakistan"
CITY_LAT = 31.4504    # Latitude of Faisalabad
CITY_LON = 73.1350    # Longitude of Faisalabad

# ============================================================
# MONGODB CONFIGURATION  
# ============================================================
# Connection string from MongoDB Atlas

MONGODB_CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING", "")
MONGODB_DATABASE_NAME = "aqi_predictor"

# Collection names (like tables in SQL database)
COLLECTION_RAW_DATA = "raw_weather_data"        # Stores raw API data
COLLECTION_FEATURES = "features"                 # Stores processed features
COLLECTION_MODELS = "model_registry"             # Stores trained models

# ============================================================
# AQI CATEGORIES (for alerts and display)
# ============================================================

AQI_CATEGORIES = {
    (0, 50): {"level": "Good", "color": "green", "health": "Air quality is satisfactory"},
    (51, 100): {"level": "Moderate", "color": "yellow", "health": "Acceptable for most"},
    (101, 150): {"level": "Unhealthy for Sensitive Groups", "color": "orange", "health": "Sensitive groups may experience effects"},
    (151, 200): {"level": "Unhealthy", "color": "red", "health": "Everyone may experience health effects"},
    (201, 300): {"level": "Very Unhealthy", "color": "purple", "health": "Health alert, everyone may experience serious effects"},
    (301, 500): {"level": "Hazardous", "color": "maroon", "health": "Emergency conditions, entire population affected"}
}

