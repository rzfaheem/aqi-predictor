"""
Configuration file for AQI Predictor Project
============================================

IMPORTANT: Copy this file to config.py and add your actual API keys!

This file contains all the configuration settings like API keys and database connection.

HOW TO USE:
1. Copy this file: cp config.example.py config.py
2. Replace "YOUR_API_KEY_HERE" with your actual OpenWeather API key
3. Replace "YOUR_MONGODB_CONNECTION_STRING" with your MongoDB Atlas connection string
"""

import os

# ============================================================
# OPENWEATHER API CONFIGURATION
# ============================================================
# Get your free API key from: https://openweathermap.org/api

# Try to get from environment variable first (for GitHub Actions)
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")

# ============================================================
# CITY CONFIGURATION
# ============================================================
CITY_NAME = "Faisalabad"
CITY_COUNTRY = "Pakistan"
CITY_LAT = 31.4504
CITY_LON = 73.1350

# ============================================================
# MONGODB CONFIGURATION  
# ============================================================
# Try to get from environment variable first (for GitHub Actions)
MONGODB_CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING", "YOUR_MONGODB_CONNECTION_STRING")
MONGODB_DATABASE_NAME = "aqi_predictor"

# Collection names
COLLECTION_RAW_DATA = "raw_weather_data"
COLLECTION_FEATURES = "features"
COLLECTION_MODELS = "model_registry"

# ============================================================
# AQI CATEGORIES
# ============================================================
AQI_CATEGORIES = {
    (0, 50): {"level": "Good", "color": "green", "health": "Air quality is satisfactory"},
    (51, 100): {"level": "Moderate", "color": "yellow", "health": "Acceptable for most"},
    (101, 150): {"level": "Unhealthy for Sensitive Groups", "color": "orange", "health": "Sensitive groups may experience effects"},
    (151, 200): {"level": "Unhealthy", "color": "red", "health": "Everyone may experience health effects"},
    (201, 300): {"level": "Very Unhealthy", "color": "purple", "health": "Health alert, everyone may experience serious effects"},
    (301, 500): {"level": "Hazardous", "color": "maroon", "health": "Emergency conditions, entire population affected"}
}
