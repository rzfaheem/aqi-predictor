"""
Configuration file for AQI Predictor Project
============================================

This file contains all the configuration settings like API keys and database connection.

HOW TO USE:
1. Replace "YOUR_API_KEY_HERE" with your actual OpenWeather API key
2. Replace "YOUR_MONGODB_CONNECTION_STRING" with your MongoDB Atlas connection string
"""

# ============================================================
# OPENWEATHER API CONFIGURATION
# ============================================================
# Get your free API key from: https://openweathermap.org/api
# After signing up, go to: https://home.openweathermap.org/api_keys

OPENWEATHER_API_KEY = "d82cd440132449f5beeaea4e50915c99"

# ============================================================
# CITY CONFIGURATION
# ============================================================
# We're predicting AQI for Faisalabad, Pakistan
# Coordinates are needed for the Air Pollution API

CITY_NAME = "Faisalabad"
CITY_COUNTRY = "Pakistan"
CITY_LAT = 31.4504    # Latitude of Faisalabad
CITY_LON = 73.1350    # Longitude of Faisalabad

# ============================================================
# MONGODB CONFIGURATION  
# ============================================================
# Get your connection string from MongoDB Atlas:
# 1. Go to your cluster
# 2. Click "Connect" 
# 3. Choose "Connect your application"
# 4. Copy the connection string
# 5. Replace <password> with your actual password

MONGODB_CONNECTION_STRING = "mongodb+srv://razafaheem001_db_user:AuzzM7dnOqCGHbJA@cluster0.cpcenwx.mongodb.net/?appName=Cluster0"
MONGODB_DATABASE_NAME = "aqi_predictor"

# Collection names (like tables in SQL database)
COLLECTION_RAW_DATA = "raw_weather_data"        # Stores raw API data
COLLECTION_FEATURES = "features"                 # Stores processed features
COLLECTION_MODELS = "model_registry"             # Stores trained models

# ============================================================
# AQI CATEGORIES (for alerts and display)
# ============================================================
# These are standard AQI categories used worldwide

AQI_CATEGORIES = {
    (0, 50): {"level": "Good", "color": "green", "health": "Air quality is satisfactory"},
    (51, 100): {"level": "Moderate", "color": "yellow", "health": "Acceptable for most"},
    (101, 150): {"level": "Unhealthy for Sensitive Groups", "color": "orange", "health": "Sensitive groups may experience effects"},
    (151, 200): {"level": "Unhealthy", "color": "red", "health": "Everyone may experience health effects"},
    (201, 300): {"level": "Very Unhealthy", "color": "purple", "health": "Health alert, everyone may experience serious effects"},
    (301, 500): {"level": "Hazardous", "color": "maroon", "health": "Emergency conditions, entire population affected"}
}
