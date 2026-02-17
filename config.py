"""
Configuration for AQI Predictor
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Configuration
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# City Configuration
CITY_NAME = "Faisalabad"
CITY_COUNTRY = "Pakistan"
CITY_LAT = 31.4504
CITY_LON = 73.1350

# MongoDB Configuration
MONGODB_CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING", "")
MONGODB_DATABASE_NAME = "aqi_predictor"

COLLECTION_RAW_DATA = "raw_weather_data"
COLLECTION_FEATURES = "features"
COLLECTION_MODELS = "model_registry"

# Standard AQI categories
AQI_CATEGORIES = {
    (0, 50): {"level": "Good", "color": "green", "health": "Air quality is satisfactory"},
    (51, 100): {"level": "Moderate", "color": "yellow", "health": "Acceptable for most"},
    (101, 150): {"level": "Unhealthy for Sensitive Groups", "color": "orange", "health": "Sensitive groups may experience effects"},
    (151, 200): {"level": "Unhealthy", "color": "red", "health": "Everyone may experience health effects"},
    (201, 300): {"level": "Very Unhealthy", "color": "purple", "health": "Health alert, everyone may experience serious effects"},
    (301, 500): {"level": "Hazardous", "color": "maroon", "health": "Emergency conditions, entire population affected"}
}
