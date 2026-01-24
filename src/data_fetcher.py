"""
Data Fetcher Module - Fetch Weather & Air Quality Data from OpenWeather API
===========================================================================

WHAT THIS FILE DOES:
- Connects to OpenWeather API (a free weather/pollution data service)
- Fetches current weather data (temperature, humidity, wind, etc.)
- Fetches air pollution data (AQI and pollutant levels)
- Fetches historical air pollution data (for training our model)

WHAT IS AN API?
- API = Application Programming Interface
- It's like a waiter in a restaurant:
  - You (the customer) make a request
  - The waiter (API) goes to the kitchen (server)
  - The kitchen prepares your order (data)
  - The waiter brings it back to you
- We send a request to OpenWeather, and it sends back weather data

OPENWEATHER API ENDPOINTS WE USE:
1. Current Weather: Gets temperature, humidity, wind speed, etc.
2. Air Pollution: Gets current AQI and pollutant levels (PM2.5, PM10, NO2, etc.)
3. Air Pollution History: Gets past pollution data for training
"""

import requests
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataFetcher:
    """
    A class to fetch weather and air quality data from OpenWeather API.
    
    HOW TO USE:
        fetcher = DataFetcher()
        current_data = fetcher.fetch_current_data()
        print(current_data)
    """
    
    def __init__(self):
        """
        Initialize the DataFetcher with API settings from config.
        """
        self.api_key = config.OPENWEATHER_API_KEY
        self.lat = config.CITY_LAT
        self.lon = config.CITY_LON
        self.city = config.CITY_NAME
        
        # Base URLs for different API endpoints
        self.weather_url = "https://api.openweathermap.org/data/2.5/weather"
        self.pollution_url = "https://api.openweathermap.org/data/2.5/air_pollution"
        self.pollution_history_url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        
        print(f"📍 DataFetcher initialized for {self.city}")
        print(f"   Coordinates: ({self.lat}, {self.lon})")
    
    def fetch_current_weather(self) -> dict:
        """
        Fetch current weather data for our city.
        
        Returns:
            dict: Weather data including temperature, humidity, wind, etc.
        
        WHAT WE GET:
        - Temperature (in Celsius)
        - Humidity (percentage)
        - Pressure (atmospheric pressure)
        - Wind speed
        - Weather description (sunny, cloudy, etc.)
        """
        # Build the API request URL with our parameters
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key,
            "units": "metric"  # Use Celsius instead of Kelvin
        }
        
        print(f"🌤️ Fetching current weather...")
        
        try:
            # Make the API request
            response = requests.get(self.weather_url, params=params)
            
            # Check if request was successful (status code 200 = OK)
            if response.status_code == 200:
                data = response.json()  # Convert JSON response to Python dictionary
                
                # Extract the data we need
                weather_data = {
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "wind_speed": data["wind"]["speed"],
                    "wind_deg": data["wind"].get("deg", 0),  # .get() returns 0 if "deg" doesn't exist
                    "clouds": data["clouds"]["all"],
                    "weather_main": data["weather"][0]["main"],
                    "weather_description": data["weather"][0]["description"],
                    "timestamp": datetime.utcnow()
                }
                
                print(f"✅ Weather data fetched successfully!")
                print(f"   Temperature: {weather_data['temperature']}°C")
                print(f"   Humidity: {weather_data['humidity']}%")
                
                return weather_data
            else:
                print(f"❌ API request failed with status code: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching weather: {e}")
            return None
    
    def fetch_current_pollution(self) -> dict:
        """
        Fetch current air pollution data for our city.
        
        Returns:
            dict: Pollution data including AQI and pollutant levels
        
        WHAT WE GET:
        - AQI (Air Quality Index): 1-5 scale from OpenWeather
        - PM2.5: Fine particles (very harmful, can enter lungs)
        - PM10: Larger particles (dust, pollen)
        - NO2: Nitrogen dioxide (from cars and factories)
        - O3: Ozone (can cause breathing problems)
        - CO: Carbon monoxide
        - SO2: Sulfur dioxide
        
        NOTE ON AQI:
        OpenWeather uses a 1-5 scale:
        - 1 = Good
        - 2 = Fair
        - 3 = Moderate
        - 4 = Poor
        - 5 = Very Poor
        
        We'll convert this to the more common 0-500 scale later.
        """
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key
        }
        
        print(f"💨 Fetching current air pollution...")
        
        try:
            response = requests.get(self.pollution_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # The pollution data is in a "list" array, we take the first item
                pollution_info = data["list"][0]
                
                pollution_data = {
                    "aqi": pollution_info["main"]["aqi"],  # 1-5 scale
                    "pm2_5": pollution_info["components"]["pm2_5"],
                    "pm10": pollution_info["components"]["pm10"],
                    "no2": pollution_info["components"]["no2"],
                    "o3": pollution_info["components"]["o3"],
                    "co": pollution_info["components"]["co"],
                    "so2": pollution_info["components"]["so2"],
                    "nh3": pollution_info["components"]["nh3"],
                    "timestamp": datetime.utcnow()
                }
                
                # Convert AQI from 1-5 scale to approximate 0-500 scale
                # This is a rough approximation for display purposes
                aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
                pollution_data["aqi_standard"] = aqi_mapping.get(pollution_data["aqi"], 100)
                
                print(f"✅ Pollution data fetched successfully!")
                print(f"   AQI (1-5 scale): {pollution_data['aqi']}")
                print(f"   PM2.5: {pollution_data['pm2_5']} μg/m³")
                print(f"   PM10: {pollution_data['pm10']} μg/m³")
                
                return pollution_data
            else:
                print(f"❌ API request failed with status code: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching pollution: {e}")
            return None
    
    def fetch_current_data(self) -> dict:
        """
        Fetch both weather and pollution data and combine them.
        
        This is the main function you'll use most often!
        
        Returns:
            dict: Combined weather and pollution data
        """
        print("=" * 50)
        print(f"📊 Fetching all current data for {self.city}")
        print("=" * 50)
        
        # Fetch weather
        weather = self.fetch_current_weather()
        
        # Fetch pollution
        pollution = self.fetch_current_pollution()
        
        if weather is None or pollution is None:
            print("❌ Failed to fetch complete data")
            return None
        
        # Combine both into one dictionary
        combined_data = {
            "city": self.city,
            "timestamp": datetime.utcnow(),
            "weather": weather,
            "pollution": pollution
        }
        
        print("=" * 50)
        print("✅ All data fetched successfully!")
        print("=" * 50)
        
        return combined_data
    
    def fetch_historical_pollution(self, days_back: int = 30) -> list:
        """
        Fetch historical pollution data for the past N days.
        This is used to get training data for our model.
        
        Parameters:
            days_back (int): Number of days of historical data to fetch
        
        Returns:
            list: List of pollution data points
        
        NOTE:
        - OpenWeather's historical API requires Unix timestamps
        - Unix timestamp = seconds since January 1, 1970
        - We'll fetch data for each hour (24 data points per day)
        """
        print("=" * 50)
        print(f"📜 Fetching historical data for past {days_back} days...")
        print("=" * 50)
        
        # Calculate start and end timestamps
        end_datetime = datetime.utcnow()
        start_datetime = end_datetime - timedelta(days=days_back)
        
        # Convert to Unix timestamps (seconds since 1970)
        start_unix = int(start_datetime.timestamp())
        end_unix = int(end_datetime.timestamp())
        
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "start": start_unix,
            "end": end_unix,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(self.pollution_history_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                historical_data = []
                for item in data["list"]:
                    # Convert Unix timestamp to datetime
                    timestamp = datetime.utcfromtimestamp(item["dt"])
                    
                    data_point = {
                        "timestamp": timestamp,
                        "aqi": item["main"]["aqi"],
                        "pm2_5": item["components"]["pm2_5"],
                        "pm10": item["components"]["pm10"],
                        "no2": item["components"]["no2"],
                        "o3": item["components"]["o3"],
                        "co": item["components"]["co"],
                        "so2": item["components"]["so2"],
                        "nh3": item["components"]["nh3"]
                    }
                    
                    # Add standard AQI scale
                    aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
                    data_point["aqi_standard"] = aqi_mapping.get(data_point["aqi"], 100)
                    
                    historical_data.append(data_point)
                
                print(f"✅ Fetched {len(historical_data)} historical data points!")
                print(f"   Date range: {start_datetime.date()} to {end_datetime.date()}")
                
                return historical_data
            else:
                print(f"❌ API request failed with status code: {response.status_code}")
                print(f"   Response: {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return []
    
    def test_api_connection(self) -> bool:
        """
        Test if the API connection is working with the provided API key.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        print("🔍 Testing API connection...")
        
        # Try a simple weather request
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(self.weather_url, params=params)
            
            if response.status_code == 200:
                print("✅ API connection successful!")
                return True
            elif response.status_code == 401:
                print("❌ Invalid API key! Please check your config.py")
                return False
            else:
                print(f"❌ API returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False


# ========================================
# QUICK TEST (runs when you execute this file directly)
# ========================================

if __name__ == "__main__":
    """
    Fetch current data and SAVE it to MongoDB.
    This runs as part of the hourly Feature Pipeline.
    """
    print("\n" + "=" * 60)
    print("🕐 HOURLY DATA FETCHER")
    print("=" * 60 + "\n")
    
    # Import database module
    from src.database import Database
    
    # Create fetcher instance
    fetcher = DataFetcher()
    
    # Test API connection
    print("\n1. Testing API Connection...")
    if not fetcher.test_api_connection():
        print("\n⚠️ API connection failed. Please check:")
        print("   1. Your API key in config.py")
        print("   2. Your internet connection")
        exit(1)
    
    # Fetch current data
    print("\n2. Fetching Current Data...")
    current_data = fetcher.fetch_current_data()
    
    if current_data:
        print("\n📋 Summary of fetched data:")
        print(f"   City: {current_data['city']}")
        print(f"   Temperature: {current_data['weather']['temperature']}°C")
        print(f"   Humidity: {current_data['weather']['humidity']}%")
        print(f"   Wind Speed: {current_data['weather']['wind_speed']} m/s")
        print(f"   AQI: {current_data['pollution']['aqi_standard']}")
        print(f"   PM2.5: {current_data['pollution']['pm2_5']} μg/m³")
        
        # SAVE TO MONGODB
        print("\n3. Saving to MongoDB...")
        db = Database()
        
        # Format data for MongoDB (flatten the structure)
        record = {
            "city": current_data['city'],
            "timestamp": current_data['timestamp'],
            # Weather data
            "temperature": current_data['weather']['temperature'],
            "humidity": current_data['weather']['humidity'],
            "pressure": current_data['weather']['pressure'],
            "wind_speed": current_data['weather']['wind_speed'],
            # Pollution data
            "aqi": current_data['pollution']['aqi'],
            "aqi_standard": current_data['pollution']['aqi_standard'],
            "pm2_5": current_data['pollution']['pm2_5'],
            "pm10": current_data['pollution']['pm10'],
            "no2": current_data['pollution']['no2'],
            "o3": current_data['pollution']['o3'],
            "co": current_data['pollution']['co'],
            "so2": current_data['pollution']['so2']
        }
        
        # Save to raw_weather_data collection
        result = db.save_raw_data(record)
        print(f"✅ Data saved to MongoDB with ID: {result}")
        
        # Count total records
        count = db.raw_data.count_documents({})
        print(f"📊 Total records in raw_weather_data: {count}")
        
        print("\n🎉 Data fetch and save complete!")
    else:
        print("❌ Failed to fetch data")
        exit(1)

