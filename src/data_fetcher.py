"""
Data Fetcher Module - Fetches weather and air pollution data from OpenWeather API.
"""

import requests
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataFetcher:
    """Handles all OpenWeather API interactions for weather and pollution data."""
    
    def __init__(self):
        self.api_key = config.OPENWEATHER_API_KEY
        self.lat = config.CITY_LAT
        self.lon = config.CITY_LON
        self.city = config.CITY_NAME
        
        self.weather_url = "https://api.openweathermap.org/data/2.5/weather"
        self.pollution_url = "https://api.openweathermap.org/data/2.5/air_pollution"
        self.pollution_history_url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        
        print(f"📍 DataFetcher initialized for {self.city}")
    
    def fetch_current_weather(self) -> dict:
        """Fetch current weather data (temperature, humidity, wind, etc.)."""
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(self.weather_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data = {
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "wind_speed": data["wind"]["speed"],
                    "wind_deg": data["wind"].get("deg", 0),
                    "clouds": data["clouds"]["all"],
                    "weather_main": data["weather"][0]["main"],
                    "weather_description": data["weather"][0]["description"],
                    "timestamp": datetime.utcnow()
                }
                
                print(f"✅ Weather: {weather_data['temperature']}°C, {weather_data['humidity']}% humidity")
                return weather_data
            else:
                print(f"❌ Weather API failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching weather: {e}")
            return None
    
    def fetch_current_pollution(self) -> dict:
        """Fetch current air pollution data (AQI, PM2.5, PM10, etc.)."""
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(self.pollution_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                pollution_info = data["list"][0]
                
                pollution_data = {
                    "aqi": pollution_info["main"]["aqi"],
                    "pm2_5": pollution_info["components"]["pm2_5"],
                    "pm10": pollution_info["components"]["pm10"],
                    "no2": pollution_info["components"]["no2"],
                    "o3": pollution_info["components"]["o3"],
                    "co": pollution_info["components"]["co"],
                    "so2": pollution_info["components"]["so2"],
                    "nh3": pollution_info["components"]["nh3"],
                    "timestamp": datetime.utcnow()
                }
                
                # Map OpenWeather's 1-5 AQI to approximate standard scale
                aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
                pollution_data["aqi_standard"] = aqi_mapping.get(pollution_data["aqi"], 100)
                
                print(f"✅ Pollution: AQI={pollution_data['aqi']}, PM2.5={pollution_data['pm2_5']} μg/m³")
                return pollution_data
            else:
                print(f"❌ Pollution API failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching pollution: {e}")
            return None
    
    def fetch_current_data(self) -> dict:
        """Fetch combined weather and pollution data."""
        print(f"\n📊 Fetching data for {self.city}...")
        
        weather = self.fetch_current_weather()
        pollution = self.fetch_current_pollution()
        
        if weather is None or pollution is None:
            print("❌ Failed to fetch complete data")
            return None
        
        combined_data = {
            "city": self.city,
            "timestamp": datetime.utcnow(),
            "weather": weather,
            "pollution": pollution
        }
        
        print("✅ All data fetched successfully!")
        return combined_data
    
    def fetch_historical_pollution(self, days_back: int = 30) -> list:
        """Fetch historical pollution data for model training."""
        print(f"📜 Fetching historical data for past {days_back} days...")
        
        end_datetime = datetime.utcnow()
        start_datetime = end_datetime - timedelta(days=days_back)
        
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
                    
                    aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
                    data_point["aqi_standard"] = aqi_mapping.get(data_point["aqi"], 100)
                    
                    historical_data.append(data_point)
                
                print(f"✅ Fetched {len(historical_data)} historical data points")
                return historical_data
            else:
                print(f"❌ Historical API failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return []
    
    def test_api_connection(self) -> bool:
        """Test API connectivity."""
        try:
            params = {"lat": self.lat, "lon": self.lon, "appid": self.api_key}
            response = requests.get(self.weather_url, params=params)
            
            if response.status_code == 200:
                print("✅ API connection successful!")
                return True
            elif response.status_code == 401:
                print("❌ Invalid API key!")
                return False
            else:
                print(f"❌ API returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False


if __name__ == "__main__":
    """Fetch current data and save to MongoDB (runs as part of hourly pipeline)."""
    from src.database import Database
    
    fetcher = DataFetcher()
    
    if not fetcher.test_api_connection():
        exit(1)
    
    current_data = fetcher.fetch_current_data()
    
    if current_data:
        db = Database()
        
        record = {
            "city": current_data['city'],
            "timestamp": current_data['timestamp'],
            "temperature": current_data['weather']['temperature'],
            "humidity": current_data['weather']['humidity'],
            "pressure": current_data['weather']['pressure'],
            "wind_speed": current_data['weather']['wind_speed'],
            "aqi": current_data['pollution']['aqi'],
            "aqi_standard": current_data['pollution']['aqi_standard'],
            "pm2_5": current_data['pollution']['pm2_5'],
            "pm10": current_data['pollution']['pm10'],
            "no2": current_data['pollution']['no2'],
            "o3": current_data['pollution']['o3'],
            "co": current_data['pollution']['co'],
            "so2": current_data['pollution']['so2']
        }
        
        result = db.save_raw_data(record)
        count = db.raw_data.count_documents({})
        print(f"📊 Total records: {count}")
    else:
        print("❌ Failed to fetch data")
        exit(1)
