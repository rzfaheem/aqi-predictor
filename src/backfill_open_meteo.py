"""
Historical Data Backfill via Open-Meteo API.

Fetches 2 years of hourly air quality AND weather data for Faisalabad
from the free Open-Meteo API (no API key required).
"""

import sys
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import Database


OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_air_quality(lat, lon, start_date, end_date):
    """Fetch hourly air quality data from Open-Meteo."""
    print(f"Fetching air quality data: {start_date} to {end_date}...")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }
    
    response = requests.get(OPEN_METEO_AQ_URL, params=params, timeout=60)
    
    if response.status_code != 200:
        print(f"Air quality API failed: {response.status_code}")
        print(response.text[:500])
        return None
    
    data = response.json()
    hourly = data.get("hourly", {})
    
    if not hourly or not hourly.get("time"):
        print("No air quality data returned!")
        return None
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "pm2_5": hourly.get("pm2_5"),
        "pm10": hourly.get("pm10"),
        "no2": hourly.get("nitrogen_dioxide"),
        "o3": hourly.get("ozone"),
        "co": hourly.get("carbon_monoxide"),
        "so2": hourly.get("sulphur_dioxide"),
    })
    
    print(f"   Got {len(df)} air quality records")
    return df


def fetch_weather(lat, lon, start_date, end_date):
    """Fetch hourly weather data from Open-Meteo Historical Weather API."""
    print(f"Fetching weather data: {start_date} to {end_date}...")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }
    
    response = requests.get(OPEN_METEO_WEATHER_URL, params=params, timeout=60)
    
    if response.status_code != 200:
        print(f"Weather API failed: {response.status_code}")
        print(response.text[:500])
        return None
    
    data = response.json()
    hourly = data.get("hourly", {})
    
    if not hourly or not hourly.get("time"):
        print("No weather data returned!")
        return None
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly.get("temperature_2m"),
        "humidity": hourly.get("relative_humidity_2m"),
        "pressure": hourly.get("surface_pressure"),
        "wind_speed": hourly.get("wind_speed_10m"),
    })
    
    print(f"   Got {len(df)} weather records")
    return df


def compute_aqi_from_pm25(pm25):
    """Convert PM2.5 concentration to AQI (US EPA standard)."""
    if pm25 is None or pd.isna(pm25):
        return None
    
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return round(aqi)
    
    if pm25 > 500.4:
        return 500
    return 0


def backfill_from_open_meteo(years_back=2):
    """
    Fetch historical data from Open-Meteo and store in MongoDB.
    
    Args:
        years_back: Number of years of historical data to fetch
    """
    print("=" * 60)
    print("HISTORICAL BACKFILL VIA OPEN-METEO")
    print("=" * 60)
    
    lat = config.CITY_LAT
    lon = config.CITY_LON
    
    end_date = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    
    print(f"Location: {config.CITY_NAME} ({lat}, {lon})")
    print(f"Period: {start_date} to {end_date}")
    print(f"Expected: ~{365 * years_back * 24} hourly records")
    
    # Fetch in chunks (Open-Meteo can handle large ranges, but let's be safe)
    chunk_days = 90
    all_aq_dfs = []
    all_weather_dfs = []
    
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_start < final_end:
        chunk_end = min(current_start + timedelta(days=chunk_days), final_end)
        s = current_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        
        aq_df = fetch_air_quality(lat, lon, s, e)
        if aq_df is not None:
            all_aq_dfs.append(aq_df)
        
        weather_df = fetch_weather(lat, lon, s, e)
        if weather_df is not None:
            all_weather_dfs.append(weather_df)
        
        current_start = chunk_end + timedelta(days=1)
    
    if not all_aq_dfs:
        print("No air quality data fetched!")
        return 0
    
    # Combine chunks
    aq_combined = pd.concat(all_aq_dfs, ignore_index=True)
    print(f"\nTotal air quality records: {len(aq_combined)}")
    
    if all_weather_dfs:
        weather_combined = pd.concat(all_weather_dfs, ignore_index=True)
        print(f"Total weather records: {len(weather_combined)}")
        
        # Merge on timestamp
        merged = pd.merge(aq_combined, weather_combined, on="timestamp", how="left")
    else:
        merged = aq_combined
    
    print(f"Merged records: {len(merged)}")
    
    # Compute AQI from PM2.5
    merged["aqi_standard"] = merged["pm2_5"].apply(compute_aqi_from_pm25)
    merged["aqi"] = merged["aqi_standard"].apply(
        lambda x: 1 if x and x <= 50 else (2 if x and x <= 100 else (3 if x and x <= 150 else (4 if x and x <= 200 else 5)))
    )
    
    # Add city info
    merged["city"] = config.CITY_NAME
    
    # Drop rows where PM2.5 is missing
    initial = len(merged)
    merged = merged.dropna(subset=["pm2_5"])
    print(f"After dropping missing PM2.5: {len(merged)} (removed {initial - len(merged)})")
    
    # Save to MongoDB
    print("\nSaving to MongoDB...")
    db = Database()
    
    # Clear existing raw data (replace with comprehensive historical data)
    existing = db.raw_data.count_documents({})
    print(f"Existing records in DB: {existing}")
    
    db.raw_data.delete_many({})
    print("Cleared existing data")
    
    records = merged.to_dict("records")
    
    batch_size = 500
    saved = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        db.raw_data.insert_many(batch)
        saved += len(batch)
        if saved % 2000 == 0:
            print(f"   Saved {saved}/{len(records)} records...")
    
    print(f"\nTotal saved: {saved} records")
    
    # Verify
    final_count = db.raw_data.count_documents({})
    print(f"Database now has: {final_count} records")
    
    # Print summary stats
    print(f"\nData Summary:")
    print(f"   Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    print(f"   PM2.5 - Mean: {merged['pm2_5'].mean():.1f}, Min: {merged['pm2_5'].min():.1f}, Max: {merged['pm2_5'].max():.1f}")
    if 'temperature' in merged.columns:
        print(f"   Temperature - Mean: {merged['temperature'].mean():.1f}C")
    print(f"   AQI - Mean: {merged['aqi_standard'].mean():.1f}")
    
    return saved


if __name__ == "__main__":
    years = 2
    if len(sys.argv) > 1:
        try:
            years = int(sys.argv[1])
        except ValueError:
            years = 2
    
    count = backfill_from_open_meteo(years_back=years)
    
    if count > 0:
        print("\nBackfill complete! Now run:")
        print("   python src/feature_engineering.py")
        print("   python src/train_model.py")
    else:
        print("\nBackfill failed!")
