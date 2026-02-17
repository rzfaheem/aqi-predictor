"""
Historical Data Backfill - Fetches past pollution data from OpenWeather API for model training.
"""

import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher
from src.database import Database


def backfill_historical_data(days_back: int = 30):
    """
    Fetch and store historical pollution data.
    
    Args:
        days_back: Number of days of historical data to fetch
    
    Returns:
        Number of records saved
    """
    print("=" * 60)
    print("HISTORICAL DATA BACKFILL")
    print("=" * 60)
    print(f"\n📅 Fetching data for the past {days_back} days...")
    
    fetcher = DataFetcher()
    db = Database()
    
    historical_data = fetcher.fetch_historical_pollution(days_back=days_back)
    
    if not historical_data:
        print("❌ No historical data retrieved!")
        return 0
    
    print(f"   Retrieved {len(historical_data)} data points")
    
    saved_count = 0
    for data_point in historical_data:
        try:
            data_point["city"] = fetcher.city
            data_point["lat"] = fetcher.lat
            data_point["lon"] = fetcher.lon
            
            db.raw_data.insert_one(data_point)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"   Saved {saved_count} records...")
                
        except Exception as e:
            print(f"   ⚠️ Error saving record: {e}")
    
    print(f"\n✅ Saved {saved_count} records")
    
    if historical_data:
        print(f"📅 Range: {historical_data[0]['timestamp'].date()} to {historical_data[-1]['timestamp'].date()}")
    
    stats = db.get_collection_stats()
    print(f"📊 Total records in database: {stats['raw_data_count']}")
    
    return saved_count


def show_sample_data():
    """Display a few sample records from the database."""
    db = Database()
    samples = list(db.raw_data.find().limit(3))
    
    if not samples:
        print("No data in database yet!")
        return
    
    for i, sample in enumerate(samples, 1):
        print(f"\n📌 Record {i}:")
        print(f"   Timestamp: {sample.get('timestamp', 'N/A')}")
        print(f"   AQI: {sample.get('aqi_standard', 'N/A')}")
        print(f"   PM2.5: {sample.get('pm2_5', 'N/A')} μg/m³")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            days = 30
    else:
        days = 30
    
    count = backfill_historical_data(days_back=days)
    
    if count > 0:
        show_sample_data()
    
    print("\n✅ Backfill complete!")
