"""
Backfill Historical Data Script
================================

WHAT THIS FILE DOES:
- Fetches historical air pollution data from the past (e.g., last 30 days)
- Stores this data in MongoDB
- This historical data is used to TRAIN our machine learning model

WHY DO WE NEED HISTORICAL DATA?
- Machine learning models learn from PAST examples
- More data = better predictions (usually)
- We need weeks/months of data to see patterns (daily, weekly, seasonal)

HOW IT WORKS:
1. We ask OpenWeather API for pollution data from the past
2. We clean and store each data point in MongoDB
3. Later, we'll use this data for EDA and model training
"""

import sys
import os
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher
from src.database import Database


def backfill_historical_data(days_back: int = 30):
    """
    Fetch and store historical pollution data.
    
    Parameters:
        days_back (int): How many days of historical data to fetch
                        Default is 30 days
    
    Returns:
        int: Number of records saved
    """
    print("=" * 60)
    print("HISTORICAL DATA BACKFILL")
    print("=" * 60)
    print(f"\n📅 Fetching data for the past {days_back} days...")
    
    # Step 1: Create our data fetcher and database connections
    print("\n1️⃣ Connecting to API and Database...")
    fetcher = DataFetcher()
    db = Database()
    
    # Step 2: Fetch historical pollution data from API
    print(f"\n2️⃣ Fetching historical data from OpenWeather API...")
    historical_data = fetcher.fetch_historical_pollution(days_back=days_back)
    
    if not historical_data:
        print("❌ No historical data retrieved!")
        return 0
    
    print(f"   Retrieved {len(historical_data)} data points")
    
    # Step 3: Save to database
    print(f"\n3️⃣ Saving data to MongoDB...")
    
    saved_count = 0
    for data_point in historical_data:
        try:
            # Add city information
            data_point["city"] = fetcher.city
            data_point["lat"] = fetcher.lat
            data_point["lon"] = fetcher.lon
            
            # Save to database (silently, without printing each one)
            db.raw_data.insert_one(data_point)
            saved_count += 1
            
            # Print progress every 100 records
            if saved_count % 100 == 0:
                print(f"   Saved {saved_count} records...")
                
        except Exception as e:
            print(f"   ⚠️ Error saving record: {e}")
    
    print(f"\n✅ Successfully saved {saved_count} records to database!")
    
    # Step 4: Show summary
    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    
    # Get date range
    if historical_data:
        first_date = historical_data[0]["timestamp"]
        last_date = historical_data[-1]["timestamp"]
        print(f"📅 Date range: {first_date.date()} to {last_date.date()}")
    
    # Show current database stats
    stats = db.get_collection_stats()
    print(f"📊 Total raw data records in database: {stats['raw_data_count']}")
    
    return saved_count


def show_sample_data():
    """
    Show a sample of the data we've stored.
    """
    print("\n" + "=" * 60)
    print("SAMPLE OF STORED DATA")
    print("=" * 60)
    
    db = Database()
    
    # Get 3 sample records
    samples = list(db.raw_data.find().limit(3))
    
    if not samples:
        print("No data in database yet!")
        return
    
    for i, sample in enumerate(samples, 1):
        print(f"\n📌 Record {i}:")
        print(f"   Timestamp: {sample.get('timestamp', 'N/A')}")
        print(f"   AQI (1-5): {sample.get('aqi', 'N/A')}")
        print(f"   AQI (standard): {sample.get('aqi_standard', 'N/A')}")
        print(f"   PM2.5: {sample.get('pm2_5', 'N/A')} μg/m³")
        print(f"   PM10: {sample.get('pm10', 'N/A')} μg/m³")


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    """
    Run this script directly: python src/backfill_historical.py
    
    Optional: Specify number of days as command line argument
        python src/backfill_historical.py 60  (for 60 days)
    """
    
    # Check if user specified number of days
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print("Invalid number of days. Using default (30).")
            days = 30
    else:
        days = 30
    
    # Run the backfill
    count = backfill_historical_data(days_back=days)
    
    # Show sample data
    if count > 0:
        show_sample_data()
    
    print("\n🎉 Backfill complete!")
