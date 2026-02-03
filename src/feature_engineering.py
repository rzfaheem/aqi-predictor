"""
Feature Engineering Script
==========================

WHAT IS FEATURE ENGINEERING?
- It's the process of creating new "features" (input variables) from raw data
- Good features = better model predictions
- It's one of the MOST IMPORTANT steps in machine learning

WHAT THIS SCRIPT DOES:
1. Loads raw data from MongoDB
2. Cleans the data (handle missing values, outliers)
3. Creates new features:
   - Time-based: hour, day, month, is_weekend
   - Lag features: yesterday's AQI, AQI 2 days ago
   - Rolling features: 3-day average, 7-day average
   - Change features: AQI change rate
4. Stores processed features in MongoDB Feature Store

FEATURES WE'LL CREATE:

Time Features (from timestamp):
- hour: 0-23 (time of day)
- day_of_week: 0-6 (Monday-Sunday)
- month: 1-12
- is_weekend: 0 or 1

Lag Features (past values):
- aqi_lag_1: AQI from 1 hour ago
- aqi_lag_24: AQI from 24 hours ago (same time yesterday)
- pm25_lag_1: PM2.5 from 1 hour ago

Rolling Features (averages over time):
- aqi_rolling_3h: Average AQI over last 3 hours
- aqi_rolling_24h: Average AQI over last 24 hours
- pm25_rolling_24h: Average PM2.5 over last 24 hours

Change Features:
- aqi_change_1h: How much AQI changed in last hour
- aqi_change_24h: How much AQI changed in last 24 hours

WHY THESE FEATURES?
- Time features: Capture daily and weekly patterns
- Lag features: "What was the AQI before?" helps predict future
- Rolling features: Smooths out noise, captures trends
- Change features: Is pollution increasing or decreasing?
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database


def load_raw_data():
    """
    Load raw data from MongoDB.
    """
    print("📊 Loading raw data from MongoDB...")
    db = Database()
    raw_data = db.get_raw_data()
    
    if not raw_data:
        print("❌ No data found!")
        return None
    
    df = pd.DataFrame(raw_data)
    print(f"✅ Loaded {len(df)} records")
    return df


def clean_data(df):
    """
    Clean the raw data.
    
    STEPS:
    1. Handle missing values
    2. Remove duplicates
    3. Sort by timestamp
    4. Handle outliers
    """
    print("\n🧹 Cleaning data...")
    initial_count = len(df)
    
    # Step 1: Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Step 2: Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Step 3: Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Step 4: Fill missing values with forward fill (use previous value)
    # This is common for time series data
    numeric_cols = ['aqi', 'aqi_standard', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    # Step 5: Handle outliers using IQR method for AQI
    # (Values too extreme might be errors)
    if 'aqi_standard' in df.columns:
        Q1 = df['aqi_standard'].quantile(0.25)
        Q3 = df['aqi_standard'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df['aqi_standard'] = df['aqi_standard'].clip(lower=max(0, lower_bound), upper=upper_bound)
    
    final_count = len(df)
    print(f"✅ Cleaned data: {initial_count} → {final_count} records")
    
    return df


def create_time_features(df):
    """
    Create time-based features from timestamp.
    
    These help the model understand:
    - What time of day affects AQI (rush hour vs midnight)
    - What day of week affects AQI (weekday vs weekend)
    - What month affects AQI (winter vs summer)
    """
    print("\n⏰ Creating time features...")
    
    if 'timestamp' not in df.columns:
        print("❌ No timestamp column found!")
        return df
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour                    # 0-23
    df['day_of_week'] = df['timestamp'].dt.dayofweek        # 0-6 (Mon-Sun)
    df['day'] = df['timestamp'].dt.day                       # 1-31
    df['month'] = df['timestamp'].dt.month                   # 1-12
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 1 if Sat/Sun
    
    # Cyclical encoding for hour (so 23 is close to 0)
    # This helps the model understand that hour 23 and hour 0 are adjacent
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    print("   ✅ Created: hour, day_of_week, day, month, is_weekend, hour_sin, hour_cos")
    return df


def create_lag_features(df, target_col='aqi_standard'):
    """
    Create lag features (past values).
    
    LAG = Looking at past values
    - lag_1 = value from 1 hour ago
    - lag_24 = value from 24 hours ago (same time yesterday)
    
    WHY LAGS?
    - If AQI was high yesterday at this time, it's likely high today too
    - Recent values are strong predictors of future values
    """
    print("\n⏪ Creating lag features...")
    
    if target_col not in df.columns:
        print(f"❌ Column {target_col} not found!")
        return df
    
    # Lag features for AQI
    df['aqi_lag_1h'] = df[target_col].shift(1)       # 1 hour ago
    df['aqi_lag_3h'] = df[target_col].shift(3)       # 3 hours ago
    df['aqi_lag_6h'] = df[target_col].shift(6)       # 6 hours ago
    df['aqi_lag_12h'] = df[target_col].shift(12)     # 12 hours ago
    df['aqi_lag_24h'] = df[target_col].shift(24)     # 24 hours ago (yesterday same time)
    
    # Lag features for PM2.5
    if 'pm2_5' in df.columns:
        df['pm25_lag_1h'] = df['pm2_5'].shift(1)
        df['pm25_lag_24h'] = df['pm2_5'].shift(24)
    
    print("   ✅ Created: aqi_lag_1h/3h/6h/12h/24h, pm25_lag_1h/24h")
    return df


def create_rolling_features(df, target_col='aqi_standard'):
    """
    Create rolling (moving average) features.
    
    ROLLING = Average over a window of time
    - rolling_3h = average of last 3 hours
    - rolling_24h = average of last 24 hours
    
    WHY ROLLING AVERAGES?
    - Smooths out random fluctuations (noise)
    - Shows the overall trend
    - More stable than single point values
    """
    print("\n📈 Creating rolling features...")
    
    if target_col not in df.columns:
        print(f"❌ Column {target_col} not found!")
        return df
    
    # Rolling averages for AQI
    df['aqi_rolling_3h'] = df[target_col].rolling(window=3, min_periods=1).mean()
    df['aqi_rolling_6h'] = df[target_col].rolling(window=6, min_periods=1).mean()
    df['aqi_rolling_12h'] = df[target_col].rolling(window=12, min_periods=1).mean()
    df['aqi_rolling_24h'] = df[target_col].rolling(window=24, min_periods=1).mean()
    
    # Rolling std (shows volatility/stability)
    df['aqi_std_24h'] = df[target_col].rolling(window=24, min_periods=1).std()
    
    # Rolling averages for PM2.5
    if 'pm2_5' in df.columns:
        df['pm25_rolling_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
        df['pm25_rolling_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    
    print("   ✅ Created: aqi_rolling_3h/6h/12h/24h, aqi_std_24h, pm25_rolling_6h/24h")
    return df


def create_change_features(df, target_col='aqi_standard'):
    """
    Create change/difference features.
    
    CHANGE = How much did the value change?
    - Positive = AQI is increasing (getting worse)
    - Negative = AQI is decreasing (improving)
    
    WHY CHANGE FEATURES?
    - Shows the DIRECTION of change (improving or worsening)
    - Recent trends help predict future values
    """
    print("\n📊 Creating change features...")
    
    if target_col not in df.columns:
        print(f"❌ Column {target_col} not found!")
        return df
    
    # Absolute change
    df['aqi_change_1h'] = df[target_col] - df[target_col].shift(1)
    df['aqi_change_3h'] = df[target_col] - df[target_col].shift(3)
    df['aqi_change_6h'] = df[target_col] - df[target_col].shift(6)
    df['aqi_change_24h'] = df[target_col] - df[target_col].shift(24)
    
    # Percentage change
    df['aqi_pct_change_1h'] = df[target_col].pct_change(1) * 100
    df['aqi_pct_change_24h'] = df[target_col].pct_change(24) * 100
    
    print("   ✅ Created: aqi_change_1h/3h/6h/24h, aqi_pct_change_1h/24h")
    return df


def create_target_variable(df, target_col='pm2_5', forecast_hours=24):
    """
    Create target variables for 3-day forecast using PM2.5.
    
    WHY PM2.5 INSTEAD OF AQI?
    - PM2.5 is continuous (145.2, 152.8, 160.5...) - better for ML training
    - AQI from OpenWeather is discrete (only 5 levels) - 97% same value problem
    - We convert PM2.5 back to AQI categories for display after prediction
    """
    print("\n🎯 Creating target variables (using PM2.5 for continuous values)...")
    
    if target_col not in df.columns:
        print(f"❌ Column {target_col} not found!")
        return df
    
    # Shift BACKWARDS to get future values
    # shift(-24) gets the PM2.5 value from 24 hours ahead
    df['target_24h'] = df[target_col].shift(-24)   # 1 day ahead
    df['target_48h'] = df[target_col].shift(-48)   # 2 days ahead
    df['target_72h'] = df[target_col].shift(-72)   # 3 days ahead
    
    print("   ✅ Created: target_24h/48h/72h (based on PM2.5)")
    return df


def prepare_final_features(df):
    """
    Prepare the final feature set for model training.
    
    STEPS:
    1. Select relevant columns
    2. Drop rows with missing values (from lag/rolling calculations)
    3. Return clean dataset
    """
    print("\n🏁 Preparing final feature set...")
    
    # Define feature columns
    feature_cols = [
        # Time features
        'hour', 'day_of_week', 'day', 'month', 'is_weekend',
        'hour_sin', 'hour_cos',
        
        # Weather features (IMPORTANT for AQI prediction!)
        'temp', 'humidity', 'pressure', 'wind_speed', 'clouds',
        
        # Pollutant values
        'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2',
        
        # Lag features
        'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h', 'aqi_lag_12h', 'aqi_lag_24h',
        'pm25_lag_1h', 'pm25_lag_24h',
        
        # Rolling features
        'aqi_rolling_3h', 'aqi_rolling_6h', 'aqi_rolling_12h', 'aqi_rolling_24h',
        'aqi_std_24h', 'pm25_rolling_6h', 'pm25_rolling_24h',
        
        # Change features
        'aqi_change_1h', 'aqi_change_3h', 'aqi_change_6h', 'aqi_change_24h',
        'aqi_pct_change_1h', 'aqi_pct_change_24h'
    ]
    
    # Target columns (only the ones we actually predict)
    target_cols = ['target_24h', 'target_48h', 'target_72h']
    
    # Select available columns
    available_features = [col for col in feature_cols if col in df.columns]
    available_targets = [col for col in target_cols if col in df.columns]
    
    # Also keep timestamp for reference
    all_cols = ['timestamp'] + available_features + available_targets
    df_final = df[all_cols].copy()
    
    # Drop rows with NaN (from lag/rolling calculations)
    initial_count = len(df_final)
    df_final = df_final.dropna()
    final_count = len(df_final)
    
    print(f"   Rows: {initial_count} → {final_count} (removed {initial_count - final_count} with missing values)")
    print(f"   Features: {len(available_features)}")
    print(f"   Targets: {len(available_targets)}")
    
    # Handle case where no complete records are available
    if final_count == 0:
        print("   ⚠️ Warning: No complete records after removing NaN. Need more historical data.")
        print("   ⚠️ This is expected for new installations with limited data.")
        # Return empty but valid structure
        return df_final, available_features, available_targets
    
    return df_final, available_features, available_targets


def save_features_to_mongodb(df, features, targets):
    """
    Save processed features to MongoDB Feature Store.
    """
    print("\n💾 Saving features to MongoDB Feature Store...")
    
    db = Database()
    
    # Convert DataFrame to list of dictionaries
    records = df.to_dict('records')
    
    # Handle empty data case
    if len(records) == 0:
        print("   ⚠️ No records to save. Skipping database update.")
        print("   ⚠️ This is expected when there's insufficient historical data.")
        return
    
    # Clear existing features (to avoid duplicates)
    db.features.delete_many({})
    
    # Save in batches
    batch_size = 100
    saved_count = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        db.features.insert_many(batch)
        saved_count += len(batch)
        
        if saved_count % 200 == 0:
            print(f"   Saved {saved_count} records...")
    
    print(f"✅ Saved {saved_count} feature records to MongoDB")
    
    # Save feature names for later use
    feature_info = {
        "feature_names": features,
        "target_names": targets,
        "created_at": datetime.utcnow(),
        "record_count": saved_count
    }
    db.db["feature_info"].delete_many({})
    db.db["feature_info"].insert_one(feature_info)
    print("✅ Saved feature metadata")
    
    return saved_count


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw data
    df = load_raw_data()
    if df is None:
        print("❌ Cannot proceed without data!")
        exit(1)
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Create all features
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_change_features(df)
    df = create_target_variable(df)
    
    # Step 4: Prepare final feature set
    df_final, feature_names, target_names = prepare_final_features(df)
    
    # Step 5: Save to MongoDB
    save_features_to_mongodb(df_final, feature_names, target_names)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"\n📌 Total Features Created: {len(feature_names)}")
    print(f"🎯 Target Variables: {len(target_names)}")
    print(f"📈 Total Records: {len(df_final)}")
    
    print("\n📋 Feature Categories:")
    print("   • Time features: hour, day_of_week, month, is_weekend, etc.")
    print("   • Lag features: aqi_lag_1h, aqi_lag_24h, etc.")
    print("   • Rolling features: aqi_rolling_3h, aqi_rolling_24h, etc.")
    print("   • Change features: aqi_change_1h, aqi_pct_change_24h, etc.")
    
    print("\n🎉 Feature engineering complete!")
