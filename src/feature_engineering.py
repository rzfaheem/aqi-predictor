"""
Feature Engineering - Transforms raw weather/pollution data into ML-ready features.

Creates time-based, lag, rolling average, and change rate features from raw data,
then stores them in MongoDB as the feature store for model training.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database


def load_raw_data():
    """Load raw data from MongoDB into a DataFrame."""
    print("Loading raw data from MongoDB...")
    db = Database()
    raw_data = db.get_raw_data()
    
    if not raw_data:
        print("No data found!")
        return None
    
    df = pd.DataFrame(raw_data)
    print(f"Loaded {len(df)} records")
    return df


def clean_data(df):
    """Clean raw data: handle missing values, duplicates, outliers."""
    print("\nCleaning data...")
    initial_count = len(df)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Forward-fill missing values (standard for time series)
    # Weather columns are often missing from historical backfill data
    numeric_cols = [
        'aqi', 'aqi_standard', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2',
        'temperature', 'humidity', 'pressure', 'wind_speed'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Cap outliers using IQR method
    if 'aqi_standard' in df.columns:
        Q1 = df['aqi_standard'].quantile(0.25)
        Q3 = df['aqi_standard'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['aqi_standard'] = df['aqi_standard'].clip(lower=max(0, lower_bound), upper=upper_bound)
    
    print(f"Cleaned: {initial_count} -> {len(df)} records")
    return df


def create_time_features(df):
    """Extract time-based features from timestamp (hour, day, month, cyclical encoding)."""
    print("\nCreating time features...")
    
    if 'timestamp' not in df.columns:
        return df
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding so hour 23 and 0 are adjacent
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


def create_lag_features(df, target_col='aqi_standard'):
    """Create lag features (past values at 1h, 3h, 6h, 12h, 24h intervals)."""
    print("\nCreating lag features...")
    
    if target_col not in df.columns:
        return df
    
    df['aqi_lag_1h'] = df[target_col].shift(1)
    df['aqi_lag_3h'] = df[target_col].shift(3)
    df['aqi_lag_6h'] = df[target_col].shift(6)
    df['aqi_lag_12h'] = df[target_col].shift(12)
    df['aqi_lag_24h'] = df[target_col].shift(24)
    
    if 'pm2_5' in df.columns:
        df['pm25_lag_1h'] = df['pm2_5'].shift(1)
        df['pm25_lag_24h'] = df['pm2_5'].shift(24)
    
    return df


def create_rolling_features(df, target_col='aqi_standard'):
    """Create rolling average features (3h, 6h, 12h, 24h windows)."""
    print("\nCreating rolling features...")
    
    if target_col not in df.columns:
        return df
    
    df['aqi_rolling_3h'] = df[target_col].rolling(window=3, min_periods=1).mean()
    df['aqi_rolling_6h'] = df[target_col].rolling(window=6, min_periods=1).mean()
    df['aqi_rolling_12h'] = df[target_col].rolling(window=12, min_periods=1).mean()
    df['aqi_rolling_24h'] = df[target_col].rolling(window=24, min_periods=1).mean()
    df['aqi_std_24h'] = df[target_col].rolling(window=24, min_periods=1).std()
    
    if 'pm2_5' in df.columns:
        df['pm25_rolling_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
        df['pm25_rolling_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    
    return df


def create_change_features(df, target_col='aqi_standard'):
    """Create rate-of-change features (absolute and percentage)."""
    print("\nCreating change features...")
    
    if target_col not in df.columns:
        return df
    
    df['aqi_change_1h'] = df[target_col] - df[target_col].shift(1)
    df['aqi_change_3h'] = df[target_col] - df[target_col].shift(3)
    df['aqi_change_6h'] = df[target_col] - df[target_col].shift(6)
    df['aqi_change_24h'] = df[target_col] - df[target_col].shift(24)
    
    df['aqi_pct_change_1h'] = df[target_col].pct_change(1) * 100
    df['aqi_pct_change_24h'] = df[target_col].pct_change(24) * 100
    
    return df


def create_target_variable(df, target_col='pm2_5', forecast_hours=24):
    """Create multi-horizon target variables using PM2.5 (continuous, better for ML)."""
    print("\nCreating target variables...")
    
    if target_col not in df.columns:
        return df
    
    df['target_24h'] = df[target_col].shift(-24)
    df['target_48h'] = df[target_col].shift(-48)
    df['target_72h'] = df[target_col].shift(-72)
    
    return df


def prepare_final_features(df):
    """Select final feature columns and drop incomplete rows."""
    print("\nPreparing final feature set...")
    
    feature_cols = [
        'hour', 'day_of_week', 'day', 'month', 'is_weekend',
        'hour_sin', 'hour_cos',
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2',
        'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h', 'aqi_lag_12h', 'aqi_lag_24h',
        'pm25_lag_1h', 'pm25_lag_24h',
        'aqi_rolling_3h', 'aqi_rolling_6h', 'aqi_rolling_12h', 'aqi_rolling_24h',
        'aqi_std_24h', 'pm25_rolling_6h', 'pm25_rolling_24h',
        'aqi_change_1h', 'aqi_change_3h', 'aqi_change_6h', 'aqi_change_24h',
        'aqi_pct_change_1h', 'aqi_pct_change_24h'
    ]
    
    target_cols = ['target_24h', 'target_48h', 'target_72h']
    
    available_features = [col for col in feature_cols if col in df.columns]
    available_targets = [col for col in target_cols if col in df.columns]
    
    all_cols = ['timestamp'] + available_features + available_targets
    df_final = df[all_cols].copy()
    
    initial_count = len(df_final)
    df_final = df_final.dropna()
    final_count = len(df_final)
    
    print(f"   Rows: {initial_count} -> {final_count}")
    print(f"   Features: {len(available_features)}, Targets: {len(available_targets)}")
    
    if final_count == 0:
        print("   Warning: No complete records. Need more historical data.")
    
    return df_final, available_features, available_targets


def save_features_to_mongodb(df, features, targets):
    """Save processed features to MongoDB feature store."""
    print("\nSaving features to MongoDB...")
    
    db = Database()
    records = df.to_dict('records')
    
    if len(records) == 0:
        print("   No records to save.")
        return
    
    db.features.delete_many({})
    
    batch_size = 100
    saved_count = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        db.features.insert_many(batch)
        saved_count += len(batch)
    
    print(f"Saved {saved_count} feature records to MongoDB")
    
    feature_info = {
        "feature_names": features,
        "target_names": targets,
        "created_at": datetime.utcnow(),
        "record_count": saved_count
    }
    db.db["feature_info"].delete_many({})
    db.db["feature_info"].insert_one(feature_info)
    
    return saved_count


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    df = load_raw_data()
    if df is None:
        exit(1)
    
    df = clean_data(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_change_features(df)
    df = create_target_variable(df)
    
    df_final, feature_names, target_names = prepare_final_features(df)
    save_features_to_mongodb(df_final, feature_names, target_names)
    
    print(f"\nSummary: {len(feature_names)} features, {len(target_names)} targets, {len(df_final)} records")
    print("Feature engineering complete!")
