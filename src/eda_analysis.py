"""
Exploratory Data Analysis - Generates visualizations and statistics from AQI data.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data_from_mongodb():
    """Load historical data from MongoDB into a DataFrame."""
    print("Loading data from MongoDB...")
    
    db = Database()
    raw_data = db.get_raw_data()
    
    if not raw_data:
        print("No data found!")
        return None
    
    df = pd.DataFrame(raw_data)
    print(f"Loaded {len(df)} records")
    return df


def clean_data(df):
    """Clean data: parse timestamps, remove duplicates, sort chronologically."""
    print("\nCleaning data...")
    
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"   Missing values in: {list(missing_cols.index)}")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    duplicates = df.duplicated(subset=['timestamp'], keep='first').sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        print(f"   Removed {duplicates} duplicates")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"{len(df)} records ready for analysis")
    return df


def generate_summary_statistics(df):
    """Print summary statistics for pollution metrics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    numeric_cols = ['aqi', 'aqi_standard', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if available_cols:
        stats = df[available_cols].describe()
        print(stats.round(2).to_string())
        
        if 'aqi_standard' in df.columns:
            print(f"\n   Average AQI: {df['aqi_standard'].mean():.1f}")
            print(f"   AQI Range: {df['aqi_standard'].min():.0f} - {df['aqi_standard'].max():.0f}")
        if 'pm2_5' in df.columns:
            print(f"   Average PM2.5: {df['pm2_5'].mean():.2f} ug/m3")


def create_visualizations(df, output_dir="notebooks/eda_charts"):
    """Generate and save EDA charts (time series, distributions, correlations)."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. AQI Over Time
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['timestamp'], df['aqi_standard'], linewidth=1, alpha=0.7, color='#e74c3c')
    ax.fill_between(df['timestamp'], df['aqi_standard'], alpha=0.3, color='#e74c3c')
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Good (0-50)')
    ax.axhline(y=100, color='yellow', linestyle='--', alpha=0.5, label='Moderate (51-100)')
    ax.axhline(y=150, color='orange', linestyle='--', alpha=0.5, label='Unhealthy for Sensitive (101-150)')
    ax.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='Unhealthy (151-200)')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('AQI (Standard Scale)', fontsize=12)
    ax.set_title('Air Quality Index Over Time - Faisalabad', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_aqi_over_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. AQI Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(df['aqi_standard'], bins=30, edgecolor='white', alpha=0.7)
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center <= 50:
            patch.set_facecolor('green')
        elif bin_center <= 100:
            patch.set_facecolor('#FFD700')
        elif bin_center <= 150:
            patch.set_facecolor('orange')
        elif bin_center <= 200:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('purple')
    ax.axvline(df['aqi_standard'].mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {df["aqi_standard"].mean():.1f}')
    ax.set_xlabel('AQI Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of AQI Values', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_aqi_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. AQI by Hour (Box Plot)
    fig, ax = plt.subplots(figsize=(14, 6))
    df.boxplot(column='aqi_standard', by='hour', ax=ax)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('AQI', fontsize=12)
    ax.set_title('AQI Patterns Throughout the Day', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_aqi_by_hour.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. AQI by Day of Week
    fig, ax = plt.subplots(figsize=(10, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df.groupby('day_name')['aqi_standard'].mean().reindex(day_order)
    colors = ['#3498db' if day not in ['Saturday', 'Sunday'] else '#e74c3c' for day in day_order]
    ax.bar(day_order, daily_avg.values, color=colors, edgecolor='white')
    ax.axhline(y=df['aqi_standard'].mean(), color='black', linestyle='--', label=f'Overall Mean: {df["aqi_standard"].mean():.1f}')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Average AQI', fontsize=12)
    ax.set_title('Average AQI by Day of Week (Red = Weekend)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_aqi_by_day.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation Heatmap
    corr_cols = ['aqi_standard', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'hour', 'day_of_week']
    available_corr_cols = [col for col in corr_cols if col in df.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[available_corr_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, linewidths=0.5, ax=ax, fmt='.2f', annot_kws={'size': 10})
    ax.set_title('Correlation Between Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. PM2.5 vs PM10 Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['pm2_5'], df['pm10'], c=df['aqi_standard'], 
                        cmap='RdYlGn_r', alpha=0.6, s=50)
    plt.colorbar(scatter, label='AQI')
    ax.set_xlabel('PM2.5 (ug/m3)', fontsize=12)
    ax.set_ylabel('PM10 (ug/m3)', fontsize=12)
    ax.set_title('PM2.5 vs PM10 (colored by AQI)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_pm25_vs_pm10.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"All charts saved to: {output_dir}/")
    return output_dir


def generate_eda_report(df, chart_dir):
    """Print key findings from the EDA."""
    print("\n" + "=" * 60)
    print("EDA REPORT")
    print("=" * 60)
    
    if 'timestamp' in df.columns:
        print(f"\nPeriod: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"   Total records: {len(df)} (hourly)")
    
    print(f"\nAQI Statistics:")
    print(f"   Mean: {df['aqi_standard'].mean():.1f}, Median: {df['aqi_standard'].median():.1f}")
    print(f"   Range: {df['aqi_standard'].min():.0f} - {df['aqi_standard'].max():.0f}")
    
    # AQI level distribution
    good = len(df[df['aqi_standard'] <= 50])
    moderate = len(df[(df['aqi_standard'] > 50) & (df['aqi_standard'] <= 100)])
    sensitive = len(df[(df['aqi_standard'] > 100) & (df['aqi_standard'] <= 150)])
    unhealthy = len(df[(df['aqi_standard'] > 150) & (df['aqi_standard'] <= 200)])
    very_unhealthy = len(df[df['aqi_standard'] > 200])
    
    print(f"\nAQI Level Distribution:")
    print(f"   Good (0-50): {good} ({100*good/len(df):.1f}%)")
    print(f"   Moderate (51-100): {moderate} ({100*moderate/len(df):.1f}%)")
    print(f"   Unhealthy for Sensitive (101-150): {sensitive} ({100*sensitive/len(df):.1f}%)")
    print(f"   Unhealthy (151-200): {unhealthy} ({100*unhealthy/len(df):.1f}%)")
    print(f"   Very Unhealthy (>200): {very_unhealthy} ({100*very_unhealthy/len(df):.1f}%)")
    
    if 'hour' in df.columns:
        hourly_avg = df.groupby('hour')['aqi_standard'].mean()
        print(f"\nWorst hour: {hourly_avg.idxmax()}:00 (Avg: {hourly_avg.max():.1f})")
        print(f"   Best hour: {hourly_avg.idxmin()}:00 (Avg: {hourly_avg.min():.1f})")
    
    if 'is_weekend' in df.columns:
        weekday_avg = df[df['is_weekend'] == 0]['aqi_standard'].mean()
        weekend_avg = df[df['is_weekend'] == 1]['aqi_standard'].mean()
        print(f"\nWeekday Avg: {weekday_avg:.1f}, Weekend Avg: {weekend_avg:.1f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    df = load_data_from_mongodb()
    if df is None:
        exit(1)
    
    df = clean_data(df)
    generate_summary_statistics(df)
    chart_dir = create_visualizations(df)
    generate_eda_report(df, chart_dir)
    
    print("\nEDA Complete!")
