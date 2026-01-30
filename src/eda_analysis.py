"""
Exploratory Data Analysis (EDA) Script
=======================================

WHAT IS EDA?
- EDA = Exploratory Data Analysis
- It means EXPLORING and UNDERSTANDING your data before building models
- We look for patterns, outliers, missing values, and trends
- This helps us make better decisions when building our model

WHAT THIS SCRIPT DOES:
1. Loads historical data from MongoDB
2. Cleans and prepares the data
3. Creates visualizations (charts and graphs)
4. Identifies patterns and insights
5. Saves all charts as images
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')  # Clean, professional look
sns.set_palette("husl")  # Nice color palette


def load_data_from_mongodb():
    """
    Load historical data from MongoDB and convert to pandas DataFrame.
    
    WHAT IS A DATAFRAME?
    - Think of it as an Excel spreadsheet in Python
    - Rows = data points (each hour's reading)
    - Columns = features (temperature, AQI, PM2.5, etc.)
    """
    print("📊 Loading data from MongoDB...")
    
    db = Database()
    raw_data = db.get_raw_data()
    
    if not raw_data:
        print("❌ No data found in database!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    print(f"✅ Loaded {len(df)} records")
    return df


def clean_data(df):
    """
    Clean and prepare the data for analysis.
    
    WHAT IS DATA CLEANING?
    - Removing or fixing bad/missing data
    - Converting data to the right format
    - Making the data consistent
    
    WHY IS IT IMPORTANT?
    - "Garbage in, garbage out" - bad data = bad model
    - Clean data gives better predictions
    """
    print("\n🧹 Cleaning data...")
    
    # Step 1: Check for missing values
    print("\n1️⃣ Checking for missing values...")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"   Found missing values in: {list(missing_cols.index)}")
    else:
        print("   ✅ No missing values found!")
    
    # Step 2: Convert timestamp to datetime if it's not already
    print("\n2️⃣ Processing timestamps...")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract useful time features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
        df['day_name'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print("   ✅ Extracted time features: hour, day, day_of_week, day_name, month, is_weekend")
    
    # Step 3: Remove duplicates
    print("\n3️⃣ Checking for duplicates...")
    duplicates = df.duplicated(subset=['timestamp'], keep='first').sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        print(f"   Removed {duplicates} duplicate records")
    else:
        print("   ✅ No duplicates found!")
    
    # Step 4: Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    print("\n4️⃣ Sorted data by timestamp")
    
    print(f"\n✅ Data cleaning complete! {len(df)} records ready for analysis.")
    return df


def generate_summary_statistics(df):
    """
    Generate and display summary statistics of the data.
    
    WHAT ARE SUMMARY STATISTICS?
    - Quick numbers that describe your data
    - Mean (average), Min, Max, Standard Deviation, etc.
    - Helps you understand the range and distribution of values
    """
    print("\n" + "=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)
    
    # Select numeric columns for analysis
    numeric_cols = ['aqi', 'aqi_standard', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if available_cols:
        stats = df[available_cols].describe()
        print("\n📈 Pollution Metrics Statistics:")
        print(stats.round(2).to_string())
        
        # Additional insights
        print(f"\n💡 Key Insights:")
        if 'aqi_standard' in df.columns:
            print(f"   • Average AQI: {df['aqi_standard'].mean():.1f}")
            print(f"   • AQI Range: {df['aqi_standard'].min():.0f} to {df['aqi_standard'].max():.0f}")
        if 'pm2_5' in df.columns:
            print(f"   • Average PM2.5: {df['pm2_5'].mean():.2f} μg/m³")
        if 'pm10' in df.columns:
            print(f"   • Average PM10: {df['pm10'].mean():.2f} μg/m³")


def create_visualizations(df, output_dir="notebooks/eda_charts"):
    """
    Create various visualizations to understand the data.
    
    TYPES OF CHARTS WE'LL CREATE:
    1. Line chart - Shows how AQI changes over time
    2. Histogram - Shows distribution of AQI values
    3. Box plot - Shows AQI by hour of day
    4. Heatmap - Shows correlations between features
    5. Bar chart - Shows AQI by day of week
    """
    print("\n" + "=" * 60)
    print("📈 CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================
    # Chart 1: AQI Over Time (Line Chart)
    # ========================================
    print("\n1️⃣ Creating AQI trend chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['timestamp'], df['aqi_standard'], linewidth=1, alpha=0.7, color='#e74c3c')
    ax.fill_between(df['timestamp'], df['aqi_standard'], alpha=0.3, color='#e74c3c')
    
    # Add AQI level zones
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
    
    chart_path = os.path.join(output_dir, '1_aqi_over_time.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {chart_path}")
    
    # ========================================
    # Chart 2: AQI Distribution (Histogram)
    # ========================================
    print("\n2️⃣ Creating AQI distribution chart...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with color-coded bins
    n, bins, patches = ax.hist(df['aqi_standard'], bins=30, edgecolor='white', alpha=0.7)
    
    # Color the bins based on AQI levels
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
    ax.set_ylabel('Frequency (Count)', fontsize=12)
    ax.set_title('Distribution of AQI Values', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, '2_aqi_distribution.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {chart_path}")
    
    # ========================================
    # Chart 3: AQI by Hour of Day (Box Plot)
    # ========================================
    print("\n3️⃣ Creating hourly AQI pattern chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Group by hour and plot
    df.boxplot(column='aqi_standard', by='hour', ax=ax)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('AQI', fontsize=12)
    ax.set_title('AQI Patterns Throughout the Day', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, '3_aqi_by_hour.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {chart_path}")
    
    # ========================================
    # Chart 4: AQI by Day of Week (Bar Chart)
    # ========================================
    print("\n4️⃣ Creating weekly AQI pattern chart...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df.groupby('day_name')['aqi_standard'].mean().reindex(day_order)
    
    colors = ['#3498db' if day not in ['Saturday', 'Sunday'] else '#e74c3c' for day in day_order]
    bars = ax.bar(day_order, daily_avg.values, color=colors, edgecolor='white')
    
    ax.axhline(y=df['aqi_standard'].mean(), color='black', linestyle='--', label=f'Overall Mean: {df["aqi_standard"].mean():.1f}')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Average AQI', fontsize=12)
    ax.set_title('Average AQI by Day of Week (Red = Weekend)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, '4_aqi_by_day.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {chart_path}")
    
    # ========================================
    # Chart 5: Correlation Heatmap
    # ========================================
    print("\n5️⃣ Creating correlation heatmap...")
    
    # Select numeric columns for correlation
    corr_cols = ['aqi_standard', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'hour', 'day_of_week']
    available_corr_cols = [col for col in corr_cols if col in df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[available_corr_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, linewidths=0.5, ax=ax, fmt='.2f',
                annot_kws={'size': 10})
    ax.set_title('Correlation Between Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, '5_correlation_heatmap.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {chart_path}")
    
    # ========================================
    # Chart 6: PM2.5 vs PM10 Scatter Plot
    # ========================================
    print("\n6️⃣ Creating PM2.5 vs PM10 scatter plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['pm2_5'], df['pm10'], c=df['aqi_standard'], 
                        cmap='RdYlGn_r', alpha=0.6, s=50)
    plt.colorbar(scatter, label='AQI')
    ax.set_xlabel('PM2.5 (μg/m³)', fontsize=12)
    ax.set_ylabel('PM10 (μg/m³)', fontsize=12)
    ax.set_title('PM2.5 vs PM10 (colored by AQI)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, '6_pm25_vs_pm10.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {chart_path}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    return output_dir


def generate_eda_report(df, chart_dir):
    """
    Generate a text-based EDA report with key findings.
    """
    print("\n" + "=" * 60)
    print("📝 EDA REPORT - KEY FINDINGS")
    print("=" * 60)
    
    # Date range
    if 'timestamp' in df.columns:
        print(f"\n📅 Data Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"   Total records: {len(df)}")
        print(f"   Data frequency: Hourly")
    
    # AQI Analysis
    print(f"\n🌫️ AQI Analysis:")
    print(f"   • Mean AQI: {df['aqi_standard'].mean():.1f}")
    print(f"   • Median AQI: {df['aqi_standard'].median():.1f}")
    print(f"   • Min AQI: {df['aqi_standard'].min():.0f}")
    print(f"   • Max AQI: {df['aqi_standard'].max():.0f}")
    print(f"   • Std Dev: {df['aqi_standard'].std():.1f}")
    
    # AQI Categories
    print(f"\n📊 AQI Level Distribution:")
    good = len(df[df['aqi_standard'] <= 50])
    moderate = len(df[(df['aqi_standard'] > 50) & (df['aqi_standard'] <= 100)])
    sensitive = len(df[(df['aqi_standard'] > 100) & (df['aqi_standard'] <= 150)])
    unhealthy = len(df[(df['aqi_standard'] > 150) & (df['aqi_standard'] <= 200)])
    very_unhealthy = len(df[df['aqi_standard'] > 200])
    
    print(f"   • Good (0-50): {good} records ({100*good/len(df):.1f}%)")
    print(f"   • Moderate (51-100): {moderate} records ({100*moderate/len(df):.1f}%)")
    print(f"   • Unhealthy for Sensitive (101-150): {sensitive} records ({100*sensitive/len(df):.1f}%)")
    print(f"   • Unhealthy (151-200): {unhealthy} records ({100*unhealthy/len(df):.1f}%)")
    print(f"   • Very Unhealthy (>200): {very_unhealthy} records ({100*very_unhealthy/len(df):.1f}%)")
    
    # Hourly patterns
    if 'hour' in df.columns:
        hourly_avg = df.groupby('hour')['aqi_standard'].mean()
        worst_hour = hourly_avg.idxmax()
        best_hour = hourly_avg.idxmin()
        print(f"\n⏰ Hourly Patterns:")
        print(f"   • Worst hour: {worst_hour}:00 (Avg AQI: {hourly_avg[worst_hour]:.1f})")
        print(f"   • Best hour: {best_hour}:00 (Avg AQI: {hourly_avg[best_hour]:.1f})")
    
    # Weekend vs Weekday
    if 'is_weekend' in df.columns:
        weekday_avg = df[df['is_weekend'] == 0]['aqi_standard'].mean()
        weekend_avg = df[df['is_weekend'] == 1]['aqi_standard'].mean()
        print(f"\n📆 Weekday vs Weekend:")
        print(f"   • Weekday Average: {weekday_avg:.1f}")
        print(f"   • Weekend Average: {weekend_avg:.1f}")
        if weekend_avg < weekday_avg:
            print(f"   💡 Air quality is better on weekends!")
        else:
            print(f"   💡 Air quality is similar or worse on weekends")
    
    print(f"\n📁 Charts saved in: {chart_dir}/")
    print("\n" + "=" * 60)


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 EXPLORATORY DATA ANALYSIS (EDA)")
    print("    Faisalabad Air Quality Data")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data_from_mongodb()
    if df is None:
        print("❌ Cannot proceed without data!")
        exit(1)
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Generate summary statistics
    generate_summary_statistics(df)
    
    # Step 4: Create visualizations
    chart_dir = create_visualizations(df)
    
    # Step 5: Generate report
    generate_eda_report(df, chart_dir)
    
    print("\n🎉 EDA Complete! Check the charts in 'notebooks/eda_charts/' folder")
