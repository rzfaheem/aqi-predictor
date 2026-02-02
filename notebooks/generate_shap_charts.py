"""
Simple Feature Analysis Script for AQI Prediction Model
=========================================================
Generates feature analysis visualizations (without heavy SHAP computations).
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database

# Create output directory
os.makedirs('notebooks/shap_charts', exist_ok=True)

print("=" * 60)
print("📊 FEATURE ANALYSIS - AQI Prediction Model")
print("=" * 60)

# ========================================
# 1. LOAD MODEL AND DATA
# ========================================
print("\n📦 Loading model and data...")

model_path = "models/best_model_multi_output_24h_48h_72h.pkl"
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']
model_name = model_data.get('model_name', 'Unknown')

print(f"✅ Model: {model_name}")

# Load data
db = Database()
features = db.get_features()
df = pd.DataFrame(features)

available_features = [f for f in feature_names if f in df.columns]
X = df[available_features].dropna()
X = X[available_features]

print(f"✅ Features: {len(available_features)}, Samples: {len(X)}")

# Get feature importance
importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': available_features,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# ========================================
# 2. CATEGORY IMPORTANCE
# ========================================
print("\n📊 Generating Category Analysis...")

categories = {
    'Weather': ['temp', 'humidity', 'pressure', 'wind_speed', 'clouds'],
    'Time': ['hour', 'day_of_week', 'day', 'month', 'is_weekend', 'hour_sin', 'hour_cos'],
    'Pollutants': ['pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2'],
    'Lag Features': [f for f in available_features if 'lag' in f.lower()],
    'Rolling Features': [f for f in available_features if 'rolling' in f.lower() or 'std' in f.lower()],
    'Change Features': [f for f in available_features if 'change' in f.lower()]
}

category_importance = {}
for cat, feats in categories.items():
    cat_feats = [f for f in feats if f in available_features]
    if cat_feats:
        cat_imp = importance_df[importance_df['Feature'].isin(cat_feats)]['Importance'].sum()
        category_importance[cat] = cat_imp

cat_df = pd.DataFrame(list(category_importance.items()), columns=['Category', 'Importance'])
cat_df = cat_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f', '#e67e22']
bars = plt.barh(cat_df['Category'], cat_df['Importance'], color=colors[:len(cat_df)])
plt.xlabel('Total Feature Importance')
plt.title('Feature Importance by Category', fontsize=14, fontweight='bold')

total = cat_df['Importance'].sum()
for i, (bar, row) in enumerate(zip(bars, cat_df.itertuples())):
    pct = row.Importance / total * 100
    plt.text(row.Importance + 0.005, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('notebooks/shap_charts/2_category_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 2_category_importance.png")

# ========================================
# 3. FEATURE EFFECT PLOT
# ========================================
print("\n📊 Generating Feature Effect Plot...")

top_feature = importance_df.iloc[0]['Feature']
feature_vals = X[top_feature].values
predictions_24h = model.predict(X)[:, 0]

# Create scatter with trend
plt.figure(figsize=(10, 6))
plt.scatter(feature_vals, predictions_24h, alpha=0.5, color='#3498db', s=30)

# Add trend line
z = np.polyfit(feature_vals, predictions_24h, 1)
p = np.poly1d(z)
x_line = np.linspace(feature_vals.min(), feature_vals.max(), 100)
plt.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend Line')

plt.xlabel(f'{top_feature} Value')
plt.ylabel('Predicted PM2.5 (24h)')
plt.title(f'Effect of {top_feature} on 24h Prediction', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/shap_charts/3_feature_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 3_feature_effect.png")

# ========================================
# 4. PREDICTION DISTRIBUTION
# ========================================
print("\n📊 Generating Prediction Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
horizon_names = ['24h', '48h', '72h']
colors = ['#2ecc71', '#3498db', '#e74c3c']

predictions = model.predict(X)

for i, (ax, name, color) in enumerate(zip(axes, horizon_names, colors)):
    ax.hist(predictions[:, i], bins=30, color=color, alpha=0.7, edgecolor='white')
    mean_val = predictions[:, i].mean()
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.set_xlabel('Predicted PM2.5 (μg/m³)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} Prediction Distribution')
    ax.legend()

plt.suptitle('Prediction Distributions Across Horizons', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('notebooks/shap_charts/4_prediction_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 4_prediction_distribution.png")

# ========================================
# 5. HORIZON COMPARISON
# ========================================
print("\n📊 Generating Horizon Comparison...")

# Get top 10 features
top10 = importance_df.head(10)

plt.figure(figsize=(12, 6))
x = np.arange(len(top10))
bars = plt.bar(x, top10['Importance'], color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top10))))
plt.xticks(x, top10['Feature'], rotation=45, ha='right')
plt.ylabel('Feature Importance')
plt.title('Top 10 Most Important Features for AQI Prediction', fontsize=14, fontweight='bold')

for bar, imp in zip(bars, top10['Importance']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{imp*100:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('notebooks/shap_charts/5_top10_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 5_top10_importance.png")

# ========================================
# SUMMARY
# ========================================
print("\n" + "=" * 60)
print("🎯 KEY INSIGHTS")
print("=" * 60)

print("\n📊 Top 5 Features:")
for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']} ({row['Importance']*100:.1f}%)")

print(f"\n📊 Category Breakdown:")
for cat, imp in sorted(category_importance.items(), key=lambda x: -x[1]):
    pct = imp / sum(category_importance.values()) * 100
    print(f"   {cat}: {pct:.1f}%")

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
print("📁 Charts: notebooks/shap_charts/")
print("=" * 60)
