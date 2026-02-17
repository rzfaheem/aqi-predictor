"""
Model Explainability - Feature importance analysis using model coefficients or tree importances.
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database


def run_explainability_analysis():
    """Analyze and visualize feature importance from the trained model."""
    print("\n" + "=" * 50)
    print("MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 50)
    
    model_path = "models/best_model_target_24h.pkl"
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data['feature_names']
    model_name = model_data['model_name']
    
    print(f"Loaded {model_name} model with {len(feature_names)} features")
    
    db = Database()
    features = db.get_features()
    
    if not features:
        print("No features found!")
        return
    
    df = pd.DataFrame(features)
    X = df[[col for col in feature_names if col in df.columns]].copy()
    X = X.ffill().bfill().fillna(0)
    
    output_dir = "notebooks/shap_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract feature importance based on model type
    if hasattr(model, 'coef_'):
        coefficients = np.abs(model.coef_)
        if scaler is not None:
            importance = coefficients * scaler.scale_
        else:
            importance = coefficients
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print("Cannot extract feature importance from this model type!")
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    total_imp = importance_df['importance'].sum()
    importance_df['percentage'] = 100 * importance_df['importance'] / total_imp
    
    # Print top features
    print("\n" + "=" * 50)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 50)
    print(f"{'Rank':<5} {'Feature':<30} {'Importance':<12} {'% of Total':<10}")
    print("-" * 57)
    
    for rank, (idx, row) in enumerate(importance_df.head(15).iterrows(), 1):
        bar_width = row['percentage'] / 2 if pd.notna(row['percentage']) else 0
        bar = "#" * int(bar_width) if bar_width > 0 else "|"
        print(f"{rank:<5} {row['feature']:<30} {row['importance']:<12.4f} {row['percentage']:<6.1f}%  {bar}")
    
    # Plot 1: Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(12, 10))
    top_n = 15
    top_features = importance_df.head(top_n)
    colors = plt.cm.Blues(np.linspace(0.9, 0.4, top_n))
    
    y_pos = range(top_n)
    bars = ax.barh(y_pos, top_features['importance'].values[::-1], color=colors[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'].values[::-1], fontsize=11)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name} Model', fontsize=14, fontweight='bold')
    
    for bar, pct in zip(bars, top_features['percentage'].values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Feature Categories Pie Chart
    categories = {
        'Pollution': ['pm', 'aqi', 'no2', 'o3', 'co', 'so2'],
        'Weather': ['temp', 'humid', 'wind', 'pressure', 'cloud', 'feels'],
        'Time': ['hour', 'day', 'month', 'weekend', 'sin', 'cos'],
        'Lag/Rolling': ['lag', 'rolling', 'std', 'change', 'pct']
    }
    
    category_importance = {}
    for cat, keywords in categories.items():
        cat_features = [f for f in importance_df['feature'] if any(k in f.lower() for k in keywords)]
        category_importance[cat] = importance_df[importance_df['feature'].isin(cat_features)]['percentage'].sum()
    
    assigned = sum(category_importance.values())
    category_importance['Other'] = 100 - assigned if assigned < 100 else 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = list(category_importance.keys())
    sizes = list(category_importance.values())
    colors_pie = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#95a5a6']
    explode = (0.05, 0.02, 0.02, 0.02, 0.02)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    ax.set_title('Feature Category Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_category_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Impact Direction (linear models only)
    if hasattr(model, 'coef_'):
        signed_coefs = model.coef_
        if scaler is not None:
            signed_coefs = signed_coefs * scaler.scale_
        
        coef_df = pd.DataFrame({
            'feature': feature_names[:len(signed_coefs)],
            'coefficient': signed_coefs
        }).sort_values('coefficient', key=abs, ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors_dir = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['coefficient'].values[::-1]]
        
        ax.barh(range(len(coef_df)), coef_df['coefficient'].values[::-1], color=colors_dir)
        ax.set_yticks(range(len(coef_df)))
        ax.set_yticklabels(coef_df['feature'].values[::-1], fontsize=11)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title('Feature Impact Direction', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.plot([], [], 's', color='#e74c3c', label='Increases AQI')
        ax.plot([], [], 's', color='#3498db', label='Decreases AQI')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_impact_direction.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Summary
    print(f"\nMost Important: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['percentage']:.1f}%)")
    for cat, pct in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
        if pct > 0:
            print(f"   {cat}: {pct:.1f}%")
    
    print(f"\nCharts saved in: {output_dir}/")
    print("Explainability analysis complete!")


if __name__ == "__main__":
    run_explainability_analysis()
