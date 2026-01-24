"""
Model Training Script
=====================

WHAT THIS FILE DOES:
1. Loads features from MongoDB Feature Store
2. Splits data into training and testing sets
3. Trains multiple ML models
4. Evaluates each model
5. Selects the best model
6. Saves the best model to MongoDB Model Registry

MODELS WE'LL TRY:

1. Ridge Regression (Simple linear model)
   - Fast and simple
   - Good baseline to compare against

2. Random Forest (Ensemble of decision trees)
   - Very popular for tabular data
   - Handles non-linear relationships
   - Less prone to overfitting

3. XGBoost (Gradient Boosting)
   - Often the best performer
   - Learns from mistakes iteratively
   - Used by many Kaggle winners

EVALUATION METRICS:

- RMSE (Root Mean Square Error): Average prediction error (lower is better)
- MAE (Mean Absolute Error): Average absolute error (lower is better)
- R² (R-squared): How much variance the model explains (higher is better, max=1)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Will use other models.")

from src.database import Database


def load_features():
    """
    Load features from MongoDB Feature Store.
    """
    print("📊 Loading features from MongoDB...")
    
    db = Database()
    features = db.get_features()
    
    if not features:
        # Try loading from CSV as fallback
        csv_path = "notebooks/features.csv"
        if os.path.exists(csv_path):
            print("   Loading from CSV instead...")
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        print("❌ No features found!")
        return None
    
    df = pd.DataFrame(features)
    print(f"✅ Loaded {len(df)} feature records")
    return df


def prepare_data(df, target_col='target_24h'):
    """
    Prepare data for model training.
    
    STEPS:
    1. Select feature columns
    2. Remove rows with NaN
    3. Split into X (features) and y (target)
    """
    print(f"\n🎯 Preparing data for target: {target_col}")
    
    # Feature columns (exclude targets and metadata)
    exclude_cols = ['_id', 'timestamp', 'saved_at', 
                    'target_1h', 'target_6h', 'target_12h', 
                    'target_24h', 'target_48h', 'target_72h']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create copies
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any remaining NaN
    X = X.fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows where target is NaN
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    
    return X, y, feature_cols


def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    
    WHY SPLIT?
    - Training set: Used to TRAIN the model
    - Testing set: Used to EVALUATE the model on unseen data
    - This prevents overfitting (model memorizing training data)
    
    IMPORTANT: For time series, we don't shuffle!
    We train on past data and test on future data.
    """
    print(f"\n✂️ Splitting data (test_size={test_size})...")
    
    # For time series: Use the last 20% as test (no shuffling!)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features to similar ranges.
    
    WHY SCALE?
    - Some features are large (e.g., AQI: 0-500)
    - Some features are small (e.g., hour: 0-23)
    - Scaling helps some algorithms work better
    
    StandardScaler: Transforms data to have mean=0, std=1
    """
    print("\n📏 Scaling features...")
    
    scaler = StandardScaler()
    
    # Fit on training data ONLY (to prevent data leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("   ✅ Features scaled")
    
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate evaluation metrics for a model.
    
    METRICS:
    - RMSE: Root Mean Square Error (penalizes large errors more)
    - MAE: Mean Absolute Error (average error)
    - R²: Coefficient of determination (how much variance explained)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 {model_name} Results:")
    print(f"   RMSE: {rmse:.2f} (lower is better)")
    print(f"   MAE:  {mae:.2f} (lower is better)")
    print(f"   R²:   {r2:.4f} (higher is better, max=1)")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_ridge(X_train, X_test, y_train, y_test):
    """
    Train Ridge Regression model.
    
    RIDGE REGRESSION:
    - Linear model with regularization
    - Fast and simple baseline
    - Works well when features are correlated
    """
    print("\n" + "=" * 40)
    print("🔵 Training Ridge Regression...")
    print("=" * 40)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Ridge Regression")
    
    return model, metrics, y_pred


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest model.
    
    RANDOM FOREST:
    - Ensemble of many decision trees
    - Each tree votes, majority wins
    - Very robust and handles non-linear relationships
    """
    print("\n" + "=" * 40)
    print("🌲 Training Random Forest...")
    print("=" * 40)
    
    model = RandomForestRegressor(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Maximum depth of each tree
        min_samples_split=5,   # Minimum samples to split a node
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    
    return model, metrics, y_pred


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train XGBoost model.
    
    XGBOOST:
    - Gradient Boosting algorithm
    - Builds trees sequentially, each fixing previous errors
    - Often the best performer on tabular data
    """
    if not XGBOOST_AVAILABLE:
        return None, None, None
    
    print("\n" + "=" * 40)
    print("🚀 Training XGBoost...")
    print("=" * 40)
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    
    return model, metrics, y_pred


def select_best_model(results):
    """
    Select the best model based on RMSE.
    
    We choose the model with LOWEST RMSE (smallest prediction error).
    """
    print("\n" + "=" * 50)
    print("🏆 MODEL COMPARISON")
    print("=" * 50)
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}
    
    if not valid_results:
        print("❌ No valid models!")
        return None, None
    
    # Print comparison table
    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)
    
    for name, data in valid_results.items():
        m = data['metrics']
        print(f"{name:<20} {m['rmse']:<10.2f} {m['mae']:<10.2f} {m['r2']:<10.4f}")
    
    # Find best model (lowest RMSE)
    best_name = min(valid_results.keys(), key=lambda x: valid_results[x]['metrics']['rmse'])
    best_data = valid_results[best_name]
    
    print(f"\n🥇 Best Model: {best_name}")
    print(f"   RMSE: {best_data['metrics']['rmse']:.2f}")
    
    return best_name, best_data


def save_model(model, scaler, feature_names, target_name, metrics, model_name):
    """
    Save the best model to file and register in MongoDB.
    """
    print("\n💾 Saving model...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save model file
    model_path = f"models/best_model_{target_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'target_name': target_name,
            'model_name': model_name,
            'trained_at': datetime.utcnow()
        }, f)
    print(f"   ✅ Model saved to: {model_path}")
    
    # Register in MongoDB
    db = Database()
    model_info = {
        'model_name': model_name,
        'model_path': model_path,
        'target_name': target_name,
        'feature_names': feature_names,
        'metrics': metrics,
        'trained_at': datetime.utcnow()
    }
    db.save_model_info(model_info)
    print("   ✅ Model registered in MongoDB")
    
    return model_path


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load features
    df = load_features()
    if df is None:
        print("❌ Cannot proceed without features!")
        exit(1)
    
    # Step 2: Train for 24-hour prediction (primary target)
    target = 'target_24h'
    
    # Prepare data
    X, y, feature_names = prepare_data(df, target_col=target)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 3: Train models
    results = {}
    
    # Train Ridge
    ridge_model, ridge_metrics, ridge_pred = train_ridge(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    results['Ridge'] = {'model': ridge_model, 'metrics': ridge_metrics, 'pred': ridge_pred}
    
    # Train Random Forest (doesn't need scaled data, but we'll use it for consistency)
    rf_model, rf_metrics, rf_pred = train_random_forest(
        X_train, X_test, y_train, y_test
    )
    results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics, 'pred': rf_pred}
    
    # Train XGBoost
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_metrics, xgb_pred = train_xgboost(
            X_train, X_test, y_train, y_test
        )
        results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics, 'pred': xgb_pred}
    
    # Step 4: Select best model
    best_name, best_data = select_best_model(results)
    
    if best_name is None:
        print("❌ No models trained successfully!")
        exit(1)
    
    # Step 5: Save best model
    # Use correct model and scaler based on best model
    if best_name == 'Ridge':
        save_model(
            best_data['model'], scaler, feature_names, target,
            best_data['metrics'], best_name
        )
    else:
        # Tree-based models don't need scaler
        save_model(
            best_data['model'], None, feature_names, target,
            best_data['metrics'], best_name
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n🎯 Target: {target} (24-hour ahead prediction)")
    print(f"🏆 Best Model: {best_name}")
    print(f"📈 RMSE: {best_data['metrics']['rmse']:.2f}")
    print(f"📈 R²: {best_data['metrics']['r2']:.4f}")
    print(f"\n✅ Model saved and registered in MongoDB!")
    
    print("\n🎉 Model training complete!")
