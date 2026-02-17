"""
Model Training - Trains and evaluates multiple ML models for AQI prediction.

Supports Ridge Regression, Random Forest, and XGBoost with multi-output
prediction for 24h, 48h, and 72h forecast horizons.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.database import Database


def load_features():
    """Load features from MongoDB feature store."""
    print("Loading features from MongoDB...")
    
    db = Database()
    features = db.get_features()
    
    if not features:
        print("No features found!")
        return None
    
    df = pd.DataFrame(features)
    print(f"Loaded {len(df)} feature records")
    return df


def prepare_data(df, target_cols=['target_24h', 'target_48h', 'target_72h']):
    """Prepare X (features) and y (multi-output targets) for training."""
    print(f"\nPreparing data for targets: {target_cols}")
    
    exclude_cols = ['_id', 'timestamp', 'saved_at', 
                    'target_1h', 'target_6h', 'target_12h', 
                    'target_24h', 'target_48h', 'target_72h']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    X = X.ffill().bfill()
    
    valid_idx = ~y.isna().any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"   Features: {len(feature_cols)}, Samples: {len(X)}")
    return X, y, feature_cols


def split_data(X, y, test_size=0.2):
    """Split data into train/test sets with random sampling."""
    print(f"\nSplitting data (test_size={test_size})...")
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Standardize features (mean=0, std=1). Fitted on training data only."""
    print("\nScaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(y_true, y_pred, model_name, target_names=['24h', '48h', '72h']):
    """Calculate RMSE, MAE, and R2 for each prediction horizon."""
    print(f"\n{model_name} Results:")
    
    metrics_per_target = {}
    for i, name in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
        
        metrics_per_target[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        print(f"   {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
    
    avg_rmse = np.mean([m['rmse'] for m in metrics_per_target.values()])
    avg_mae = np.mean([m['mae'] for m in metrics_per_target.values()])
    avg_r2 = np.mean([m['r2'] for m in metrics_per_target.values()])
    
    print(f"   ---------------------------------")
    print(f"   AVERAGE: RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}, R2={avg_r2:.4f}")
    
    return {
        'rmse': avg_rmse, 
        'mae': avg_mae, 
        'r2': avg_r2,
        'per_horizon': metrics_per_target
    }


def train_ridge(X_train, X_test, y_train, y_test):
    """Train Ridge Regression (multi-output, native support)."""
    print("\n" + "=" * 40)
    print("Training Ridge Regression...")
    print("=" * 40)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Ridge Regression")
    
    return model, metrics, y_pred


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest (multi-output, native support)."""
    print("\n" + "=" * 40)
    print("Training Random Forest...")
    print("=" * 40)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    
    return model, metrics, y_pred


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost (wrapped with MultiOutputRegressor)."""
    if not XGBOOST_AVAILABLE:
        return None, None, None
    
    print("\n" + "=" * 40)
    print("Training XGBoost...")
    print("=" * 40)
    
    base_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    
    return model, metrics, y_pred


def select_best_model(results):
    """Select best model by lowest average RMSE."""
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    
    valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}
    
    if not valid_results:
        print("No valid models!")
        return None, None
    
    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R2':<10}")
    print("-" * 50)
    
    for name, data in valid_results.items():
        m = data['metrics']
        print(f"{name:<20} {m['rmse']:<10.2f} {m['mae']:<10.2f} {m['r2']:<10.4f}")
    
    best_name = min(valid_results.keys(), key=lambda x: valid_results[x]['metrics']['rmse'])
    best_data = valid_results[best_name]
    
    print(f"\nBest Model: {best_name} (RMSE: {best_data['metrics']['rmse']:.2f})")
    return best_name, best_data


def save_model(model, scaler, feature_names, target_name, metrics, model_name):
    """Save best model locally and to MongoDB."""
    print("\nSaving model...")
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/best_model_{target_name}.pkl"
    
    model_dict = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_name': target_name,
        'model_name': model_name,
        'metrics': metrics,
        'trained_at': datetime.utcnow()
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"   Model saved to: {model_path}")
    
    # Save to MongoDB for cloud deployment
    db = Database()
    model_binary = pickle.dumps(model_dict)
    db.save_model_binary(model_binary, model_name, metrics, feature_names)
    
    # Register metadata
    model_info = {
        'model_name': model_name,
        'model_path': model_path,
        'target_name': target_name,
        'feature_names': feature_names,
        'metrics': metrics,
        'trained_at': datetime.utcnow()
    }
    db.save_model_info(model_info)
    
    return model_path


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    df = load_features()
    if df is None:
        exit(1)
    
    target_cols = ['target_24h', 'target_48h', 'target_72h']
    X, y, feature_names = prepare_data(df, target_cols=target_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train all models
    results = {}
    
    ridge_model, ridge_metrics, ridge_pred = train_ridge(X_train_scaled, X_test_scaled, y_train, y_test)
    results['Ridge'] = {'model': ridge_model, 'metrics': ridge_metrics, 'pred': ridge_pred}
    
    rf_model, rf_metrics, rf_pred = train_random_forest(X_train, X_test, y_train, y_test)
    results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics, 'pred': rf_pred}
    
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_metrics, xgb_pred = train_xgboost(X_train, X_test, y_train, y_test)
        results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics, 'pred': xgb_pred}
    
    best_name, best_data = select_best_model(results)
    if best_name is None:
        exit(1)
    
    model_filename = 'multi_output_24h_48h_72h'
    
    if best_name == 'Ridge':
        save_model(best_data['model'], scaler, feature_names, model_filename, best_data['metrics'], best_name)
    else:
        save_model(best_data['model'], None, feature_names, model_filename, best_data['metrics'], best_name)
    
    print(f"\nBest Model: {best_name}")
    print(f"Avg RMSE: {best_data['metrics']['rmse']:.2f}, Avg R2: {best_data['metrics']['r2']:.4f}")
    
    if 'per_horizon' in best_data['metrics']:
        for horizon, m in best_data['metrics']['per_horizon'].items():
            print(f"   {horizon}: RMSE={m['rmse']:.2f}, R2={m['r2']:.4f}")
    
    print("\nTraining complete!")
