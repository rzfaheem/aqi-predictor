# 🌫️ Air Quality Index (AQI) Prediction System
## Comprehensive Project Report

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Machine Learning Approach](#4-machine-learning-approach)
5. [Challenges & Solutions](#5-challenges--solutions)
6. [Results & Metrics](#6-results--metrics)
7. [Technologies Used](#7-technologies-used)
8. [Lessons Learned](#8-lessons-learned)
9. [Future Improvements](#9-future-improvements)
10. [Conclusion](#10-conclusion)

---

## 1. Project Overview

### 1.1 Objective
Build a real-time Air Quality Index (AQI) prediction system for Faisalabad, Pakistan that:
- Fetches live weather and pollution data
- Predicts AQI for the next 3 days (24h, 48h, 72h)
- Displays predictions on an interactive dashboard
- Runs automated data collection and model retraining pipelines

### 1.2 Key Features
- **Real-time Data Fetching**: Current weather and pollution data from OpenWeather API
- **Feature Engineering**: 34+ engineered features including lag, rolling, and time-based features
- **Multi-Output ML Model**: ONE model predicts 24h, 48h, and 72h forecasts simultaneously
- **Interactive Dashboard**: Streamlit-based UI with charts and health recommendations
- **Automated Pipelines**: GitHub Actions for hourly data collection and daily retraining
- **Cloud Database**: MongoDB Atlas for scalable data storage

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  OpenWeather │────▶│  Data Fetcher │────▶│   MongoDB    │
│     API      │     │  (Hourly)    │     │   Atlas      │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Dashboard   │◀────│ ML Model     │◀────│  Feature     │
│  (Streamlit) │     │ (Daily)      │     │  Engineering │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 2.1 Components
| Component | File | Function |
|-----------|------|----------|
| Data Fetcher | `src/data_fetcher.py` | Fetches weather & pollution data from API |
| Database | `src/database.py` | MongoDB operations (save/retrieve) |
| Feature Engineering | `src/feature_engineering.py` | Creates 34+ features for ML |
| Model Training | `src/train_model.py` | Trains multi-output ML models |
| Dashboard | `app.py` | Interactive Streamlit web app |
| Feature Pipeline | `.github/workflows/feature_pipeline.yml` | Runs hourly |
| Training Pipeline | `.github/workflows/training_pipeline.yml` | Runs daily |

---

## 3. Data Pipeline

### 3.1 Data Collection (Hourly)
```
Every Hour:
├── Fetch current weather (temp, humidity, wind, pressure)
├── Fetch current pollution (PM2.5, PM10, NO2, O3, CO, SO2)
├── Save raw data to MongoDB
└── Generate engineered features
```

### 3.2 Features Created
| Category | Features | Count |
|----------|----------|-------|
| Time-based | hour, day_of_week, month, is_weekend, hour_sin, hour_cos | 6 |
| Pollutants | pm2_5, pm10, no2, o3, co, so2 | 6 |
| Weather | temp, humidity, pressure, wind_speed, clouds | 5 |
| Lag Features | aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h | 5 |
| Rolling Features | aqi_rolling_3h, aqi_rolling_6h, aqi_rolling_12h, aqi_rolling_24h | 4 |
| Change Features | aqi_change_1h, aqi_change_3h, temp_change_3h, humidity_change_3h | 4 |
| Interaction | temp_humidity_interaction, wind_pollution_interaction | 2 |
| **Total** | | **34+** |

### 3.3 Target Variables
- `target_24h`: PM2.5 value 24 hours in the future
- `target_48h`: PM2.5 value 48 hours in the future
- `target_72h`: PM2.5 value 72 hours in the future

---

## 4. Machine Learning Approach

### 4.1 Models Trained
| Model | Type | Multi-Output Support |
|-------|------|---------------------|
| Ridge Regression | Linear | Native ✓ |
| Random Forest | Ensemble | Native ✓ |
| XGBoost | Gradient Boosting | Via MultiOutputRegressor |

### 4.2 Multi-Output Model
We use a **single model** that predicts all three time horizons simultaneously:

```python
# One model predicts 3 outputs at once
Input (34 features) → Model → [24h_prediction, 48h_prediction, 72h_prediction]
```

**Benefits:**
- Model learns relationships between 24h, 48h, and 72h patterns
- Predictions are correlated and consistent
- More efficient than training 3 separate models

### 4.3 Cross-Validation
We use **TimeSeriesSplit** with 5 folds to respect temporal order:
```
Fold 1: Train=[1..33],  Test=[34..66]
Fold 2: Train=[1..66],  Test=[67..99]
Fold 3: Train=[1..99],  Test=[100..132]
Fold 4: Train=[1..132], Test=[133..165]
Fold 5: Train=[1..165], Test=[166..198]  ← Final evaluation
```

### 4.4 Evaluation Metrics
- **RMSE** (Root Mean Square Error): Primary metric for model selection
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Variance explained by model

---

## 5. Challenges & Solutions

This section documents ALL problems we encountered during development and how we solved them.

---

### 🔴 Challenge #1: 97% of AQI Values Were Identical

**Problem:**
The OpenWeather API returned AQI as discrete levels (1-5 only). Due to high pollution in Faisalabad, 97% of our data had the same AQI value (level 5 = 250).

```
AQI Value Distribution:
├── AQI = 250: 97% of records
├── AQI = 150: 2% of records
└── AQI = 100: 1% of records
```

**Impact:**
- Model couldn't learn patterns (all targets same)
- Ridge regression had all zero coefficients
- "Perfect" R²=1.0 was misleading (predicting constant value works!)

**Solution:**
Switched from discrete AQI to **continuous PM2.5 (μg/m³)** as the prediction target.

```python
# Before (discrete)
df['target_24h'] = df['aqi_standard'].shift(-24)  # Only 5 possible values

# After (continuous)
df['target_24h'] = df['pm2_5'].shift(-24)  # Infinite possible values
```

**Result:**
- PM2.5 has 709 unique values vs. AQI's 5 values
- Range: 8.1 - 733.6 μg/m³
- Model can now learn meaningful patterns

**Evaluator's Feedback:** "This is a good approach and perfectly acceptable"

---

### 🔴 Challenge #2: GitHub Actions Pipeline Not Running

**Problem:**
The scheduled pipelines (hourly feature collection, daily training) were not triggering automatically on GitHub Actions.

**Symptoms:**
- Cron schedules defined correctly
- Manual triggers worked
- But automatic hourly/daily runs never happened

**Root Cause:**
GitHub Actions requires at least one successful workflow run (via push) to "activate" scheduled triggers for new repositories.

**Solution:**
Added a `push` trigger to the workflow files:

```yaml
# Before
on:
  schedule:
    - cron: '0 * * * *'

# After
on:
  schedule:
    - cron: '0 * * * *'
  push:
    branches: [main]
  workflow_dispatch:
```

**Result:**
- Pushed changes to trigger first run
- Schedules now work automatically

---

### 🔴 Challenge #3: Inappropriate Train/Test Split

**Problem:**
We initially used a random 80/20 train/test split:

```python
X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why This Is Wrong for Time Series:**
- Random split causes **data leakage**
- Future data points end up in training set
- Model "cheats" by seeing future values
- Evaluation metrics are overly optimistic

**Solution:**
Implemented **TimeSeriesSplit** for proper temporal validation:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

**Result:**
- Training only uses past data
- Testing uses future data (as in real deployment)
- More realistic performance estimates

---

### 🔴 Challenge #4: Single 24h Model with 5% Estimation

**Problem:**
We initially trained ONE model for 24h prediction only, then estimated 48h and 72h by adding 5% per day:

```python
# Old approach (wrong)
pm25_48h = pm25_24h * 1.05  # Just adding 5%
pm25_72h = pm25_24h * 1.10  # Just adding 10%
```

**Why This Is Wrong:**
- No actual prediction for Day 2 and Day 3
- 5% is an arbitrary guess
- Doesn't capture real AQI trends (can go up OR down)

**Solution:**
Implemented **Multi-Output Model** that predicts all 3 horizons at once:

```python
# New approach (correct)
target_cols = ['target_24h', 'target_48h', 'target_72h']
y = df[target_cols]  # 3 targets

model = RandomForestRegressor()  # Natively supports multi-output
model.fit(X_train, y_train)

predictions = model.predict(X_new)  # Returns [24h, 48h, 72h]
```

**Result:**
- Each forecast horizon is a real ML prediction
- Model learns actual 48h and 72h patterns from historical data
- More accurate and honest predictions

---

### 🔴 Challenge #5: SHAP Explainability Issues

**Problem:**
Multiple issues when implementing SHAP (SHapley Additive exPlanations) for model interpretability:

**Issue 5.1: Zero Variance Error**
```
Error: "Additivity check failed in TreeExplainer"
```
Cause: Low variance in features (mostly constant values)

**Issue 5.2: NaN Values in Calculations**
```
Error: "RuntimeWarning: invalid value encountered in divide"
```
Cause: Division by zero when calculating importance percentages

**Issue 5.3: KernelExplainer Too Slow**
KernelExplainer took hours for just 100 samples

**Solution:**
Switched from SHAP to **model coefficients** for feature importance:

```python
# For Ridge Regression - use coefficients directly
importance = np.abs(model.coef_)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)
```

**Result:**
- Fast execution
- Reliable results
- Still provides interpretability

---

### 🔴 Challenge #6: Dashboard Showing Random Values

**Problem:**
The 3-day forecast was using random numbers instead of actual predictions:

```python
# Found in app.py (wrong)
variation = np.random.uniform(-10, 10)  # RANDOM!
forecast_val = current_aqi + variation
```

**Impact:**
- Forecast values changed on every page refresh
- No connection to actual ML model
- Completely fake predictions

**Solution:**
Replaced with actual model predictions:

```python
# Fixed approach
pm25_24h, pm25_48h, pm25_72h, aqi_24h, aqi_48h, aqi_72h = make_prediction(model_data, features)
```

**Result:**
- Real ML predictions displayed
- Consistent values
- Honest forecasting

---

### 🔴 Challenge #7: Model Selection Changed After Multi-Output

**Problem:**
When we switched from single-output to multi-output:
- XGBoost WAS the best model (single-output)
- Random Forest BECAME the best model (multi-output)

**Why This Happened:**
```python
# XGBoost doesn't natively support multi-output
# We had to wrap it:
model = MultiOutputRegressor(XGBRegressor())  # Trains 3 separate internal models

# Random Forest natively supports multi-output
model = RandomForestRegressor()  # Learns all 3 together
```

**Impact:**
- Random Forest shares information across horizons
- More efficient learning
- Better correlated predictions

**Resolution:**
Accepted Random Forest as the best model for this multi-output task. This is technically correct and defensible.

---

### 🔴 Challenge #8: Forecast Exploding to AQI 500

**Problem:**
Long-term forecasts (48h, 72h) showed unrealistically high AQI values (up to 500):

```python
# Bug in code
pm25_forecast = base_pm25 * (h / 24) * 0.95

# For 72 hours:
# = base_pm25 * (72/24) * 0.95
# = base_pm25 * 3 * 0.95
# = base_pm25 × 2.85  ← WRONG! Multiplied by nearly 3!
```

**Solution:**
Fixed to use proper multi-output predictions (see Challenge #4)

---

### 🔴 Challenge #9: Current vs Forecast AQI Mismatch

**Problem:**
Dashboard showed different AQI values for current conditions vs. the 1-hour forecast starting point:
- Current Conditions: AQI = 250 (from API)
- Forecast 1h: AQI = 210 (from PM2.5 conversion)

**Root Cause:**
- Current: Using API's discrete AQI value
- Forecast: Using our PM2.5 → AQI conversion

**Solution:**
Updated Current Conditions to also use PM2.5 → AQI conversion:

```python
# Before
current_aqi = pollution.get('aqi_standard', 0)  # API value

# After
current_pm25 = pollution.get('pm2_5', 0)
current_aqi = pm25_to_aqi(current_pm25)  # Consistent conversion
```

**Result:**
Both sections now use the same calculation method

---

### 🔴 Challenge #10: Ridge Model All Zero Coefficients

**Problem:**
After training, Ridge regression showed all coefficients = 0:

```
Feature Coefficients:
hour: 0.0
temp: 0.0
humidity: 0.0
pm2_5: 0.0
... (all zeros)
```

**Root Cause:**
When 97% of targets are the same value (AQI=250), the model learns that "predicting 250 always" is optimal. No feature matters.

**Solution:**
Switching to PM2.5 as target (Challenge #1) resolved this. Now Ridge has 17+ non-zero coefficients.

---

### 🔴 Challenge #11: MongoDB Connection in GitHub Actions

**Problem:**
GitHub Actions couldn't connect to MongoDB Atlas initially.

**Root Cause:**
- MongoDB connection string contains credentials
- Pushing credentials to GitHub is a security risk
- Actions didn't have access to the connection string

**Solution:**
Used **GitHub Secrets** to store sensitive information:

1. Added `MONGO_URI` and `OPENWEATHER_API_KEY` as repository secrets
2. Accessed in workflow via `${{ secrets.MONGO_URI }}`
3. Never exposed credentials in code

---

### 🔴 Challenge #12: PM2.5 to AQI Conversion Accuracy

**Problem:**
Converting PM2.5 concentrations to AQI values required understanding EPA standards.

**Research Required:**
- AQI has 6 categories (Good to Hazardous)
- Each category has PM2.5 breakpoints
- Linear interpolation between breakpoints

**Solution:**
Implemented EPA-standard conversion function:

```python
def pm25_to_aqi(pm25):
    breakpoints = [
        (0, 12.0, 0, 50),      # Good
        (12.1, 35.4, 51, 100), # Moderate
        (35.5, 55.4, 101, 150),# Unhealthy for Sensitive
        (55.5, 150.4, 151, 200),# Unhealthy
        (150.5, 250.4, 201, 300),# Very Unhealthy
        (250.5, 500.4, 301, 500),# Hazardous
    ]
    # Linear interpolation within breakpoint ranges
```

---

## 6. Results & Metrics

### 6.1 Current Model Performance

| Metric | 24h Horizon | 48h Horizon | 72h Horizon | Average |
|--------|-------------|-------------|-------------|---------|
| RMSE | ~X | ~X | ~X | ~X |
| MAE | ~X | ~X | ~X | ~X |
| R² | ~X | ~X | ~X | ~X |

*Note: With limited data (~700 records), model accuracy will improve as more data is collected.*

### 6.2 Best Model
**Random Forest (Multi-Output)**
- Natively supports predicting multiple targets
- Robust to outliers
- Handles non-linear relationships

### 6.3 Dashboard Features
- Real-time AQI display with color coding
- 3-day forecast chart
- Health recommendations based on AQI level
- Current weather conditions
- Pollutant breakdown (PM2.5, PM10, NO2, O3, CO, SO2)

---

## 7. Technologies Used

### 7.1 Programming & Libraries
| Technology | Purpose |
|------------|---------|
| Python 3.10 | Primary programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | ML models, preprocessing, cross-validation |
| XGBoost | Gradient boosting (for comparison) |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| Requests | API calls |

### 7.2 Infrastructure
| Service | Purpose |
|---------|---------|
| MongoDB Atlas | Cloud database |
| GitHub Actions | CI/CD pipelines |
| OpenWeather API | Weather & pollution data |
| GitHub | Version control |

### 7.3 Key Python Packages
```python
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
streamlit>=1.28.0
plotly>=5.18.0
pymongo>=4.5.0
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## 8. Lessons Learned

### 8.1 Data Quality Matters Most
> "Garbage in, garbage out"

The 97% same-AQI problem taught us that model performance is limited by data quality. No algorithm can find patterns that don't exist.

### 8.2 Validation Must Respect Time
For time series forecasting:
- Never use random train/test split
- TimeSeriesSplit prevents data leakage
- Always validate on future data

### 8.3 Understand Your API
The OpenWeather API returns simplified AQI (1-5 scale), not the full 0-500 AQI. Understanding API behavior early would have saved debugging time.

### 8.4 Multi-Output > Multiple Models
Training one multi-output model is often better than training separate models because:
- Predictions are correlated
- Shared learning across horizons
- More efficient

### 8.5 GitHub Actions Require Activation
Scheduled workflows need at least one push-triggered run to start working.

### 8.6 Be Honest About Limitations
With limited data, model accuracy will be modest. It's better to acknowledge this than to hide it or use misleading metrics.

---

## 9. Future Improvements

### 9.1 Short-Term
- [ ] Collect more data (10+ days) to improve model accuracy
- [ ] Add more cities (Lahore, Karachi, Islamabad)
- [ ] Implement model monitoring and alerts

### 9.2 Medium-Term
- [ ] Try deep learning models (LSTM, Transformer)
- [ ] Add seasonal features (Ramadan, Eid, smog season)
- [ ] Incorporate traffic and industrial data

### 9.3 Long-Term
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Mobile app development
- [ ] SMS/WhatsApp alerts for hazardous AQI

---

## 10. Conclusion

This project successfully demonstrates a complete MLOps pipeline for AQI prediction:

✅ **Data Collection**: Automated hourly data fetching from OpenWeather API  
✅ **Feature Engineering**: 34+ engineered features for ML  
✅ **Model Training**: Multi-output Random Forest with TimeSeriesSplit validation  
✅ **Deployment**: Interactive Streamlit dashboard  
✅ **Automation**: GitHub Actions for continuous data collection and retraining  

The project faced multiple challenges, from data quality issues (97% same AQI) to implementation bugs (random forecasts, exploding predictions). Each challenge was documented and resolved, providing valuable learning experiences.

**Key Takeaway**: Building a production ML system is 20% algorithm selection and 80% data engineering, debugging, and infrastructure.

---

## Appendix A: File Structure

```
AQI/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Hourly data collection
│       └── training_pipeline.yml   # Daily model retraining
├── models/
│   └── best_model_multi_output_24h_48h_72h.pkl
├── outputs/
│   └── shap_analysis/             # Feature importance charts
├── src/
│   ├── data_fetcher.py            # API data fetching
│   ├── database.py                # MongoDB operations
│   ├── feature_engineering.py      # Feature creation
│   ├── train_model.py             # Model training
│   ├── eda_analysis.py            # Exploratory analysis
│   └── shap_explainer.py          # Model interpretability
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Dependencies
├── .env                           # Environment variables (not in git)
└── PROJECT_REPORT.md              # This document
```

---

## Appendix B: How to Run

### Local Development
```bash
# Clone repository
git clone https://github.com/rzfaheem/aqi-predictor.git
cd aqi-predictor

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Create .env file with MONGO_URI and OPENWEATHER_API_KEY

# Run dashboard
streamlit run app.py

# Run training manually
python src/train_model.py
```

### Production
Pipelines run automatically via GitHub Actions:
- **Feature Pipeline**: Every hour
- **Training Pipeline**: Every day at midnight

---

*Report generated on: February 3, 2026*  
*Project: AQI Prediction System for Faisalabad, Pakistan*
