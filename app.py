"""
AQI Predictor Dashboard - Streamlit web app for real-time AQI monitoring and 3-day forecast.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.database import Database

# Page config
st.set_page_config(
    page_title="🌫️ Faisalabad AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .aqi-good { background: linear-gradient(135deg, #00c853 0%, #69f0ae 100%); }
    .aqi-moderate { background: linear-gradient(135deg, #ffd600 0%, #ffff00 100%); color: #333; }
    .aqi-sensitive { background: linear-gradient(135deg, #ff9100 0%, #ffab40 100%); }
    .aqi-unhealthy { background: linear-gradient(135deg, #ff1744 0%, #ff5252 100%); }
    .aqi-very-unhealthy { background: linear-gradient(135deg, #7b1fa2 0%, #ab47bc 100%); }
    .aqi-hazardous { background: linear-gradient(135deg, #880e4f 0%, #c51162 100%); }
</style>
""", unsafe_allow_html=True)


def get_aqi_category(aqi):
    """Get AQI category, emoji, color, and health message."""
    if aqi <= 50:
        return "Good", "🟢", "#00c853", "Air quality is satisfactory"
    elif aqi <= 100:
        return "Moderate", "🟡", "#ffd600", "Acceptable for most people"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#ff9100", "Sensitive groups may experience effects"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ff1744", "Everyone may experience health effects"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#7b1fa2", "Health alert: serious effects possible"
    else:
        return "Hazardous", "⚫", "#880e4f", "Emergency conditions"


def pm25_to_aqi(pm25):
    """Convert PM2.5 concentration (μg/m³) to AQI using EPA breakpoints."""
    breakpoints = [
        (0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    
    if pm25 < 0:
        return 0
    if pm25 > 500:
        return 500
    
    for pm_low, pm_high, aqi_low, aqi_high in breakpoints:
        if pm_low <= pm25 <= pm_high:
            aqi = ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25 - pm_low) + aqi_low
            return round(aqi)
    
    return 500


@st.cache_resource
def load_model():
    """Load trained model from local file or MongoDB."""
    import pickle
    from src.database import Database
    
    model_path = "models/best_model_multi_output_24h_48h_72h.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    
    old_model_path = "models/best_model_target_24h.pkl"
    if os.path.exists(old_model_path):
        with open(old_model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    
    try:
        db = Database()
        model_doc = db.load_model_binary()
        if model_doc:
            model_data = pickle.loads(model_doc["model_binary"])
            return model_data
    except Exception as e:
        print(f"⚠️ Could not load from MongoDB: {e}")
    
    return None


@st.cache_data(ttl=3600)
def fetch_current_data():
    """Fetch current weather and pollution data (cached 1 hour)."""
    try:
        fetcher = DataFetcher()
        return fetcher.fetch_current_data()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def get_feature_for_prediction(current_data, db):
    """Prepare features for prediction from current data and recent history."""
    recent_features = db.get_latest_features(n=24)
    
    if not recent_features:
        return None
    
    df = pd.DataFrame(recent_features)
    latest = df.iloc[0].to_dict()
    
    if current_data and 'pollution' in current_data:
        latest['aqi_standard'] = current_data['pollution'].get('aqi_standard', latest.get('aqi_standard', 100))
        latest['pm2_5'] = current_data['pollution'].get('pm2_5', latest.get('pm2_5', 50))
        latest['pm10'] = current_data['pollution'].get('pm10', latest.get('pm10', 80))
    
    return latest


def make_prediction(model_data, features):
    """Make multi-horizon PM2.5 predictions and convert to AQI."""
    if model_data is None or features is None:
        return None, None, None, None, None, None
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    scaler = model_data.get('scaler')
    
    X = pd.DataFrame([features])
    
    available_features = [f for f in feature_names if f in X.columns]
    X = X[available_features]
    
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0
    
    X = X[feature_names]
    
    if scaler is not None:
        X = scaler.transform(X)
    
    predictions = model.predict(X)[0]
    
    if hasattr(predictions, '__len__') and len(predictions) == 3:
        pm25_24h, pm25_48h, pm25_72h = predictions
    else:
        pm25_24h = predictions
        pm25_48h = pm25_24h * 1.05
        pm25_72h = pm25_24h * 1.10
    
    aqi_24h = pm25_to_aqi(pm25_24h)
    aqi_48h = pm25_to_aqi(pm25_48h)
    aqi_72h = pm25_to_aqi(pm25_72h)
    
    return pm25_24h, pm25_48h, pm25_72h, aqi_24h, aqi_48h, aqi_72h


def main():
    st.markdown('<h1 class="main-header">🌫️ Faisalabad Air Quality Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time AQI monitoring and 3-day forecast</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    with st.spinner("Fetching latest air quality data..."):
        current_data = fetch_current_data()
    
    # Current Conditions
    st.header("📊 Current Conditions")
    
    if current_data:
        pollution = current_data.get('pollution', {})
        weather = current_data.get('weather', {})
        
        current_pm25 = pollution.get('pm2_5', 0)
        current_aqi = pm25_to_aqi(current_pm25)
        category, emoji, color, health_msg = get_aqi_category(current_aqi)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background: {color}; padding: 2rem; border-radius: 1rem; text-align: center; color: white;">
                <h1 style="font-size: 4rem; margin: 0;">{emoji} {current_aqi:.0f}</h1>
                <h2 style="margin: 0;">{category}</h2>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">{health_msg}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Weather")
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                st.metric("Wind Speed", f"{weather.get('wind_speed', 'N/A')} m/s")
            with wcol2:
                st.metric("Humidity", f"{weather.get('humidity', 'N/A')}%")
                st.metric("Pressure", f"{weather.get('pressure', 'N/A')} hPa")
        
        with col2:
            st.subheader("💨 Pollutants")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.metric("PM2.5", f"{pollution.get('pm2_5', 'N/A'):.1f} μg/m³")
                st.metric("NO₂", f"{pollution.get('no2', 'N/A'):.1f} μg/m³")
            with pcol2:
                st.metric("PM10", f"{pollution.get('pm10', 'N/A'):.1f} μg/m³")
                st.metric("O₃", f"{pollution.get('o3', 'N/A'):.1f} μg/m³")
    else:
        st.error("❌ Could not fetch current data. Please check your API connection.")
    
    st.markdown("---")
    
    # 3-Day Forecast
    st.header("📈 3-Day AQI Forecast")
    
    model_data = load_model()
    
    if model_data:
        db = Database()
        
        current_pm25 = current_data['pollution']['pm2_5'] if current_data else 100
        current_aqi = pm25_to_aqi(current_pm25)
        
        features = get_feature_for_prediction(current_data, db)
        result = make_prediction(model_data, features)
        pm25_24h, pm25_48h, pm25_72h, aqi_24h, aqi_48h, aqi_72h = result
        
        if pm25_24h is not None:
            now = datetime.now()
            
            forecast_data = [
                {'hours': 1, 'pm25': current_pm25 + (pm25_24h - current_pm25) * (1/24), 'type': 'interpolated'},
                {'hours': 6, 'pm25': current_pm25 + (pm25_24h - current_pm25) * (6/24), 'type': 'interpolated'},
                {'hours': 12, 'pm25': current_pm25 + (pm25_24h - current_pm25) * (12/24), 'type': 'interpolated'},
                {'hours': 24, 'pm25': pm25_24h, 'type': 'ML predicted ✓'},
                {'hours': 48, 'pm25': pm25_48h, 'type': 'ML predicted ✓'},
                {'hours': 72, 'pm25': pm25_72h, 'type': 'ML predicted ✓'},
            ]
            
            forecast_hours = [d['hours'] for d in forecast_data]
            forecast_pm25 = [d['pm25'] for d in forecast_data]
            forecast_values = [pm25_to_aqi(max(0, pm)) for pm in forecast_pm25]
            forecast_types = [d['type'] for d in forecast_data]
        else:
            st.warning("⚠️ Model prediction unavailable. Showing estimated values.")
            forecast_hours = [1, 6, 12, 24, 48, 72]
            forecast_values = [current_aqi] * 6
            forecast_pm25 = [current_pm25] * 6
            forecast_types = ['estimated'] * 6
        
        now = datetime.now()
        forecast_df = pd.DataFrame({
            'Hours Ahead': forecast_hours,
            'Time': [now + timedelta(hours=h) for h in forecast_hours],
            'Predicted AQI': forecast_values,
            'Predicted PM2.5': forecast_pm25,
            'Prediction Type': forecast_types
        })
        
        # Forecast chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[now], y=[current_aqi],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Current AQI'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Time'], y=forecast_df['Predicted AQI'],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=10),
            name='Predicted AQI'
        ))
        
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
        fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="72-Hour AQI Forecast",
            xaxis_title="Time",
            yaxis_title="AQI",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecast Details")
        
        forecast_display = []
        for _, row in forecast_df.iterrows():
            cat, emoji, _, _ = get_aqi_category(row['Predicted AQI'])
            forecast_display.append({
                'Time': row['Time'].strftime('%b %d, %I:%M %p'),
                'Predicted AQI': f"{row['Predicted AQI']:.0f}",
                'Category': f"{emoji} {cat}"
            })
        
        st.dataframe(pd.DataFrame(forecast_display), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Model not loaded. Run model training first.")
    
    st.markdown("---")
    
    # Health Recommendations
    st.header("💡 Health Recommendations")
    
    if current_data:
        current_aqi = current_data['pollution']['aqi_standard']
        
        if current_aqi <= 50:
            st.success("✅ **Air quality is good!** Great day for outdoor activities.")
        elif current_aqi <= 100:
            st.info("🟡 **Moderate** — Sensitive people should consider reducing prolonged outdoor exertion.")
        elif current_aqi <= 150:
            st.warning("🟠 **Unhealthy for Sensitive Groups** — People with respiratory/heart conditions should limit outdoor activity.")
        elif current_aqi <= 200:
            st.error("🔴 **Unhealthy** — Everyone should reduce outdoor exertion. Consider wearing a mask outdoors.")
        else:
            st.error("🟣 **Very Unhealthy / Hazardous** — Stay indoors. Use air purifiers. Wear N95 masks if going outside.")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        Real-time air quality monitoring and ML-based predictions for **Faisalabad, Pakistan**.
        
        **Data:** OpenWeather API  
        **Model:** Multi-output Random Forest  
        **Updates:** Hourly data collection, daily retraining
        
        **AQI Scale:**
        - 🟢 0-50: Good
        - 🟡 51-100: Moderate
        - 🟠 101-150: Unhealthy (Sensitive)
        - 🔴 151-200: Unhealthy
        - 🟣 201-300: Very Unhealthy
        - ⚫ 301+: Hazardous
        """)
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
