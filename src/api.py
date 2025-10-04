"""
FastAPI REST API for Air Quality Forecasting
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Air Quality Forecasting API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )
    

# Global variables for model and data
forecaster = None
preprocessor = None
latest_data = None


class ForecastRequest(BaseModel):
    hours: int = 24
    lat: Optional[float] = None
    lon: Optional[float] = None


class ForecastResponse(BaseModel):
    timestamp: str
    hours_ahead: int
    predicted_aqi: float
    aqi_category: str
    health_message: str


class CurrentAQIResponse(BaseModel):
    timestamp: str
    aqi: float
    aqi_category: str
    pollutants: Dict[str, float]
    weather: Dict[str, float]


class ModelMetricsResponse(BaseModel):
    model_name: str
    rmse: float
    mae: float
    r2: float
    feature_importance: Dict[str, float]


def get_aqi_category(aqi: float) -> str:
    """Convert AQI value to category"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_health_message(aqi: float) -> str:
    """Get health message based on AQI"""
    if aqi <= 50:
        return "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi <= 150:
        return "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi <= 200:
        return "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi <= 300:
        return "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Health warning of emergency conditions: everyone is more likely to be affected."


@app.on_event("startup")
async def startup_event():
    """Initialize model and data on startup"""
    global forecaster, preprocessor, latest_data
    
    from src.model import AirQualityForecaster
    from src.preprocessor import AirQualityPreprocessor
    import yaml
    import os
    import pandas as pd
    from dotenv import load_dotenv

    # Load .env if present so env vars can be specified in a file
    load_dotenv()
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    forecaster = AirQualityForecaster(config)
    preprocessor = AirQualityPreprocessor(config)
    # Try to load existing model from configurable path
    model_path = os.getenv('MODEL_PATH', './data/models/best_model.pkl')
    # If MODEL_URL is provided and model file does not exist, try to download it
    model_url = os.getenv('MODEL_URL')
    if model_url and not os.path.exists(model_path):
        try:
            import requests
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info(f"Downloading model from {model_url} to {model_path}")
            r = requests.get(model_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download model from MODEL_URL: {e}")
    if os.path.exists(model_path):
        try:
            forecaster.load_model(model_path)
            logger.info(f"Loaded existing model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    else:
        logger.warning(f"Model not found at {model_path}. Start training and save a model to this path.")

    # Optionally load latest data into memory for the API to serve
    latest_data_path = os.getenv('LATEST_DATA_PATH', './data/processed/processed_data.csv')
    # If LATEST_DATA_URL is provided and data file does not exist, try to download it
    latest_data_url = os.getenv('LATEST_DATA_URL')
    if latest_data_url and not os.path.exists(latest_data_path):
        try:
            import requests
            os.makedirs(os.path.dirname(latest_data_path), exist_ok=True)
            logger.info(f"Downloading latest data from {latest_data_url} to {latest_data_path}")
            r = requests.get(latest_data_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(latest_data_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Latest data downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download latest data from LATEST_DATA_URL: {e}")
    if os.path.exists(latest_data_path):
        try:
            latest_data = pd.read_csv(latest_data_path, parse_dates=['timestamp'])
            logger.info(f"Loaded latest data from {latest_data_path} with {len(latest_data)} records")
        except Exception as e:
            logger.error(f"Failed to load latest data from {latest_data_path}: {e}")
    else:
        logger.warning(f"Latest data not found at {latest_data_path}. API will return 503 for data endpoints until updated.")
    
    logger.info("API started successfully")


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "message": "Air Quality Forecasting API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/current", response_model=CurrentAQIResponse)
async def get_current_aqi():
    """Get current air quality conditions"""
    
    if latest_data is None or latest_data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    current = latest_data.iloc[-1]
    
    return CurrentAQIResponse(
        timestamp=current['timestamp'].isoformat(),
        aqi=float(current['aqi']),
        aqi_category=get_aqi_category(current['aqi']),
        pollutants={
            'no2': float(current.get('no2', 0)),
            'hcho': float(current.get('hcho', 0)),
            'pm25': float(current.get('pm25', 0))
        },
        weather={
            'temperature': float(current.get('temperature', 0)),
            'wind_speed': float(current.get('wind_speed', 0)),
            'humidity': float(current.get('humidity', 0))
        }
    )


@app.post("/forecast", response_model=List[ForecastResponse])
async def get_forecast(request: ForecastRequest):
    """Get air quality forecast for next N hours"""
    
    if forecaster is None or forecaster.best_model_name is None:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    if latest_data is None or latest_data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    try:
        # Generate forecast
        forecast_df = forecaster.forecast_next_hours(latest_data, hours=request.hours)
        
        # Format response
        forecasts = []
        for _, row in forecast_df.iterrows():
            aqi = row['predicted_aqi']
            forecasts.append(ForecastResponse(
                timestamp=row['timestamp'].isoformat(),
                hours_ahead=int(row['hours_ahead']),
                predicted_aqi=float(aqi),
                aqi_category=get_aqi_category(aqi),
                health_message=get_health_message(aqi)
            ))
        
        return forecasts
    
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical")
async def get_historical_data(days: int = 7):
    """Get historical air quality data"""
    
    if latest_data is None or latest_data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    # Get last N days
    cutoff_date = datetime.now() - timedelta(days=days)
    historical = latest_data[latest_data['timestamp'] >= cutoff_date]
    
    return {
        "data": historical[['timestamp', 'aqi', 'no2', 'hcho', 'pm25']].to_dict('records'),
        "count": len(historical)
    }


@app.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Get model performance metrics"""
    
    if forecaster is None or forecaster.best_model_name is None:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    # Load saved metrics (you'll need to save these during training)
    try:
        import json
        with open('./data/models/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        best_model_metrics = metrics[forecaster.best_model_name]
        
        return ModelMetricsResponse(
            model_name=forecaster.best_model_name,
            rmse=best_model_metrics['rmse'],
            mae=best_model_metrics['mae'],
            r2=best_model_metrics['r2'],
            feature_importance=forecaster.get_feature_importance(top_n=10)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics not available: {str(e)}")


@app.get("/alerts")
async def get_alerts(threshold: int = 100):
    """Get alerts when AQI exceeds threshold"""
    
    if forecaster is None or latest_data is None:
        raise HTTPException(status_code=503, detail="Service not available")
    
    # Get forecast
    forecast_df = forecaster.forecast_next_hours(latest_data, hours=24)
    
    # Find hours where AQI exceeds threshold
    alerts = []
    for _, row in forecast_df.iterrows():
        if row['predicted_aqi'] > threshold:
            alerts.append({
                'timestamp': row['timestamp'].isoformat(),
                'hours_ahead': int(row['hours_ahead']),
                'predicted_aqi': float(row['predicted_aqi']),
                'category': get_aqi_category(row['predicted_aqi']),
                'message': get_health_message(row['predicted_aqi'])
            })
    
    return {
        'alert_count': len(alerts),
        'threshold': threshold,
        'alerts': alerts
    }


@app.get("/health-recommendations")
async def get_health_recommendations():
    """Get personalized health recommendations based on current and forecast AQI"""
    
    if latest_data is None or forecaster is None:
        raise HTTPException(status_code=503, detail="Service not available")
    
    current_aqi = latest_data.iloc[-1]['aqi']
    forecast_df = forecaster.forecast_next_hours(latest_data, hours=12)
    max_forecast_aqi = forecast_df['predicted_aqi'].max()
    
    recommendations = {
        'current_aqi': float(current_aqi),
        'max_forecast_aqi': float(max_forecast_aqi),
        'recommendations': []
    }
    
    if max_forecast_aqi > 150:
        recommendations['recommendations'].extend([
            "Avoid prolonged outdoor activities",
            "Keep windows closed",
            "Use air purifiers indoors",
            "Wear N95 masks if going outside"
        ])
    elif max_forecast_aqi > 100:
        recommendations['recommendations'].extend([
            "Limit prolonged outdoor exertion",
            "Sensitive groups should reduce outdoor activities",
            "Consider indoor exercise alternatives"
        ])
    else:
        recommendations['recommendations'].append("Air quality is acceptable for outdoor activities")
    
    return recommendations


# Helper function to update data (call this from main training script)
def update_latest_data(data: pd.DataFrame):
    """Update the latest data for API"""
    global latest_data
    latest_data = data
    logger.info(f"Updated latest data: {len(data)} records")