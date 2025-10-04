"""
Main script to orchestrate data fetching, training, and API launch
"""
import os
import yaml
import logging
import pandas as pd
import json
from datetime import datetime, timedelta

from src.data_fetcher import NASADataFetcher
from src.preprocessor import AirQualityPreprocessor
from src.model import AirQualityForecaster
from src.api import update_latest_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    dirs = ['data/raw', 'data/processed', 'data/models', 'notebooks', 'tests']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Directories created")


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def fetch_all_data(config):
    """Fetch data from all sources"""
    logger.info("Starting data fetch...")
    
    fetcher = NASADataFetcher(config)
    
    # Authenticate with NASA (interactive)
    try:
        fetcher.authenticate()
    except:
        logger.warning("NASA authentication failed, using synthetic data")
    
    # Define date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=config['data']['time_range']['history_days'])).strftime('%Y-%m-%d')
    
    # Fetch data
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    tempo_df = fetcher.fetch_tempo_data(start_date, end_date)
    weather_df = fetcher.fetch_weather_data(start_date, end_date)
    ground_df = fetcher.fetch_ground_truth(start_date, end_date)
    
    # Save raw data
    tempo_df.to_csv('./data/raw/tempo_data.csv', index=False)
    weather_df.to_csv('./data/raw/weather_data.csv', index=False)
    ground_df.to_csv('./data/raw/ground_data.csv', index=False)
    
    logger.info("Data fetch complete")
    
    return tempo_df, weather_df, ground_df


def preprocess_data(config, tempo_df, weather_df, ground_df):
    """Preprocess and engineer features"""
    logger.info("Starting data preprocessing...")
    
    preprocessor = AirQualityPreprocessor(config)
    
    # Merge datasets
    merged_df = preprocessor.merge_datasets(tempo_df, weather_df, ground_df)
    
    # Calculate AQI
    merged_df = preprocessor.calculate_aqi(merged_df)
    
    # Engineer features
    processed_df = preprocessor.engineer_features(merged_df)
    
    # Save processed data
    processed_df.to_csv('./data/processed/processed_data.csv', index=False)
    
    logger.info("Data preprocessing complete")
    
    return processed_df, preprocessor


def train_models(config, processed_df):
    """Train and evaluate models"""
    logger.info("Starting model training...")
    
    preprocessor = AirQualityPreprocessor(config)
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_train_test(processed_df)
    
    # Initialize forecaster
    forecaster = AirQualityForecaster(config)
    
    # Train models
    training_results = forecaster.train(X_train, y_train)
    
    # Evaluate models
    evaluation_results = forecaster.evaluate(X_test, y_test)
    
    # Save model
    forecaster.save_model('./data/models/best_model.pkl')
    
    # Save metrics
    metrics = {
        model_name: {
            'rmse': evaluation_results[model_name]['rmse'],
            'mae': evaluation_results[model_name]['mae'],
            'r2': evaluation_results[model_name]['r2']
        }
        for model_name in evaluation_results.keys()
    }
    
    with open('./data/models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Model training complete")
    
    return forecaster, evaluation_results


def generate_sample_forecast(forecaster, processed_df):
    """Generate and save sample forecast"""
    logger.info("Generating sample forecast...")
    
    forecast_df = forecaster.forecast_next_hours(processed_df, hours=24)
    forecast_df.to_csv('./data/processed/sample_forecast.csv', index=False)
    
    logger.info("Sample forecast generated")
    logger.info(f"\n{forecast_df.head(10)}")
    
    return forecast_df


def print_summary(evaluation_results, forecaster):
    """Print training summary"""
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  R²:   {metrics['r2']:.3f}")
    
    print(f"\nBest Model: {forecaster.best_model_name}")
    
    print("\nTop 10 Feature Importance:")
    importance = forecaster.get_feature_importance(top_n=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"  {i:2d}. {feature:30s}: {score:.4f}")
    
    print("\n" + "="*60)


def main():
    """Main execution pipeline"""
    
    print("="*60)
    print("AIR QUALITY FORECASTING - ML MODEL TRAINING")
    print("="*60)
    
    # Setup
    create_directories()
    config = load_config()
    
    # Step 1: Fetch Data
    print("\n[1/5] Fetching data from NASA and ground stations...")
    tempo_df, weather_df, ground_df = fetch_all_data(config)
    
    # Step 2: Preprocess Data
    print("\n[2/5] Preprocessing and feature engineering...")
    processed_df, preprocessor = preprocess_data(config, tempo_df, weather_df, ground_df)
    
    # Step 3: Train Models
    print("\n[3/5] Training machine learning models...")
    forecaster, evaluation_results = train_models(config, processed_df)
    
    # Step 4: Generate Sample Forecast
    print("\n[4/5] Generating sample forecast...")
    forecast_df = generate_sample_forecast(forecaster, processed_df)
    
    # Step 5: Update API data
    print("\n[5/5] Updating API with latest data...")
    update_latest_data(processed_df)
    
    # Print summary
    print_summary(evaluation_results, forecaster)
    
    print("\n✅ Training complete! Model saved to ./data/models/best_model.pkl")
    print("\nTo start the API server, run:")
    print("  uvicorn src.api:app --reload")
    print("\nAPI will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")


if __name__ == "__main__":
    main()