"""train.py
Minimal training script that runs the data fetch -> preprocess -> train pipeline
and writes a versioned model artifact and processed data for the API to serve.
"""
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data_fetcher import NASADataFetcher
from src.preprocessor import AirQualityPreprocessor
from src.model import AirQualityForecaster


def main():
    # Load config
    import yaml
    from dotenv import load_dotenv
    load_dotenv()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)

    # Fetch data (will use environment credentials if set)
    fetcher = NASADataFetcher(config)
    try:
        fetcher.authenticate()
    except Exception:
        logger.warning('Authentication failed - continuing and letting fetcher use synthetic data')

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - 
                  timedelta(days=config['data']['time_range']['history_days'])).strftime('%Y-%m-%d')

    tempo_df = fetcher.fetch_tempo_data(start_date, end_date)
    weather_df = fetcher.fetch_weather_data(start_date, end_date)
    ground_df = fetcher.fetch_ground_truth(start_date, end_date)

    tempo_df.to_csv('./data/raw/tempo_data.csv', index=False)
    weather_df.to_csv('./data/raw/weather_data.csv', index=False)
    ground_df.to_csv('./data/raw/ground_data.csv', index=False)

    # Preprocess
    preprocessor = AirQualityPreprocessor(config)
    merged, _ = preprocessor.merge_datasets(tempo_df, weather_df, ground_df), preprocessor
    processed_df = preprocessor.calculate_aqi(merged)
    processed_df = preprocessor.engineer_features(processed_df)
    processed_df.to_csv('./data/processed/processed_data.csv', index=False)

    # Train
    forecaster = AirQualityForecaster(config)
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_train_test(processed_df)
    training_results = forecaster.train(X_train, y_train)
    evaluation_results = forecaster.evaluate(X_test, y_test)

    # Save model with timestamped filename and a stable symlink
    version = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    model_filename = f'./data/models/model_{version}.pkl'
    forecaster.save_model(model_filename)

    # Update 'best_model.pkl' symlink-like copy
    stable_path = './data/models/best_model.pkl'
    try:
        # Overwrite stable path
        forecaster.save_model(stable_path)
    except Exception as e:
        logger.error(f'Failed to write stable model path: {e}')

    # Save metrics
    metrics_path = './data/models/metrics.json'
    metrics = {}
    for model_name, metrics_data in evaluation_results.items():
        metrics[model_name] = metrics_data
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info('Training complete')
    logger.info(f'Models saved to {model_filename} and {stable_path}')


if __name__ == '__main__':
    from datetime import timedelta
    main()
