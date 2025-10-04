"""
Data Preprocessing and Feature Engineering
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class AirQualityPreprocessor:
    def __init__(self, config: dict):
        self.config = config
    
    def merge_datasets(self, tempo_df: pd.DataFrame, weather_df: pd.DataFrame, 
                       ground_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources on timestamp"""
        
        # Ensure timestamps are datetime
        tempo_df['timestamp'] = pd.to_datetime(tempo_df['timestamp'])
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        ground_df['timestamp'] = pd.to_datetime(ground_df['timestamp'])
        
        # Round to nearest hour for merging
        tempo_df['timestamp'] = tempo_df['timestamp'].dt.floor('H')
        weather_df['timestamp'] = weather_df['timestamp'].dt.floor('H')
        ground_df['timestamp'] = ground_df['timestamp'].dt.floor('H')
        
        # Merge datasets
        df = tempo_df.merge(weather_df, on='timestamp', how='inner')
        df = df.merge(ground_df[['timestamp', 'pm25']], on='timestamp', how='left')
        
        logger.info(f"Merged dataset shape: {df.shape}")
        return df
    
    def calculate_aqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Air Quality Index from pollutant concentrations
        Simplified AQI calculation based on PM2.5
        """
        df = df.copy()
        
        # US EPA AQI breakpoints for PM2.5 (24-hour average)
        def pm25_to_aqi(pm25):
            if pd.isna(pm25):
                return np.nan
            elif pm25 <= 12.0:
                return self._linear_interpolation(pm25, 0, 12.0, 0, 50)
            elif pm25 <= 35.4:
                return self._linear_interpolation(pm25, 12.1, 35.4, 51, 100)
            elif pm25 <= 55.4:
                return self._linear_interpolation(pm25, 35.5, 55.4, 101, 150)
            elif pm25 <= 150.4:
                return self._linear_interpolation(pm25, 55.5, 150.4, 151, 200)
            elif pm25 <= 250.4:
                return self._linear_interpolation(pm25, 150.5, 250.4, 201, 300)
            else:
                return self._linear_interpolation(pm25, 250.5, 500.4, 301, 500)
        
        # Apply AQI calculation
        if 'pm25' in df.columns:
            df['aqi'] = df['pm25'].apply(pm25_to_aqi)
        else:
            # Estimate AQI from NO2 if PM2.5 not available
            df['aqi'] = df['no2'].apply(lambda x: min(x * 3, 500) if pd.notna(x) else np.nan)
        
        return df
    
    @staticmethod
    def _linear_interpolation(value, conc_low, conc_high, aqi_low, aqi_high):
        """Linear interpolation for AQI calculation"""
        return ((aqi_high - aqi_low) / (conc_high - conc_low)) * (value - conc_low) + aqi_low
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ML model"""
        df = df.copy()
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for hour and month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Wind components
        if 'wind_direction' in df.columns and 'wind_speed' in df.columns:
            df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
            df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
        
        # Lagged features (previous values)
        pollutant_cols = ['no2', 'hcho', 'aqi']
        for col in pollutant_cols:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag3'] = df[col].shift(3)
                df[f'{col}_lag6'] = df[col].shift(6)
                df[f'{col}_lag24'] = df[col].shift(24)
        
        # Rolling statistics
        for col in pollutant_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6, min_periods=1).mean()
                df[f'{col}_rolling_std_6h'] = df[col].rolling(window=6, min_periods=1).std()
                df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean()
        
        # Interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity'] = df['temperature'] * df['humidity']
        
        if 'no2' in df.columns and 'wind_speed' in df.columns:
            df['no2_wind_interaction'] = df['no2'] / (df['wind_speed'] + 0.1)
        
        # Drop rows with NaN in target variable
        df = df.dropna(subset=['aqi'])
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        logger.info(f"Features: {df.columns.tolist()}")
        
        return df
    
    def prepare_train_test(self, df: pd.DataFrame, target_col: str = 'aqi') -> Tuple:
        """Prepare training and testing datasets"""
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = ['timestamp', 'lat', 'lon', 'location', target_col, 'pm25']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many NaNs
        feature_cols = [col for col in feature_cols if df[col].isna().sum() / len(df) < 0.3]
        
        # Fill remaining NaNs
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # Time-based split (last 20% for testing)
        split_idx = int(len(df) * 0.8)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Using features: {feature_cols}")
        
        return X_train, X_test, y_train, y_test, feature_cols