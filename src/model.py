"""
Machine Learning Models for Air Quality Forecasting
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class AirQualityForecaster:
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.feature_names = []
        self.best_model_name = None
        self.scaler = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train multiple models and select the best one"""
        
        self.feature_names = X_train.columns.tolist()
        results = {}
        
        # Model 1: Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config['model']['random_state'],
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Cross-validation
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, 
                                        cv=5, scoring='neg_mean_squared_error')
        rf_rmse_cv = np.sqrt(-rf_cv_scores.mean())
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = {
            'cv_rmse': rf_rmse_cv,
            'feature_importance': dict(zip(self.feature_names, rf_model.feature_importances_))
        }
        
        logger.info(f"Random Forest CV RMSE: {rf_rmse_cv:.2f}")
        
        # Model 2: Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config['model']['random_state']
        )
        gb_model.fit(X_train, y_train)
        
        gb_cv_scores = cross_val_score(gb_model, X_train, y_train, 
                                        cv=5, scoring='neg_mean_squared_error')
        gb_rmse_cv = np.sqrt(-gb_cv_scores.mean())
        
        self.models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'cv_rmse': gb_rmse_cv,
            'feature_importance': dict(zip(self.feature_names, gb_model.feature_importances_))
        }
        
        logger.info(f"Gradient Boosting CV RMSE: {gb_rmse_cv:.2f}")
        
        # Select best model
        self.best_model_name = min(results.keys(), key=lambda k: results[k]['cv_rmse'])
        logger.info(f"Best model: {self.best_model_name}")
        
        return results
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate models on test set"""
        
        evaluation = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            evaluation[name] = metrics
            logger.info(f"{name} - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return evaluation
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model"""
        
        if self.best_model_name is None:
            raise ValueError("Model not trained yet!")
        
        model = self.models[self.best_model_name]
        predictions = model.predict(X)
        
        # Ensure predictions are within valid AQI range
        predictions = np.clip(predictions, 0, 500)
        
        return predictions
    
    def forecast_next_hours(self, current_data: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
        """
        Forecast air quality for the next N hours
        Uses iterative prediction with rolling window
        """
        
        if self.best_model_name is None:
            raise ValueError("Model not trained yet!")
        
        forecasts = []
        current_features = current_data[self.feature_names].iloc[-1:].copy()
        
        for h in range(1, hours + 1):
            # Predict next hour
            pred_aqi = self.predict(current_features)[0]
            
            # Update temporal features
            next_hour = (current_features['hour'].values[0] + h) % 24
            current_features['hour'] = next_hour
            current_features['hour_sin'] = np.sin(2 * np.pi * next_hour / 24)
            current_features['hour_cos'] = np.cos(2 * np.pi * next_hour / 24)
            
            # Update lagged features (simple approach)
            if 'aqi_lag1' in current_features.columns:
                current_features['aqi_lag1'] = pred_aqi
            
            forecasts.append({
                'hours_ahead': h,
                'predicted_aqi': pred_aqi,
                'timestamp': pd.Timestamp.now() + pd.Timedelta(hours=h)
            })
        
        return pd.DataFrame(forecasts)
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """Get top N most important features"""
        
        if self.best_model_name is None:
            return {}
        
        model = self.models[self.best_model_name]
        importance = dict(zip(self.feature_names, model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return sorted_importance
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        
        model_data = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.best_model_name = model_data['best_model_name']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', self.config)
        
        logger.info(f"Model loaded from {filepath}")