"""
NASA Data Fetcher - Retrieves TEMPO and weather data
"""
import earthaccess
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NASADataFetcher:
    def __init__(self, config: Dict):
        self.config = config
        self.auth = None
        
    def authenticate(self):
        """Authenticate with NASA Earthdata"""
        try:
            self.auth = earthaccess.login(strategy="interactive")
            logger.info("Successfully authenticated with NASA Earthdata")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def fetch_tempo_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch TEMPO NO2 and HCHO data
        Note: TEMPO data is still being populated, so we'll simulate with similar datasets
        """
        lat = self.config['data']['location']['lat']
        lon = self.config['data']['location']['lon']
        
        try:
            # Search for TEMPO NO2 data
            # For prototype: using OMI (Ozone Monitoring Instrument) as proxy
            results = earthaccess.search_data(
                short_name='OMNO2d',  # Daily OMI NO2
                temporal=(start_date, end_date),
                bounding_box=(
                    lon - 1, lat - 1,
                    lon + 1, lat + 1
                )
            )
            
            if not results:
                logger.warning("No TEMPO data found, generating synthetic data")
                return self._generate_synthetic_tempo_data(start_date, end_date)
            
            # Download and process
            files = earthaccess.download(results[:10], "./data/raw")
            
            data_list = []
            for file in files:
                ds = xr.open_dataset(file)
                # Extract relevant variables
                data_list.append(self._process_tempo_file(ds, lat, lon))
                ds.close()
            
            df = pd.concat(data_list, ignore_index=True)
            return df
            
        except Exception as e:
            logger.error(f"TEMPO data fetch failed: {e}")
            return self._generate_synthetic_tempo_data(start_date, end_date)
    
    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch MERRA-2 weather data
        """
        lat = self.config['data']['location']['lat']
        lon = self.config['data']['location']['lon']
        
        try:
            # MERRA-2 meteorological data
            results = earthaccess.search_data(
                short_name='M2T1NXSLV',  # MERRA-2 hourly single-level
                temporal=(start_date, end_date)
            )
            
            if not results:
                logger.warning("No weather data found, generating synthetic data")
                return self._generate_synthetic_weather_data(start_date, end_date)
            
            files = earthaccess.download(results[:10], "./data/raw")
            
            data_list = []
            for file in files:
                ds = xr.open_dataset(file)
                data_list.append(self._process_weather_file(ds, lat, lon))
                ds.close()
            
            df = pd.concat(data_list, ignore_index=True)
            return df
            
        except Exception as e:
            logger.error(f"Weather data fetch failed: {e}")
            return self._generate_synthetic_weather_data(start_date, end_date)
    
    def fetch_ground_truth(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch ground station data from OpenAQ
        """
        lat = self.config['data']['location']['lat']
        lon = self.config['data']['location']['lon']
        
        try:
            # OpenAQ API v2
            url = "https://api.openaq.org/v2/measurements"
            params = {
                'coordinates': f"{lat},{lon}",
                'radius': 50000,  # 50km radius
                'date_from': start_date,
                'date_to': end_date,
                'parameter': 'pm25',
                'limit': 10000
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                measurements = data.get('results', [])
                
                if measurements:
                    df = pd.DataFrame([{
                        'timestamp': m['date']['utc'],
                        'pm25': m['value'],
                        'location': m['location'],
                        'lat': m['coordinates']['latitude'],
                        'lon': m['coordinates']['longitude']
                    } for m in measurements])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
            
            logger.warning("No ground truth data, generating synthetic")
            return self._generate_synthetic_ground_data(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Ground truth fetch failed: {e}")
            return self._generate_synthetic_ground_data(start_date, end_date)
    
    def _process_tempo_file(self, ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
        """Extract data from TEMPO/OMI netCDF file"""
        # Find nearest grid point
        lat_idx = np.argmin(np.abs(ds['lat'].values - lat))
        lon_idx = np.argmin(np.abs(ds['lon'].values - lon))
        
        data = {
            'timestamp': pd.to_datetime(ds['time'].values),
            'no2': ds['NO2'].values[lat_idx, lon_idx] if 'NO2' in ds else np.nan,
            'hcho': ds['HCHO'].values[lat_idx, lon_idx] if 'HCHO' in ds else np.nan,
        }
        
        return pd.DataFrame([data])
    
    def _process_weather_file(self, ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
        """Extract weather data from MERRA-2 file"""
        lat_idx = np.argmin(np.abs(ds['lat'].values - lat))
        lon_idx = np.argmin(np.abs(ds['lon'].values - lon))
        
        times = pd.to_datetime(ds['time'].values)
        
        records = []
        for i, time in enumerate(times):
            records.append({
                'timestamp': time,
                'temperature': ds['T2M'].values[i, lat_idx, lon_idx] if 'T2M' in ds else np.nan,
                'wind_speed': np.sqrt(
                    ds['U10M'].values[i, lat_idx, lon_idx]**2 + 
                    ds['V10M'].values[i, lat_idx, lon_idx]**2
                ) if 'U10M' in ds and 'V10M' in ds else np.nan,
                'humidity': ds['RH'].values[i, lat_idx, lon_idx] if 'RH' in ds else np.nan,
                'pressure': ds['PS'].values[i, lat_idx, lon_idx] if 'PS' in ds else np.nan,
            })
        
        return pd.DataFrame(records)
    
    def _generate_synthetic_tempo_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic TEMPO data for prototype"""
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        np.random.seed(42)
        
        # Realistic patterns: higher NO2 during rush hours, seasonal variations
        hour = dates.hour
        day_of_year = dates.dayofyear
        
        # Base levels with diurnal and seasonal patterns
        no2_base = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        no2_diurnal = 5 * (np.exp(-((hour - 8)**2) / 20) + np.exp(-((hour - 18)**2) / 20))
        no2 = no2_base + no2_diurnal + np.random.normal(0, 3, len(dates))
        no2 = np.maximum(no2, 0)  # No negative values
        
        hcho = 8 + 4 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 1.5, len(dates))
        hcho = np.maximum(hcho, 0)
        
        return pd.DataFrame({
            'timestamp': dates,
            'no2': no2,
            'hcho': hcho
        })
    
    def _generate_synthetic_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic weather data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        np.random.seed(42)
        
        hour = dates.hour
        day_of_year = dates.dayofyear
        
        # Temperature with diurnal and seasonal patterns (Celsius)
        temp_seasonal = 28 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp_diurnal = 3 * np.sin(2 * np.pi * (hour - 6) / 24)
        temperature = temp_seasonal + temp_diurnal + np.random.normal(0, 1, len(dates))
        
        # Wind speed (m/s)
        wind_speed = 3 + 2 * np.random.exponential(1, len(dates))
        wind_speed = np.minimum(wind_speed, 15)
        
        # Wind direction (degrees)
        wind_direction = np.random.uniform(0, 360, len(dates))
        
        # Humidity (%)
        humidity = 70 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5, len(dates))
        humidity = np.clip(humidity, 40, 95)
        
        # Pressure (Pa)
        pressure = 101325 + np.random.normal(0, 500, len(dates))
        
        return pd.DataFrame({
            'timestamp': dates,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'humidity': humidity,
            'pressure': pressure
        })
    
    def _generate_synthetic_ground_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic ground truth PM2.5 data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        np.random.seed(42)
        
        hour = dates.hour
        day_of_year = dates.dayofyear
        
        # PM2.5 patterns
        pm25_base = 35 + 15 * np.sin(2 * np.pi * day_of_year / 365)
        pm25_diurnal = 10 * (np.exp(-((hour - 8)**2) / 15) + np.exp(-((hour - 19)**2) / 15))
        pm25 = pm25_base + pm25_diurnal + np.random.gamma(2, 3, len(dates))
        pm25 = np.maximum(pm25, 5)
        
        return pd.DataFrame({
            'timestamp': dates,
            'pm25': pm25,
            'lat': self.config['data']['location']['lat'],
            'lon': self.config['data']['location']['lon']
        })