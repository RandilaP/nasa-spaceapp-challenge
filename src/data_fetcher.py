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
            # Prefer non-interactive auth when credentials are provided via env vars
            import os
            username = os.getenv('EARTHDATA_USERNAME')
            password = os.getenv('EARTHDATA_PASSWORD')

            if username and password:
                # Attempt non-interactive login using provided credentials.
                try:
                    # earthaccess API may accept username/password depending on version
                    self.auth = earthaccess.login(username=username, password=password)
                    logger.info("Authenticated with NASA Earthdata using environment credentials")
                    return
                except Exception as e:
                    logger.warning(f"Non-interactive Earthdata auth failed: {e} - falling back to interactive")

            # Fallback to interactive login for developer convenience
            self.auth = earthaccess.login(strategy="interactive")
            logger.info("Successfully authenticated with NASA Earthdata (interactive)")

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
            # Remove empty results
            cleaned = [d for d in data_list if isinstance(d, pd.DataFrame) and not d.empty]
            if not cleaned:
                logger.warning("Downloaded TEMPO files did not contain usable variables; generating synthetic TEMPO data")
                return self._generate_synthetic_tempo_data(start_date, end_date)

            df = pd.concat(cleaned, ignore_index=True)
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                logger.warning("Processed TEMPO data missing 'timestamp' column; generating synthetic TEMPO data")
                return self._generate_synthetic_tempo_data(start_date, end_date)

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

            cleaned = [d for d in data_list if isinstance(d, pd.DataFrame) and not d.empty]
            if not cleaned:
                logger.warning("Downloaded weather files did not contain usable variables; generating synthetic weather data")
                return self._generate_synthetic_weather_data(start_date, end_date)

            df = pd.concat(cleaned, ignore_index=True)
            if 'timestamp' not in df.columns:
                logger.warning("Processed weather data missing 'timestamp' column; generating synthetic weather data")
                return self._generate_synthetic_weather_data(start_date, end_date)

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
        try:
            # Try to find latitude/longitude arrays (datasets vary in naming)
            lat_candidates = ['lat', 'latitude', 'lats', 'nav_lat']
            lon_candidates = ['lon', 'longitude', 'lons', 'nav_lon']

            lat_arr = None
            lon_arr = None
            for name in lat_candidates:
                if name in ds:
                    lat_arr = ds[name].values
                    break
                if name in ds.coords:
                    lat_arr = ds.coords[name].values
                    break

            for name in lon_candidates:
                if name in ds:
                    lon_arr = ds[name].values
                    break
                if name in ds.coords:
                    lon_arr = ds.coords[name].values
                    break

            if lat_arr is None or lon_arr is None:
                logger.warning(f"No lat/lon coordinates found in TEMPO dataset. Variables on the dataset include {list(ds.variables.keys())}")
                return pd.DataFrame()

            # Find nearest grid point
            lat_idx = np.argmin(np.abs(lat_arr - lat))
            lon_idx = np.argmin(np.abs(lon_arr - lon))

            # Time
            time_vals = None
            if 'time' in ds:
                time_vals = ds['time'].values
            elif 'Time' in ds:
                time_vals = ds['Time'].values
            else:
                # fallback to now
                time_vals = [pd.Timestamp.now()]

            # Helper to extract variable if present and indexed by lat/lon
            def extract_var(var_names):
                for v in var_names:
                    if v in ds:
                        arr = ds[v].values
                        try:
                            # try [lat, lon] ordering or time,lat,lon
                            if arr.ndim == 2:
                                return arr[lat_idx, lon_idx]
                            elif arr.ndim >= 3:
                                # assume time, lat, lon or lat, lon, time
                                # try time, lat, lon
                                return arr[0, lat_idx, lon_idx]
                            else:
                                return arr.item()
                        except Exception:
                            return np.nan
                return np.nan

            no2 = extract_var(['NO2', 'no2', 'tropospheric_NO2_column_amount'])
            hcho = extract_var(['HCHO', 'hcho'])

            data = {
                'timestamp': pd.to_datetime(time_vals),
                'no2': no2,
                'hcho': hcho,
            }

            return pd.DataFrame([data])

        except Exception as e:
            logger.error(f"Error processing TEMPO file: {e}")
            return pd.DataFrame()
    
    def _process_weather_file(self, ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
        """Extract weather data from MERRA-2 file"""
        try:
            # Find lat/lon arrays similarly to TEMPO processing
            lat_candidates = ['lat', 'latitude', 'lats', 'nav_lat']
            lon_candidates = ['lon', 'longitude', 'lons', 'nav_lon']

            lat_arr = None
            lon_arr = None
            for name in lat_candidates:
                if name in ds:
                    lat_arr = ds[name].values
                    break
                if name in ds.coords:
                    lat_arr = ds.coords[name].values
                    break

            for name in lon_candidates:
                if name in ds:
                    lon_arr = ds[name].values
                    break
                if name in ds.coords:
                    lon_arr = ds.coords[name].values
                    break

            if lat_arr is None or lon_arr is None:
                logger.warning(f"No lat/lon coordinates found in weather dataset. Variables on the dataset include {list(ds.variables.keys())}")
                return pd.DataFrame()

            lat_idx = np.argmin(np.abs(lat_arr - lat))
            lon_idx = np.argmin(np.abs(lon_arr - lon))

            # Time values
            if 'time' in ds:
                times = pd.to_datetime(ds['time'].values)
            else:
                times = pd.to_datetime([pd.Timestamp.now()])

            records = []
            for i, time in enumerate(times):
                def safe_get(var_names):
                    for v in var_names:
                        if v in ds:
                            arr = ds[v].values
                            try:
                                if arr.ndim >= 3:
                                    return arr[i, lat_idx, lon_idx]
                                elif arr.ndim == 2:
                                    return arr[lat_idx, lon_idx]
                                else:
                                    return arr.item()
                            except Exception:
                                return np.nan
                    return np.nan

                temperature = safe_get(['T2M', 'T', 't2m', 'Temperature'])
                u = safe_get(['U10M', 'U10', 'u10'])
                v = safe_get(['V10M', 'V10', 'v10'])
                humidity = safe_get(['RH', 'rh', 'RelativeHumidity'])
                pressure = safe_get(['PS', 'pressure', 'P'])

                wind_speed = np.nan
                try:
                    if not np.isnan(u) and not np.isnan(v):
                        wind_speed = np.sqrt(u**2 + v**2)
                except Exception:
                    wind_speed = np.nan

                records.append({
                    'timestamp': time,
                    'temperature': temperature,
                    'wind_speed': wind_speed,
                    'humidity': humidity,
                    'pressure': pressure,
                })

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Error processing weather file: {e}")
            return pd.DataFrame()
    
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