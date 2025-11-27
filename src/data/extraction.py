"""
Data extraction module for fetching forex data from Twelve Data API.
Implements retry logic, rate limiting, and quality checks.
"""
import time
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from config.config import TWELVE_DATA_CONFIG, DATA_QUALITY_CONFIG, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("data_extraction")


class TwelveDataClient:
    """Client for interacting with Twelve Data API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Twelve Data API client.
        
        Args:
            api_key: API key for Twelve Data (uses config if not provided)
        """
        self.api_key = api_key or TWELVE_DATA_CONFIG["api_key"]
        self.base_url = TWELVE_DATA_CONFIG["base_url"]
        self.symbol = TWELVE_DATA_CONFIG["symbol"]
        self.interval = TWELVE_DATA_CONFIG["interval"]
        
        if not self.api_key:
            raise ValueError("TWELVE_DATA_API_KEY not found in environment variables")
        
        logger.info(f"Initialized TwelveDataClient for {self.symbol} with interval {self.interval}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def fetch_time_series(
        self, 
        outputsize: int = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Fetch time series data from Twelve Data API with retry logic.
        
        Args:
            outputsize: Number of data points to fetch
            start_date: Start date for data (YYYY-MM-DD format)
            end_date: End date for data (YYYY-MM-DD format)
            
        Returns:
            JSON response from API
        """
        outputsize = outputsize or TWELVE_DATA_CONFIG["fetch_size"]
        
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON"
        }
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        url = f"{self.base_url}/time_series"
        
        logger.info(f"Fetching data: {self.symbol}, interval={self.interval}, outputsize={outputsize}")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "status" in data and data["status"] == "error":
                error_msg = data.get("message", "Unknown API error")
                logger.error(f"API Error: {error_msg}")
                raise Exception(f"Twelve Data API Error: {error_msg}")
            
            if "values" not in data:
                logger.error(f"No 'values' field in API response: {data}")
                raise Exception("Invalid API response format")
            
            logger.info(f"Successfully fetched {len(data['values'])} data points")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def fetch_to_dataframe(
        self,
        outputsize: int = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch time series data and convert to pandas DataFrame.
        
        Args:
            outputsize: Number of data points to fetch
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        data = self.fetch_time_series(outputsize, start_date, end_date)
        
        # Convert to DataFrame
        df = pd.DataFrame(data["values"])
        
        # Convert data types
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        
        # Sort by datetime (API returns newest first)
        df = df.sort_values("datetime").reset_index(drop=True)
        
        logger.info(f"Converted to DataFrame: shape={df.shape}")
        return df
    
    def save_raw_data(self, df: pd.DataFrame, timestamp: Optional[str] = None) -> str:
        """
        Save raw data to local storage with timestamp.
        
        Args:
            df: DataFrame to save
            timestamp: Custom timestamp (uses current time if not provided)
            
        Returns:
            Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"raw_data_{timestamp}.csv"
        filepath = RAW_DATA_DIR / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved raw data to {filepath}")
        
        return str(filepath)


class DataQualityChecker:
    """Validates data quality according to defined thresholds."""
    
    @staticmethod
    def check_null_values(df: pd.DataFrame, max_null_pct: float = None) -> bool:
        """
        Check if null values exceed threshold.
        
        Args:
            df: DataFrame to check
            max_null_pct: Maximum allowed null percentage
            
        Returns:
            True if quality check passes
        """
        max_null_pct = max_null_pct or DATA_QUALITY_CONFIG["max_null_percentage"]
        
        key_columns = ["datetime", "open", "high", "low", "close"]
        null_counts = df[key_columns].isnull().sum()
        null_pct = (null_counts / len(df)) * 100
        
        failed_columns = null_pct[null_pct > max_null_pct]
        
        if not failed_columns.empty:
            logger.error(f"Null value check FAILED. Columns exceeding {max_null_pct}% threshold:")
            for col, pct in failed_columns.items():
                logger.error(f"  {col}: {pct:.2f}%")
            return False
        
        logger.info("Null value check PASSED")
        return True
    
    @staticmethod
    def check_schema(df: pd.DataFrame) -> bool:
        """
        Validate DataFrame schema.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if schema is valid
        """
        required_columns = ["datetime", "open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Schema check FAILED. Missing columns: {missing_columns}")
            return False
        
        # Check data types
        numeric_columns = ["open", "high", "low", "close"]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Schema check FAILED. Column '{col}' is not numeric")
                return False
        
        logger.info("Schema check PASSED")
        return True
    
    @staticmethod
    def check_minimum_data_points(df: pd.DataFrame, min_points: int = None) -> bool:
        """
        Check if DataFrame has minimum required data points.
        
        Args:
            df: DataFrame to check
            min_points: Minimum required data points
            
        Returns:
            True if check passes
        """
        min_points = min_points or DATA_QUALITY_CONFIG["min_data_points"]
        
        if len(df) < min_points:
            logger.error(f"Data points check FAILED. Got {len(df)}, need at least {min_points}")
            return False
        
        logger.info(f"Data points check PASSED ({len(df)} points)")
        return True
    
    @classmethod
    def validate_data(cls, df: pd.DataFrame) -> bool:
        """
        Run all data quality checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if all checks pass
            
        Raises:
            Exception if any quality check fails
        """
        logger.info("Starting data quality validation...")
        
        checks = [
            cls.check_schema(df),
            cls.check_null_values(df),
            cls.check_minimum_data_points(df),
        ]
        
        if not all(checks):
            raise Exception("Data quality validation FAILED")
        
        logger.info("All data quality checks PASSED ✓")
        return True


def extract_forex_data(
    outputsize: int = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> pd.DataFrame:
    """
    Main function to extract forex data with quality checks.
    
    Args:
        outputsize: Number of data points to fetch
        start_date: Start date for data
        end_date: End date for data
        save_raw: Whether to save raw data to disk
        
    Returns:
        Validated DataFrame with forex data
    """
    client = TwelveDataClient()
    
    # Fetch data
    df = client.fetch_to_dataframe(outputsize, start_date, end_date)
    
    # Save raw data
    if save_raw:
        client.save_raw_data(df)
    
    # Validate data quality
    DataQualityChecker.validate_data(df)
    
    return df
