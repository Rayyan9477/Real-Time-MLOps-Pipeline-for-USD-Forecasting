"""
Data transformation module for feature engineering and volatility calculation.
Implements lag features, rolling statistics, and target variable creation.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats

from config.config import MODEL_CONFIG, DATA_QUALITY_CONFIG, PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("data_transformation")


class FeatureEngineer:
    """Feature engineering for time series forex data."""

    def __init__(self, rolling_window: int = None):
        """
        Initialize feature engineer.

        Args:
            rolling_window: Window size for rolling statistics
        """
        self.rolling_window = rolling_window or MODEL_CONFIG["rolling_window_size"]
        logger.info(
            f"Initialized FeatureEngineer with rolling_window={self.rolling_window}"
        )

    @staticmethod
    def calculate_log_returns(df: pd.DataFrame, column: str = "close") -> pd.Series:
        """
        Calculate log returns for a given column.

        Args:
            df: DataFrame with price data
            column: Column to calculate returns for

        Returns:
            Series with log returns
        """
        return np.log(df[column] / df[column].shift(1))

    def create_lag_features(
        self, df: pd.DataFrame, column: str = "close", lags: list = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series forecasting.

        Args:
            df: DataFrame with price data
            column: Column to create lags for
            lags: List of lag periods (default: 1-24 hours)

        Returns:
            DataFrame with lag features added
        """
        if lags is None:
            lags = [1, 2, 3, 4, 6, 8, 12, 24]  # Important lags for hourly data

        df_copy = df.copy()

        for lag in lags:
            df_copy[f"{column}_lag_{lag}"] = df_copy[column].shift(lag)

        logger.info(f"Created {len(lags)} lag features for '{column}'")
        return df_copy

    def create_rolling_features(
        self, df: pd.DataFrame, column: str = "close"
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.

        Args:
            df: DataFrame with price data
            column: Column to calculate rolling stats for

        Returns:
            DataFrame with rolling features added
        """
        df_copy = df.copy()

        windows = [4, 8, 24]  # 4h, 8h, 24h windows

        for window in windows:
            # Rolling mean
            df_copy[f"{column}_rolling_mean_{window}"] = (
                df_copy[column].rolling(window=window).mean()
            )

            # Rolling std (volatility proxy)
            df_copy[f"{column}_rolling_std_{window}"] = (
                df_copy[column].rolling(window=window).std()
            )

            # Rolling min/max
            df_copy[f"{column}_rolling_min_{window}"] = (
                df_copy[column].rolling(window=window).min()
            )
            df_copy[f"{column}_rolling_max_{window}"] = (
                df_copy[column].rolling(window=window).max()
            )

        logger.info(f"Created rolling features for {len(windows)} windows")
        return df_copy

    def calculate_volatility(self, df: pd.DataFrame, window: int = None) -> pd.Series:
        """
        Calculate volatility as standard deviation of log returns.

        Args:
            df: DataFrame with price data
            window: Rolling window size for volatility calculation

        Returns:
            Series with volatility values
        """
        window = window or self.rolling_window

        # Calculate log returns
        log_returns = self.calculate_log_returns(df, "close")

        # Calculate rolling volatility
        volatility = log_returns.rolling(window=window).std()

        logger.info(f"Calculated volatility with window={window}")
        return volatility

    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime column.

        Args:
            df: DataFrame with datetime column

        Returns:
            DataFrame with time features added
        """
        df_copy = df.copy()

        # Extract time components
        df_copy["hour"] = df_copy["datetime"].dt.hour
        df_copy["day_of_week"] = df_copy["datetime"].dt.dayofweek
        df_copy["day_of_month"] = df_copy["datetime"].dt.day
        df_copy["month"] = df_copy["datetime"].dt.month

        # Cyclical encoding for hour (important for forex)
        df_copy["hour_sin"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
        df_copy["hour_cos"] = np.cos(2 * np.pi * df_copy["hour"] / 24)

        # Cyclical encoding for day of week
        df_copy["day_sin"] = np.sin(2 * np.pi * df_copy["day_of_week"] / 7)
        df_copy["day_cos"] = np.cos(2 * np.pi * df_copy["day_of_week"] / 7)

        logger.info("Created time-based features")
        return df_copy

    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional price-based features.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with price features added
        """
        df_copy = df.copy()

        # Price range
        df_copy["price_range"] = df_copy["high"] - df_copy["low"]

        # Price change
        df_copy["price_change"] = df_copy["close"] - df_copy["open"]
        df_copy["price_change_pct"] = (df_copy["close"] - df_copy["open"]) / df_copy[
            "open"
        ]

        # Average price
        df_copy["avg_price"] = (df_copy["high"] + df_copy["low"]) / 2

        logger.info("Created price-based features")
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: Raw DataFrame with OHLC data

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")

        df_transformed = df.copy()

        # Create time features
        df_transformed = self.create_time_features(df_transformed)

        # Create price features
        df_transformed = self.create_price_features(df_transformed)

        # Create lag features
        df_transformed = self.create_lag_features(df_transformed, "close")

        # Create rolling features
        df_transformed = self.create_rolling_features(df_transformed, "close")

        # Calculate log returns
        df_transformed["log_return"] = self.calculate_log_returns(
            df_transformed, "close"
        )

        # Calculate volatility (target variable)
        df_transformed["volatility"] = self.calculate_volatility(df_transformed)

        # Shift volatility to create target (predict next hour's volatility)
        df_transformed["target_volatility"] = df_transformed["volatility"].shift(-1)

        logger.info(f"Feature engineering complete. Shape: {df_transformed.shape}")
        return df_transformed


class DataCleaner:
    """Data cleaning and preprocessing."""

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame, method: str = "forward"
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: DataFrame to clean
            method: Method for handling nulls ('forward', 'backward', 'drop')

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        initial_nulls = df_clean.isnull().sum().sum()

        if method == "forward":
            df_clean = df_clean.ffill()
        elif method == "backward":
            df_clean = df_clean.bfill()
        elif method == "drop":
            df_clean = df_clean.dropna()

        final_nulls = df_clean.isnull().sum().sum()

        logger.info(
            f"Handled missing values: {initial_nulls} -> {final_nulls} ({method})"
        )
        return df_clean

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame, columns: list = None, std_threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove outliers based on z-score.

        Args:
            df: DataFrame to clean
            columns: Columns to check for outliers
            std_threshold: Z-score threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        std_threshold = std_threshold or DATA_QUALITY_CONFIG["outlier_std_threshold"]

        if columns is None:
            columns = ["close", "open", "high", "low"]

        df_clean = df.copy()
        initial_rows = len(df_clean)

        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < std_threshold) | df_clean[col].isna()]

        removed_rows = initial_rows - len(df_clean)
        logger.info(f"Removed {removed_rows} outlier rows (z-score > {std_threshold})")

        return df_clean

    @staticmethod
    def drop_nan_target(
        df: pd.DataFrame, target_col: str = "target_volatility"
    ) -> pd.DataFrame:
        """
        Drop rows with NaN in target column.

        Args:
            df: DataFrame to clean
            target_col: Name of target column

        Returns:
            DataFrame without NaN targets
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)

        df_clean = df_clean.dropna(subset=[target_col])

        removed_rows = initial_rows - len(df_clean)
        logger.info(f"Dropped {removed_rows} rows with NaN target")

        return df_clean


def transform_data(df: pd.DataFrame, save_processed: bool = True) -> pd.DataFrame:
    """
    Main transformation pipeline: clean data and engineer features.

    Args:
        df: Raw DataFrame from extraction
        save_processed: Whether to save processed data to disk

    Returns:
        Transformed DataFrame ready for modeling
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA TRANSFORMATION PIPELINE")
    logger.info("=" * 60)

    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.handle_missing_values(df, method="forward")
    df_clean = cleaner.remove_outliers(df_clean)

    # Engineer features
    engineer = FeatureEngineer()
    df_transformed = engineer.transform(df_clean)

    # Drop rows with NaN target or insufficient features
    df_final = cleaner.drop_nan_target(df_transformed)
    df_final = df_final.dropna()

    logger.info(f"Final dataset shape: {df_final.shape}")
    logger.info(f"Features: {df_final.shape[1]} columns")

    # Save processed data
    if save_processed:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_data_{timestamp}.parquet"
        filepath = PROCESSED_DATA_DIR / filename

        df_final.to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")

    logger.info("=" * 60)
    logger.info("TRANSFORMATION PIPELINE COMPLETE ✓")
    logger.info("=" * 60)

    return df_final


def generate_data_profile(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Generate data quality report using pandas profiling.

    Args:
        df: DataFrame to profile
        output_path: Path to save HTML report

    Returns:
        Path to generated report
    """
    try:
        from ydata_profiling import ProfileReport

        logger.info("Generating data profile report...")

        profile = ProfileReport(
            df, title="USD Volatility Data Profile", minimal=False, explorative=True
        )

        if output_path is None:
            from datetime import datetime
            from config.config import REPORTS_DIR

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(REPORTS_DIR / f"data_profile_{timestamp}.html")

        profile.to_file(output_path)
        logger.info(f"Data profile report saved to {output_path}")

        return output_path

    except ImportError:
        logger.warning("ydata-profiling not available. Skipping profile generation.")
        return None
