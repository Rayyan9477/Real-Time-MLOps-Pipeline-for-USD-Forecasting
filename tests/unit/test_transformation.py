"""
Unit tests for data transformation module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.transformation import FeatureEngineer, DataCleaner


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample forex data for testing."""
        return pd.DataFrame({
            "datetime": pd.date_range("2025-11-01", periods=100, freq="h"),
            "open": np.random.uniform(1.08, 1.09, 100),
            "high": np.random.uniform(1.08, 1.09, 100),
            "low": np.random.uniform(1.08, 1.09, 100),
            "close": np.random.uniform(1.08, 1.09, 100)
        })
    
    def test_calculate_log_returns(self, sample_data):
        """Test log returns calculation."""
        engineer = FeatureEngineer()
        log_returns = engineer.calculate_log_returns(sample_data, "close")
        
        assert len(log_returns) == len(sample_data)
        assert pd.isna(log_returns.iloc[0])  # First value should be NaN
        assert not pd.isna(log_returns.iloc[1])
    
    def test_create_lag_features(self, sample_data):
        """Test lag feature creation."""
        engineer = FeatureEngineer()
        df_with_lags = engineer.create_lag_features(sample_data, "close", lags=[1, 2, 3])
        
        assert "close_lag_1" in df_with_lags.columns
        assert "close_lag_2" in df_with_lags.columns
        assert "close_lag_3" in df_with_lags.columns
        assert df_with_lags.shape[1] == sample_data.shape[1] + 3
    
    def test_create_rolling_features(self, sample_data):
        """Test rolling feature creation."""
        engineer = FeatureEngineer()
        df_with_rolling = engineer.create_rolling_features(sample_data, "close")
        
        # Check that rolling features exist
        assert "close_rolling_mean_4" in df_with_rolling.columns
        assert "close_rolling_std_4" in df_with_rolling.columns
        assert "close_rolling_mean_24" in df_with_rolling.columns
    
    def test_calculate_volatility(self, sample_data):
        """Test volatility calculation."""
        engineer = FeatureEngineer()
        volatility = engineer.calculate_volatility(sample_data, window=24)
        
        assert len(volatility) == len(sample_data)
        # First 24 values should be NaN due to rolling window
        assert pd.isna(volatility.iloc[0])
        assert not all(pd.isna(volatility))
    
    def test_create_time_features(self, sample_data):
        """Test time feature creation."""
        engineer = FeatureEngineer()
        df_with_time = engineer.create_time_features(sample_data)
        
        assert "hour" in df_with_time.columns
        assert "day_of_week" in df_with_time.columns
        assert "hour_sin" in df_with_time.columns
        assert "hour_cos" in df_with_time.columns
        
        # Check cyclical encoding range
        assert df_with_time["hour_sin"].between(-1, 1).all()
        assert df_with_time["hour_cos"].between(-1, 1).all()
    
    def test_create_price_features(self, sample_data):
        """Test price feature creation."""
        engineer = FeatureEngineer()
        df_with_price = engineer.create_price_features(sample_data)
        
        assert "price_range" in df_with_price.columns
        assert "price_change" in df_with_price.columns
        assert "price_change_pct" in df_with_price.columns
        assert "avg_price" in df_with_price.columns


class TestDataCleaner:
    """Test DataCleaner functionality."""
    
    def test_handle_missing_values_forward(self):
        """Test forward fill for missing values."""
        df = pd.DataFrame({
            "close": [1.085, None, None, 1.087, 1.088]
        })
        
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(df, method="forward")
        
        assert df_clean["close"].isna().sum() == 0
        assert df_clean["close"].iloc[1] == 1.085
        assert df_clean["close"].iloc[2] == 1.085
    
    def test_remove_outliers(self):
        """Test outlier removal."""
        df = pd.DataFrame({
            "close": [1.085] * 95 + [10.0] * 5  # 5 extreme outliers
        })
        
        cleaner = DataCleaner()
        df_clean = cleaner.remove_outliers(df, columns=["close"], std_threshold=3.0)
        
        assert len(df_clean) < len(df)
        assert df_clean["close"].max() < 2.0  # Outliers removed
    
    def test_drop_nan_target(self):
        """Test dropping NaN targets."""
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "target_volatility": [0.001, 0.002, None, 0.003, None]
        })
        
        cleaner = DataCleaner()
        df_clean = cleaner.drop_nan_target(df, target_col="target_volatility")
        
        assert len(df_clean) == 3
        assert df_clean["target_volatility"].isna().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
