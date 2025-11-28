"""
Integration tests for the ML pipeline.
Tests the complete flow from data processing to model prediction.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.transformation import FeatureEngineer, DataCleaner, transform_data
from src.monitoring.drift_detector import DriftDetector


class TestDataPipeline:
    """Integration tests for data pipeline."""
    
    @pytest.fixture
    def sample_forex_data(self):
        """Create sample forex data mimicking Twelve Data API response."""
        np.random.seed(42)
        n_samples = 200
        
        base_price = 1.0850
        price_changes = np.random.normal(0, 0.0005, n_samples).cumsum()
        close_prices = base_price + price_changes
        
        return pd.DataFrame({
            "datetime": pd.date_range(
                start=datetime.now() - timedelta(hours=n_samples),
                periods=n_samples,
                freq="h"
            ),
            "open": close_prices + np.random.uniform(-0.0002, 0.0002, n_samples),
            "high": close_prices + np.random.uniform(0, 0.001, n_samples),
            "low": close_prices - np.random.uniform(0, 0.001, n_samples),
            "close": close_prices
        })
    
    def test_full_transformation_pipeline(self, sample_forex_data):
        """Test complete transformation from raw to processed data."""
        # Clean data
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(sample_forex_data, method="forward")
        
        # Engineer features
        engineer = FeatureEngineer(rolling_window=24)
        df_transformed = engineer.transform(df_clean)
        
        # Verify output
        assert "target_volatility" in df_transformed.columns
        assert "log_return" in df_transformed.columns
        assert "close_lag_1" in df_transformed.columns
        assert "hour_sin" in df_transformed.columns
        assert "price_range" in df_transformed.columns
        
        # Check no excessive NaN
        nan_ratio = df_transformed.isna().sum().sum() / df_transformed.size
        assert nan_ratio < 0.3  # Less than 30% NaN (expected due to rolling windows)
    
    def test_feature_engineering_output_shape(self, sample_forex_data):
        """Test that feature engineering produces expected number of features."""
        engineer = FeatureEngineer()
        df_transformed = engineer.transform(sample_forex_data)
        
        # Should have significantly more columns than input
        assert df_transformed.shape[1] > sample_forex_data.shape[1]
        
        # At least 30 features expected
        assert df_transformed.shape[1] >= 30
    
    def test_volatility_calculation(self, sample_forex_data):
        """Test volatility calculation produces valid values."""
        engineer = FeatureEngineer(rolling_window=24)
        
        volatility = engineer.calculate_volatility(sample_forex_data, window=24)
        
        # First 24 should be NaN
        assert volatility.iloc[:24].isna().all()
        
        # Rest should be positive (std dev is always >= 0)
        valid_volatility = volatility.dropna()
        assert (valid_volatility >= 0).all()
    
    def test_data_cleaning_handles_outliers(self, sample_forex_data):
        """Test that outlier removal works correctly."""
        # Add outliers
        df_with_outliers = sample_forex_data.copy()
        df_with_outliers.loc[50, "close"] = 100  # Extreme outlier
        df_with_outliers.loc[100, "close"] = -50  # Extreme outlier
        
        cleaner = DataCleaner()
        df_clean = cleaner.remove_outliers(df_with_outliers, columns=["close"], std_threshold=3)
        
        # Should have removed at least the extreme outliers
        assert len(df_clean) < len(df_with_outliers)
        assert df_clean["close"].max() < 10
        assert df_clean["close"].min() > -10


class TestDriftMonitoring:
    """Integration tests for drift monitoring."""
    
    @pytest.fixture
    def training_data(self):
        """Create training data features."""
        np.random.seed(42)
        return pd.DataFrame({
            "close_lag_1": np.random.normal(1.085, 0.01, 1000),
            "close_rolling_mean_24": np.random.normal(1.085, 0.005, 1000),
            "close_rolling_std_24": np.random.uniform(0.001, 0.003, 1000),
            "log_return": np.random.normal(0, 0.001, 1000),
            "hour_sin": np.random.uniform(-1, 1, 1000),
            "hour_cos": np.random.uniform(-1, 1, 1000)
        })
    
    def test_drift_detection_integration(self, training_data):
        """Test full drift detection workflow."""
        # Fit detector on training data
        detector = DriftDetector(significance_level=0.05)
        detector.fit(training_data)
        
        # Create similar production data
        np.random.seed(43)
        production_data = pd.DataFrame({
            "close_lag_1": np.random.normal(1.085, 0.01, 200),
            "close_rolling_mean_24": np.random.normal(1.085, 0.005, 200),
            "close_rolling_std_24": np.random.uniform(0.001, 0.003, 200),
            "log_return": np.random.normal(0, 0.001, 200),
            "hour_sin": np.random.uniform(-1, 1, 200),
            "hour_cos": np.random.uniform(-1, 1, 200)
        })
        
        # Check for drift
        results = detector.detect_ks_drift(production_data)
        
        # Similar data should not show significant drift
        drift_ratio = sum(1 for r in results.values() if r.is_drift) / len(results)
        assert drift_ratio < 0.5
    
    def test_drift_detection_detects_shift(self, training_data):
        """Test that drift detector catches distribution shift."""
        detector = DriftDetector(significance_level=0.05)
        detector.fit(training_data)
        
        # Create shifted production data
        np.random.seed(44)
        shifted_data = pd.DataFrame({
            "close_lag_1": np.random.normal(1.15, 0.01, 200),  # Mean shifted
            "close_rolling_mean_24": np.random.normal(1.15, 0.005, 200),  # Mean shifted
            "close_rolling_std_24": np.random.uniform(0.001, 0.003, 200),
            "log_return": np.random.normal(0.01, 0.001, 200),  # Mean shifted
            "hour_sin": np.random.uniform(-1, 1, 200),
            "hour_cos": np.random.uniform(-1, 1, 200)
        })
        
        results = detector.detect_ks_drift(shifted_data)
        
        # Should detect drift in shifted features
        drifted_features = [r.feature_name for r in results.values() if r.is_drift]
        assert "close_lag_1" in drifted_features or "close_rolling_mean_24" in drifted_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
