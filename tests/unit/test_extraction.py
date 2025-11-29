"""
Unit tests for data extraction module.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.data.data_extraction import TwelveDataClient, DataQualityChecker, extract_forex_data


class TestTwelveDataClient:
    """Test TwelveDataClient functionality."""
    
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch('src.data.data_extraction.TWELVE_DATA_CONFIG', {
            "api_key": "",
            "base_url": "https://api.test.com",
            "symbol": "EUR/USD",
            "interval": "1h"
        }):
            with pytest.raises(ValueError, match="TWELVE_DATA_API_KEY not found"):
                TwelveDataClient()
    
    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        with patch('src.data.data_extraction.TWELVE_DATA_CONFIG', {
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "symbol": "EUR/USD",
            "interval": "1h"
        }):
            client = TwelveDataClient("test_key")
            assert client.api_key == "test_key"
            assert client.symbol == "EUR/USD"
    
    @patch('requests.get')
    def test_fetch_time_series_success(self, mock_get):
        """Test successful API call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "values": [
                {"datetime": "2025-11-26 10:00:00", "close": "1.0850"},
                {"datetime": "2025-11-26 11:00:00", "close": "1.0855"}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('src.data.data_extraction.TWELVE_DATA_CONFIG', {
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "symbol": "EUR/USD",
            "interval": "1h",
            "fetch_size": 100
        }):
            client = TwelveDataClient("test_key")
            data = client.fetch_time_series(outputsize=10)
            
            assert "values" in data
            assert len(data["values"]) == 2
    
    @patch('requests.get')
    def test_fetch_time_series_api_error(self, mock_get):
        """Test API error handling."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "error",
            "message": "API key invalid"
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('src.data.data_extraction.TWELVE_DATA_CONFIG', {
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "symbol": "EUR/USD",
            "interval": "1h",
            "fetch_size": 100
        }):
            client = TwelveDataClient("test_key")
            
            with pytest.raises(Exception, match="Twelve Data API Error"):
                client.fetch_time_series()


class TestDataQualityChecker:
    """Test DataQualityChecker functionality."""
    
    def test_check_null_values_pass(self):
        """Test null value check passes with clean data."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-11-26", periods=100, freq="h"),
            "open": [1.085] * 100,
            "high": [1.086] * 100,
            "low": [1.084] * 100,
            "close": [1.085] * 100
        })
        
        assert DataQualityChecker.check_null_values(df) is True
    
    def test_check_null_values_fail(self):
        """Test null value check fails with too many nulls."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-11-26", periods=100, freq="h"),
            "open": [1.085] * 100,
            "high": [1.086] * 100,
            "low": [1.084] * 100,
            "close": [1.085] * 95 + [None] * 5  # 5% nulls
        })
        
        assert DataQualityChecker.check_null_values(df, max_null_pct=1.0) is False
    
    def test_check_schema_pass(self):
        """Test schema check passes with correct columns."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-11-26", periods=10, freq="h"),
            "open": [1.085] * 10,
            "high": [1.086] * 10,
            "low": [1.084] * 10,
            "close": [1.085] * 10
        })
        
        assert DataQualityChecker.check_schema(df) is True
    
    def test_check_schema_fail_missing_column(self):
        """Test schema check fails with missing columns."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-11-26", periods=10, freq="h"),
            "open": [1.085] * 10
        })
        
        assert DataQualityChecker.check_schema(df) is False
    
    def test_check_minimum_data_points_pass(self):
        """Test minimum data points check passes."""
        df = pd.DataFrame({
            "close": [1.085] * 30
        })
        
        assert DataQualityChecker.check_minimum_data_points(df, min_points=24) is True
    
    def test_check_minimum_data_points_fail(self):
        """Test minimum data points check fails."""
        df = pd.DataFrame({
            "close": [1.085] * 10
        })
        
        assert DataQualityChecker.check_minimum_data_points(df, min_points=24) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
