"""
Unit tests for monitoring module.
"""
import pytest
import pandas as pd
import numpy as np
from src.monitoring.drift_detector import DriftDetector, DriftResult
from src.monitoring.alert_manager import AlertManager, AlertRule, AlertSeverity


class TestDriftDetector:
    """Test DriftDetector functionality."""
    
    @pytest.fixture
    def reference_data(self):
        """Create reference data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.normal(100, 10, 1000),
            "feature_3": np.random.uniform(0, 1, 1000)
        })
    
    @pytest.fixture
    def detector(self, reference_data):
        """Create fitted drift detector."""
        detector = DriftDetector(significance_level=0.05)
        detector.fit(reference_data)
        return detector
    
    def test_fit_computes_statistics(self, reference_data):
        """Test that fit() computes reference statistics."""
        detector = DriftDetector()
        detector.fit(reference_data)
        
        assert len(detector.reference_statistics) == 3
        assert "mean" in detector.reference_statistics["feature_1"]
        assert "std" in detector.reference_statistics["feature_1"]
    
    def test_detect_no_drift(self, detector, reference_data):
        """Test no drift detected for similar distribution."""
        np.random.seed(43)
        similar_data = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 500),
            "feature_2": np.random.normal(100, 10, 500),
            "feature_3": np.random.uniform(0, 1, 500)
        })
        
        results = detector.detect_ks_drift(similar_data)
        
        # Most features should not show drift
        drift_count = sum(1 for r in results.values() if r.is_drift)
        assert drift_count <= 1  # At most 1 false positive
    
    def test_detect_drift_shifted_mean(self, detector):
        """Test drift detected when mean shifts."""
        np.random.seed(44)
        shifted_data = pd.DataFrame({
            "feature_1": np.random.normal(5, 1, 500),  # Mean shifted from 0 to 5
            "feature_2": np.random.normal(100, 10, 500),
            "feature_3": np.random.uniform(0, 1, 500)
        })
        
        results = detector.detect_ks_drift(shifted_data)
        
        # feature_1 should show drift
        assert results["feature_1"].is_drift == True
    
    def test_detect_point_drift(self, detector):
        """Test point drift detection using z-scores."""
        # Normal point
        normal_features = {"feature_1": 0.5, "feature_2": 105, "feature_3": 0.5}
        is_drift, ratio, drifted = detector.detect_point_drift(normal_features)
        
        assert is_drift is False
        assert ratio < 0.2
        
        # Anomalous point
        anomalous_features = {"feature_1": 10, "feature_2": 200, "feature_3": 5}
        is_drift, ratio, drifted = detector.detect_point_drift(anomalous_features)
        
        assert len(drifted) > 0
    
    def test_compute_drift_metrics(self, detector, reference_data):
        """Test comprehensive drift metrics computation."""
        np.random.seed(45)
        current_data = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 500),
            "feature_2": np.random.normal(100, 10, 500),
            "feature_3": np.random.uniform(0, 1, 500)
        })
        
        metrics = detector.compute_drift_metrics(current_data)
        
        assert "drift_ratio" in metrics
        assert "num_drifted_features" in metrics
        assert "avg_wasserstein_distance" in metrics


class TestAlertManager:
    """Test AlertManager functionality."""
    
    def test_default_rules_added(self):
        """Test that default rules are added on initialization."""
        manager = AlertManager()
        
        assert len(manager.rules) > 0
        assert "high_latency" in manager.rules
        assert "high_drift" in manager.rules
    
    def test_check_metrics_triggers_alert(self):
        """Test that checking metrics triggers appropriate alerts."""
        manager = AlertManager()
        
        # Metrics that should trigger high_latency alert
        metrics = {
            "prediction_latency_seconds": 0.8,
            "drift_ratio": 0.1
        }
        
        alerts = manager.check_metrics(metrics)
        
        assert len(alerts) >= 1
        alert_names = [a.name for a in alerts]
        assert "high_latency" in alert_names
    
    def test_check_metrics_no_alert(self):
        """Test that normal metrics don't trigger alerts."""
        manager = AlertManager()
        
        # Normal metrics
        metrics = {
            "prediction_latency_seconds": 0.1,
            "drift_ratio": 0.05
        }
        
        alerts = manager.check_metrics(metrics)
        
        assert len(alerts) == 0
    
    def test_add_custom_rule(self):
        """Test adding custom alert rule."""
        manager = AlertManager()
        
        custom_rule = AlertRule(
            name="custom_test",
            metric_name="test_metric",
            threshold=100,
            comparison="gt",
            severity=AlertSeverity.INFO,
            message_template="Test metric {value} exceeds {threshold}"
        )
        
        manager.add_rule(custom_rule)
        
        assert "custom_test" in manager.rules
        
        # Test that it triggers
        alerts = manager.check_metrics({"test_metric": 150})
        assert len(alerts) == 1
        assert alerts[0].name == "custom_test"
    
    def test_alert_history(self):
        """Test alert history tracking."""
        manager = AlertManager()
        
        # Trigger some alerts
        manager.check_metrics({"prediction_latency_seconds": 0.8})
        manager.check_metrics({"drift_ratio": 0.3})
        
        history = manager.get_alert_history()
        assert len(history) >= 2
    
    def test_alert_summary(self):
        """Test alert summary generation."""
        manager = AlertManager()
        
        manager.check_metrics({"prediction_latency_seconds": 0.8})
        manager.check_metrics({"drift_ratio": 0.6})
        
        summary = manager.get_alert_summary()
        
        assert "total" in summary
        assert "by_severity" in summary
        assert summary["total"] >= 2
    
    def test_callback_execution(self):
        """Test that callbacks are executed on alert."""
        manager = AlertManager()
        callback_results = []
        
        def test_callback(alert):
            callback_results.append(alert.name)
        
        manager.add_callback(test_callback)
        manager.check_metrics({"prediction_latency_seconds": 0.8})
        
        assert len(callback_results) == 1
        assert "high_latency" in callback_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
