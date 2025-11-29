"""
Drift detection module for monitoring data and model performance drift.
Uses statistical tests to detect distribution changes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import ks_2samp, wasserstein_distance
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("drift_detector")


@dataclass
class DriftResult:
    """Container for drift detection results."""
    feature_name: str
    is_drift: bool
    p_value: float
    test_statistic: float
    test_type: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "is_drift": self.is_drift,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "test_type": self.test_type
        }


class DriftDetector:
    """Detect data drift using statistical tests."""
    
    def __init__(
        self,
        significance_level: float = 0.05,
        z_score_threshold: float = 3.0
    ):
        """
        Initialize drift detector.
        
        Args:
            significance_level: P-value threshold for drift detection
            z_score_threshold: Z-score threshold for out-of-distribution detection
        """
        self.significance_level = significance_level
        self.z_score_threshold = z_score_threshold
        self.reference_statistics: Dict[str, Dict] = {}
        
        logger.info(
            f"Initialized DriftDetector with significance={significance_level}, "
            f"z_score_threshold={z_score_threshold}"
        )
    
    def fit(self, reference_data: pd.DataFrame) -> "DriftDetector":
        """
        Compute reference statistics from training data.
        
        Args:
            reference_data: Training/reference DataFrame
            
        Returns:
            Self for chaining
        """
        self.reference_statistics = {}
        
        for column in reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_statistics[column] = {
                "mean": reference_data[column].mean(),
                "std": reference_data[column].std(),
                "min": reference_data[column].min(),
                "max": reference_data[column].max(),
                "median": reference_data[column].median(),
                "q1": reference_data[column].quantile(0.25),
                "q3": reference_data[column].quantile(0.75),
                "distribution": reference_data[column].dropna().values
            }
        
        logger.info(f"Computed reference statistics for {len(self.reference_statistics)} features")
        return self
    
    def detect_ks_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, DriftResult]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            current_data: Current data to check for drift
            features: Specific features to check (all if None)
            
        Returns:
            Dictionary of feature names to DriftResult
        """
        if not self.reference_statistics:
            raise ValueError("Must call fit() first with reference data")
        
        features = features or list(self.reference_statistics.keys())
        results = {}
        
        for feature in features:
            if feature not in self.reference_statistics:
                continue
            
            if feature not in current_data.columns:
                continue
            
            reference = self.reference_statistics[feature]["distribution"]
            current = current_data[feature].dropna().values
            
            if len(current) < 10:
                continue
            
            statistic, p_value = ks_2samp(reference, current)
            is_drift = p_value < self.significance_level
            
            results[feature] = DriftResult(
                feature_name=feature,
                is_drift=is_drift,
                p_value=p_value,
                test_statistic=statistic,
                test_type="kolmogorov_smirnov"
            )
        
        drift_count = sum(1 for r in results.values() if r.is_drift)
        logger.info(f"KS drift detection: {drift_count}/{len(results)} features drifted")
        
        return results
    
    def detect_point_drift(
        self,
        features: Dict[str, float]
    ) -> Tuple[bool, float, List[str]]:
        """
        Detect drift for a single data point using z-scores.
        
        Args:
            features: Dictionary of feature names to values
            
        Returns:
            Tuple of (drift_detected, drift_ratio, drifted_features)
        """
        if not self.reference_statistics:
            raise ValueError("Must call fit() first with reference data")
        
        drifted_features = []
        total_features = 0
        
        for feature_name, value in features.items():
            if feature_name not in self.reference_statistics:
                continue
            
            stats = self.reference_statistics[feature_name]
            mean = stats["mean"]
            std = stats["std"]
            
            if std > 0:
                z_score = abs((value - mean) / std)
                if z_score > self.z_score_threshold:
                    drifted_features.append(feature_name)
            
            total_features += 1
        
        drift_ratio = len(drifted_features) / total_features if total_features > 0 else 0.0
        drift_detected = drift_ratio > 0.2  # More than 20% features drifted
        
        return drift_detected, drift_ratio, drifted_features
    
    def compute_drift_metrics(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute comprehensive drift metrics.
        
        Args:
            current_data: Current data to analyze
            
        Returns:
            Dictionary of drift metrics
        """
        ks_results = self.detect_ks_drift(current_data)
        
        total_features = len(ks_results)
        drifted_features = sum(1 for r in ks_results.values() if r.is_drift)
        
        # Compute Wasserstein distances
        wasserstein_distances = []
        for feature, stats in self.reference_statistics.items():
            if feature in current_data.columns:
                reference = stats["distribution"]
                current = current_data[feature].dropna().values
                if len(current) > 0:
                    wd = wasserstein_distance(reference, current)
                    wasserstein_distances.append(wd)
        
        avg_wasserstein = np.mean(wasserstein_distances) if wasserstein_distances else 0.0
        
        return {
            "drift_ratio": drifted_features / total_features if total_features > 0 else 0.0,
            "num_drifted_features": drifted_features,
            "total_features": total_features,
            "avg_wasserstein_distance": avg_wasserstein,
            "max_wasserstein_distance": max(wasserstein_distances) if wasserstein_distances else 0.0
        }
    
    def get_statistics(self, feature_name: str) -> Optional[Dict]:
        """Get reference statistics for a feature."""
        if feature_name in self.reference_statistics:
            stats = self.reference_statistics[feature_name].copy()
            # Don't return the full distribution array
            stats.pop("distribution", None)
            return stats
        return None
