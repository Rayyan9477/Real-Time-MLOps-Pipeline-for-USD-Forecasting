"""
Alert management module for monitoring and notifications.
Handles threshold-based alerting for drift and performance metrics.
"""
import logging
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("alert_manager")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Container for alert information."""
    name: str
    message: str
    severity: AlertSeverity
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'gte', 'lte'
    severity: AlertSeverity
    message_template: str
    
    def check(self, value: float) -> bool:
        """Check if alert should be triggered."""
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        return False


class AlertManager:
    """Manage monitoring alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.callbacks: List[Callable[[Alert], None]] = []
        
        # Add default rules
        self._add_default_rules()
        
        logger.info("Initialized AlertManager with default rules")
    
    def _add_default_rules(self):
        """Add default monitoring rules."""
        default_rules = [
            AlertRule(
                name="high_latency",
                metric_name="prediction_latency_seconds",
                threshold=0.5,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                message_template="Prediction latency ({value:.3f}s) exceeds threshold ({threshold:.3f}s)"
            ),
            AlertRule(
                name="critical_latency",
                metric_name="prediction_latency_seconds",
                threshold=1.0,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                message_template="CRITICAL: Prediction latency ({value:.3f}s) exceeds {threshold:.3f}s"
            ),
            AlertRule(
                name="high_drift",
                metric_name="drift_ratio",
                threshold=0.2,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                message_template="Data drift ratio ({value:.2%}) exceeds threshold ({threshold:.2%})"
            ),
            AlertRule(
                name="critical_drift",
                metric_name="drift_ratio",
                threshold=0.5,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                message_template="CRITICAL: Data drift ratio ({value:.2%}) exceeds {threshold:.2%}"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                threshold=0.05,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                message_template="Error rate ({value:.2%}) exceeds threshold ({threshold:.2%})"
            ),
            AlertRule(
                name="model_degradation",
                metric_name="rmse_increase_pct",
                threshold=0.1,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                message_template="Model RMSE increased by {value:.1%}, exceeds threshold ({threshold:.1%})"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule):
        """
        Add an alert rule.
        
        Args:
            rule: AlertRule to add
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of rule to remove
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """
        Add a callback function to be called when alerts are triggered.
        
        Args:
            callback: Function that takes an Alert as argument
        """
        self.callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")
    
    def check_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check metrics against all rules and trigger alerts.
        
        Args:
            metrics: Dictionary of metric names to values
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule_name, rule in self.rules.items():
            if rule.metric_name in metrics:
                value = metrics[rule.metric_name]
                
                if rule.check(value):
                    alert = Alert(
                        name=rule.name,
                        message=rule.message_template.format(
                            value=value,
                            threshold=rule.threshold
                        ),
                        severity=rule.severity,
                        metric_name=rule.metric_name,
                        metric_value=value,
                        threshold=rule.threshold
                    )
                    
                    triggered_alerts.append(alert)
                    self.alert_history.append(alert)
                    
                    # Call registered callbacks
                    for callback in self.callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")
                    
                    logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        return triggered_alerts
    
    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get alert history, optionally filtered by severity.
        
        Args:
            severity: Filter by severity level
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        alerts = self.alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts[-limit:]
    
    def get_alert_summary(self) -> Dict:
        """
        Get summary of alert history.
        
        Returns:
            Dictionary with alert counts by severity
        """
        summary = {
            "total": len(self.alert_history),
            "by_severity": {
                "info": 0,
                "warning": 0,
                "critical": 0
            },
            "by_metric": {}
        }
        
        for alert in self.alert_history:
            summary["by_severity"][alert.severity.value] += 1
            
            if alert.metric_name not in summary["by_metric"]:
                summary["by_metric"][alert.metric_name] = 0
            summary["by_metric"][alert.metric_name] += 1
        
        return summary
    
    def clear_history(self):
        """Clear alert history."""
        self.alert_history = []
        logger.info("Cleared alert history")


def log_alert(alert: Alert):
    """Default callback to log alerts."""
    log_level = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.CRITICAL: logging.ERROR
    }
    logger.log(log_level[alert.severity], f"[ALERT] {alert.name}: {alert.message}")
