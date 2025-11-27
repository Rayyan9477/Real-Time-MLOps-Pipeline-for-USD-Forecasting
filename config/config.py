"""
Configuration management for the MLOps pipeline.
Loads environment variables and provides centralized configuration.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Twelve Data API Configuration
TWELVE_DATA_CONFIG = {
    "api_key": os.getenv("TWELVE_DATA_API_KEY", ""),
    "base_url": os.getenv("TWELVE_DATA_BASE_URL", "https://api.twelvedata.com"),
    "symbol": os.getenv("FOREX_SYMBOL", "EUR/USD"),
    "interval": os.getenv("DATA_INTERVAL", "1h"),
    "fetch_size": int(os.getenv("FETCH_SIZE", "168")),
}

# MinIO Configuration
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "bucket": os.getenv("MINIO_BUCKET", "processed-data"),
    "secure": os.getenv("MINIO_SECURE", "false").lower() == "true",
}

# MLflow Configuration
MLFLOW_CONFIG = {
    "tracking_uri": os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/your_username/Real-Time-MLOps-Pipeline-for-USD-Forecasting.mlflow"
    ),
    "tracking_username": os.getenv("MLFLOW_TRACKING_USERNAME", ""),
    "tracking_password": os.getenv("MLFLOW_TRACKING_PASSWORD", ""),
    "experiment_name": "usd_volatility_prediction",
    "model_name": os.getenv("MODEL_NAME", "usd_volatility_predictor"),
    "registry_stage": os.getenv("MODEL_REGISTRY_STAGE", "Production"),
}

# Model Configuration
MODEL_CONFIG = {
    "rolling_window_size": int(os.getenv("ROLLING_WINDOW_SIZE", "24")),
    "train_test_split": float(os.getenv("TRAIN_TEST_SPLIT", "0.8")),
    "random_state": 42,
    "xgboost_params": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
}

# Data Quality Configuration
DATA_QUALITY_CONFIG = {
    "max_null_percentage": 1.0,  # Maximum 1% null values allowed
    "min_data_points": 24,  # Minimum data points required for volatility calculation
    "outlier_std_threshold": 3.0,  # Remove outliers beyond 3 standard deviations
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
    "grafana_port": int(os.getenv("GRAFANA_PORT", "3000")),
    "alert_latency_threshold_ms": int(os.getenv("ALERT_LATENCY_THRESHOLD_MS", "500")),
    "alert_drift_threshold_pct": float(os.getenv("ALERT_DRIFT_THRESHOLD_PCT", "20")),
    "drift_zscore_threshold": 3.0,  # Z-score threshold for out-of-distribution detection
}

# API Configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4")),
}

# Docker Configuration
DOCKER_CONFIG = {
    "username": os.getenv("DOCKER_USERNAME", ""),
    "image_name": os.getenv("DOCKER_IMAGE_NAME", "usd-volatility-predictor"),
    "image_tag": os.getenv("DOCKER_IMAGE_TAG", "latest"),
}

# Dagshub Configuration
DAGSHUB_CONFIG = {
    "repo_owner": os.getenv("DAGSHUB_REPO_OWNER", ""),
    "repo_name": os.getenv("DAGSHUB_REPO_NAME", "Real-Time-MLOps-Pipeline-for-USD-Forecasting"),
    "token": os.getenv("DAGSHUB_TOKEN", ""),
}
