"""
FastAPI prediction service for USD volatility forecasting.
Includes health checks, Prometheus metrics, and drift monitoring.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
from scipy import stats
import logging

from config.config import MLFLOW_CONFIG, MONITORING_CONFIG
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("prediction_api")

# Initialize FastAPI app
app = FastAPI(
    title="USD Volatility Prediction API",
    description="Real-time prediction service for USD volatility forecasting",
    version="1.0.0"
)

# Prometheus metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions made'
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

drift_gauge = Gauge(
    'data_drift_ratio',
    'Ratio of out-of-distribution features'
)

error_counter = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)

# Global model variable
model = None
model_version = None
feature_statistics = {}


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and values",
        example={
            "close_lag_1": 1.0854,
            "close_rolling_mean_24": 1.0850,
            "close_rolling_std_24": 0.0015,
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "log_return": 0.0002
        }
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: float = Field(..., description="Predicted volatility value")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    drift_detected: bool = Field(..., description="Whether data drift was detected")
    drift_ratio: float = Field(..., description="Ratio of out-of-distribution features")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


def load_model():
    """Load model from MLflow registry."""
    global model, model_version, feature_statistics
    
    try:
        logger.info("Loading model from MLflow registry...")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        
        # Load production model
        model_name = MLFLOW_CONFIG["model_name"]
        stage = MLFLOW_CONFIG["registry_stage"]
        model_uri = f"models:/{model_name}/{stage}"
        
        logger.info(f"Loading model: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get model version
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        
        if latest_versions:
            model_version = latest_versions[0].version
            logger.info(f"Model loaded successfully. Version: {model_version}")
        else:
            model_version = "unknown"
            logger.warning("Could not determine model version")
        
        # Load feature statistics for drift detection (from training data)
        # In production, this should be loaded from a saved artifact
        feature_statistics = {}  # Placeholder
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting USD Volatility Prediction API...")
    
    success = load_model()
    
    if not success:
        logger.warning("API started without model. Model loading failed.")
    else:
        logger.info("API started successfully with model loaded ✓")
    
    # Initialize Prometheus instrumentation
    Instrumentator().instrument(app).expose(app)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "USD Volatility Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        timestamp=datetime.utcnow().isoformat()
    )


def detect_drift(features: Dict[str, float]) -> tuple[bool, float]:
    """
    Detect if input features are out of distribution.
    
    Args:
        features: Input features
        
    Returns:
        Tuple of (drift_detected, drift_ratio)
    """
    if not feature_statistics:
        # No baseline statistics available
        return False, 0.0
    
    z_threshold = MONITORING_CONFIG["drift_zscore_threshold"]
    out_of_dist_count = 0
    total_features = len(features)
    
    for feature_name, value in features.items():
        if feature_name in feature_statistics:
            mean = feature_statistics[feature_name]["mean"]
            std = feature_statistics[feature_name]["std"]
            
            if std > 0:
                z_score = abs((value - mean) / std)
                if z_score > z_threshold:
                    out_of_dist_count += 1
    
    drift_ratio = out_of_dist_count / total_features if total_features > 0 else 0.0
    drift_detected = drift_ratio > (MONITORING_CONFIG["alert_drift_threshold_pct"] / 100)
    
    # Update Prometheus gauge
    drift_gauge.set(drift_ratio)
    
    return drift_detected, drift_ratio


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make volatility prediction.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response with volatility forecast
    """
    if model is None:
        error_counter.inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        # Start timing
        start_time = datetime.utcnow()
        
        # Detect drift
        drift_detected, drift_ratio = detect_drift(request.features)
        
        # Prepare features for prediction
        feature_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Calculate latency
        end_time = datetime.utcnow()
        latency = (end_time - start_time).total_seconds()
        
        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(latency)
        
        # Log prediction
        logger.info(
            f"Prediction: {prediction:.6f} | "
            f"Latency: {latency:.4f}s | "
            f"Drift: {drift_detected} ({drift_ratio:.2%})"
        )
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version=model_version or "unknown",
            timestamp=end_time.isoformat(),
            drift_detected=drift_detected,
            drift_ratio=drift_ratio
        )
        
    except Exception as e:
        error_counter.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Make batch predictions.
    
    Args:
        requests: List of prediction requests
        
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        results = []
        
        for req in requests:
            response = await predict(req)
            results.append(response.dict())
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        error_counter.inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/reload_model")
async def reload_model():
    """Reload model from registry (for updates)."""
    try:
        success = load_model()
        
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "version": model_version
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reload model"
            )
            
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )


@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_name": MLFLOW_CONFIG["model_name"],
        "model_version": model_version,
        "registry_stage": MLFLOW_CONFIG["registry_stage"],
        "tracking_uri": MLFLOW_CONFIG["tracking_uri"]
    }


if __name__ == "__main__":
    import uvicorn
    from config.config import API_CONFIG
    
    uvicorn.run(
        "src.api.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        workers=1,  # Use 1 worker for development
        log_level="info"
    )
