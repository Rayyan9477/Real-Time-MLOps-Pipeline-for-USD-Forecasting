"""
FastAPI prediction service for USD volatility forecasting.
Includes health checks, Prometheus metrics, drift monitoring, and UI dashboard.
Now with local model loading and prediction storage.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
import logging
import sqlite3
from collections import deque
import threading

from config.config import MODELS_DIR, DATA_DIR, MONITORING_CONFIG
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("prediction_api")

# Get the base path for UI files
UI_DIR = Path(__file__).parent.parent / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

# Database path for storing predictions
DB_PATH = DATA_DIR / "predictions.db"

# Initialize FastAPI app
app = FastAPI(
    title="USD Volatility Prediction API",
    description="Real-time prediction service for USD volatility forecasting",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

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

# Global state
model = None
model_version = None
model_metadata = {}
feature_statistics = {}
recent_predictions = deque(maxlen=100)  # Keep last 100 predictions in memory
latency_history = deque(maxlen=100)  # Track latencies


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
    
    model_config = {"protected_namespaces": ()}
    
    prediction: float = Field(..., description="Predicted volatility value")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    drift_detected: bool = Field(..., description="Whether data drift was detected")
    drift_ratio: float = Field(..., description="Ratio of out-of-distribution features")
    latency_ms: float = Field(default=0.0, description="Prediction latency in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


def init_database():
    """Initialize SQLite database for storing predictions."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction REAL NOT NULL,
                features TEXT NOT NULL,
                latency_ms REAL,
                drift_detected INTEGER,
                drift_ratio REAL,
                model_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {DB_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


def save_prediction_to_db(prediction_data: dict):
    """Save prediction to SQLite database."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, prediction, features, latency_ms, drift_detected, drift_ratio, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_data['timestamp'],
            prediction_data['prediction'],
            json.dumps(prediction_data.get('features', {})),
            prediction_data.get('latency_ms', 0),
            1 if prediction_data.get('drift_detected', False) else 0,
            prediction_data.get('drift_ratio', 0),
            prediction_data.get('model_version', 'unknown')
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")


def load_local_model():
    """Load model from local files instead of MLflow."""
    global model, model_version, model_metadata, feature_statistics
    
    try:
        logger.info("Loading model from local storage...")
        
        # Try to load latest model
        latest_model_path = MODELS_DIR / "latest_model.pkl"
        latest_metadata_path = MODELS_DIR / "latest_metadata.json"
        
        if not latest_model_path.exists():
            # Fallback to finding any model
            model_files = sorted(MODELS_DIR.glob("xgboost_model_*.pkl"))
            if model_files:
                latest_model_path = model_files[-1]
                # Try to find matching metadata
                timestamp = latest_model_path.stem.replace("xgboost_model_", "")
                latest_metadata_path = MODELS_DIR / f"model_metadata_{timestamp}.json"
            else:
                logger.error("No model files found in models directory")
                return False
        
        # Load model
        with open(latest_model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from: {latest_model_path}")
        
        # Load metadata
        if latest_metadata_path.exists():
            with open(latest_metadata_path, 'r') as f:
                model_metadata = json.load(f)
            
            model_version = model_metadata.get('timestamp', 'unknown')
            
            # Calculate feature statistics from metadata for drift detection
            feature_names = model_metadata.get('feature_names', [])
            logger.info(f"Model metadata loaded. Features: {len(feature_names)}")
        else:
            model_version = "local"
            logger.warning("Model metadata not found")
        
        logger.info(f"Model loaded successfully. Version: {model_version}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load local model: {str(e)}")
        return False


# Initialize Prometheus instrumentation at module level
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.on_event("startup")
async def startup_event():
    """Initialize model and database on startup."""
    logger.info("Starting USD Volatility Prediction API...")
    
    # Initialize database
    init_database()
    
    # Load model
    success = load_local_model()
    
    if not success:
        logger.warning("API started without model. Model loading failed.")
    else:
        logger.info("API started successfully with model loaded ✓")


# ==================== UI Routes ====================
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard UI."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse(content="<h1>Dashboard templates not found</h1>", status_code=404)


@app.get("/ui", response_class=HTMLResponse)
async def ui_redirect(request: Request):
    """Redirect to dashboard."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse(content="<h1>Dashboard templates not found</h1>", status_code=404)


# ==================== API Routes ====================
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "USD Volatility Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "dashboard": "/dashboard"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


def detect_drift(features: Dict[str, float]) -> tuple:
    """
    Detect if input features are out of distribution.
    Uses simple z-score based detection.
    """
    # For now, use a simpler detection based on expected ranges
    z_threshold = MONITORING_CONFIG.get("drift_zscore_threshold", 3.0)
    
    # Expected feature ranges (approximate)
    expected_ranges = {
        "close_lag_1": (0.9, 1.2),
        "close_rolling_mean_24": (0.9, 1.2),
        "close_rolling_std_24": (0, 0.05),
        "log_return": (-0.1, 0.1),
        "hour_sin": (-1, 1),
        "hour_cos": (-1, 1),
    }
    
    out_of_range = 0
    total = len(features)
    
    for name, value in features.items():
        if name in expected_ranges:
            low, high = expected_ranges[name]
            if value < low or value > high:
                out_of_range += 1
    
    drift_ratio = out_of_range / total if total > 0 else 0.0
    drift_detected = drift_ratio > 0.3  # More than 30% out of range
    
    drift_gauge.set(drift_ratio)
    
    return drift_detected, drift_ratio


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make volatility prediction."""
    if model is None:
        error_counter.inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        start_time = datetime.now(timezone.utc)
        
        # Detect drift
        drift_detected, drift_ratio = detect_drift(request.features)
        
        # Get expected feature names from model metadata
        expected_features = model_metadata.get('feature_names', list(request.features.keys()))
        
        # Build feature vector with correct order
        feature_values = {}
        for feat in expected_features:
            if feat in request.features:
                feature_values[feat] = request.features[feat]
            else:
                # Use default value for missing features
                feature_values[feat] = 0.0
        
        # Prepare features for prediction
        feature_df = pd.DataFrame([feature_values])
        
        # Make prediction
        prediction = float(model.predict(feature_df)[0])
        
        # Calculate latency
        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000)
        latency_history.append(latency_ms)
        
        # Store prediction
        prediction_data = {
            'timestamp': end_time.isoformat(),
            'prediction': prediction,
            'features': request.features,
            'latency_ms': latency_ms,
            'drift_detected': drift_detected,
            'drift_ratio': drift_ratio,
            'model_version': model_version
        }
        
        recent_predictions.appendleft(prediction_data)
        save_prediction_to_db(prediction_data)
        
        logger.info(
            f"Prediction: {prediction:.6f} | "
            f"Latency: {latency_ms:.2f}ms | "
            f"Drift: {drift_detected} ({drift_ratio:.2%})"
        )
        
        return PredictionResponse(
            prediction=prediction,
            model_version=model_version or "unknown",
            timestamp=end_time.isoformat(),
            drift_detected=drift_detected,
            drift_ratio=drift_ratio,
            latency_ms=latency_ms
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
    """Make batch predictions."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        results = []
        for req in requests:
            response = await predict(req)
            results.append(response.model_dump())
        
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
    """Reload model from local storage."""
    try:
        success = load_local_model()
        
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


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model including real metrics."""
    if model is None:
        return {
            "status": "not_loaded",
            "model_loaded": False
        }
    
    return {
        "status": "loaded",
        "model_loaded": True,
        "model_version": model_version,
        "model_type": model_metadata.get("model_type", "XGBRegressor"),
        "metrics": model_metadata.get("metrics", {}),
        "n_features": model_metadata.get("metrics", {}).get("n_features", 0),
        "feature_names": model_metadata.get("feature_names", [])[:10],  # First 10
        "training_timestamp": model_metadata.get("timestamp", "unknown")
    }


@app.get("/api/stats")
async def get_stats():
    """Get real-time API statistics."""
    # Calculate stats from recent predictions
    avg_latency = np.mean(list(latency_history)) if latency_history else 0
    
    # Count drift alerts in recent predictions
    drift_alerts = sum(1 for p in recent_predictions if p.get('drift_detected', False))
    
    # Get model metrics
    metrics = model_metadata.get('metrics', {})
    
    return {
        "total_predictions": int(prediction_counter._value.get()),
        "avg_latency_ms": round(avg_latency, 2),
        "drift_alerts": drift_alerts,
        "model_accuracy": round((1 - metrics.get('mape', 0) / 100) * 100, 1) if metrics else 0,
        "model_metrics": {
            "rmse": round(metrics.get('rmse', 0), 6),
            "mae": round(metrics.get('mae', 0), 6),
            "r2": round(metrics.get('r2', 0), 4),
            "mape": round(metrics.get('mape', 0), 2)
        },
        "model_loaded": model is not None,
        "model_version": model_version
    }


@app.get("/api/predictions/recent")
async def get_recent_predictions(limit: int = 20):
    """Get recent predictions."""
    predictions = list(recent_predictions)[:limit]
    
    return {
        "predictions": [
            {
                "timestamp": p['timestamp'],
                "prediction": round(p['prediction'], 6),
                "latency_ms": round(p['latency_ms'], 2),
                "drift_detected": p['drift_detected'],
                "drift_ratio": round(p['drift_ratio'], 4)
            }
            for p in predictions
        ],
        "count": len(predictions)
    }


@app.get("/api/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get prediction history from database."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, prediction, latency_ms, drift_detected, drift_ratio
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {
            "predictions": [
                {
                    "timestamp": row[0],
                    "prediction": row[1],
                    "latency_ms": row[2],
                    "drift_detected": bool(row[3]),
                    "drift_ratio": row[4]
                }
                for row in rows
            ],
            "count": len(rows)
        }
        
    except Exception as e:
        logger.error(f"Failed to get prediction history: {e}")
        return {"predictions": [], "count": 0}


@app.get("/api/latency/distribution")
async def get_latency_distribution():
    """Get latency distribution for charts."""
    latencies = list(latency_history)
    
    if not latencies:
        return {"buckets": [], "counts": []}
    
    # Create buckets
    buckets = ["<10ms", "10-25ms", "25-50ms", "50-100ms", ">100ms"]
    counts = [0, 0, 0, 0, 0]
    
    for lat in latencies:
        if lat < 10:
            counts[0] += 1
        elif lat < 25:
            counts[1] += 1
        elif lat < 50:
            counts[2] += 1
        elif lat < 100:
            counts[3] += 1
        else:
            counts[4] += 1
    
    return {"buckets": buckets, "counts": counts}


@app.get("/api/drift/history")
async def get_drift_history():
    """Get drift score history."""
    predictions = list(recent_predictions)
    
    if not predictions:
        return {"timestamps": [], "scores": []}
    
    # Get last 24 data points
    data = predictions[:24]
    
    return {
        "timestamps": [p['timestamp'] for p in reversed(data)],
        "scores": [round(p['drift_ratio'] * 100, 1) for p in reversed(data)],
        "threshold": 20  # 20% drift threshold
    }


if __name__ == "__main__":
    import uvicorn
    from config.config import API_CONFIG
    
    uvicorn.run(
        "src.api.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        workers=1,
        log_level="info"
    )
