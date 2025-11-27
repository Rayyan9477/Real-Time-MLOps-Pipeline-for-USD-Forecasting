# USD Volatility Prediction - Real-Time MLOps Pipeline

A production-grade MLOps pipeline for real-time USD volatility forecasting using EUR/USD forex data. This project demonstrates end-to-end ML lifecycle management with automated data ingestion, model training, deployment, and monitoring.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

### Problem Statement
Predict next-hour USD volatility using EUR/USD forex pair data with hourly granularity. The system automatically adapts to concept drift (market regime changes) through continuous monitoring and retraining.

### Key Features
- ✅ **Automated Data Pipeline**: Airflow DAG for ETL with quality gates
- ✅ **Feature Engineering**: Lag features, rolling statistics, time encodings
- ✅ **Experiment Tracking**: MLflow integration with PostgreSQL + MinIO
- ✅ **Data Versioning**: DVC with Google Drive remote storage
- ✅ **CI/CD Pipeline**: GitHub Actions with CML for model comparison
- ✅ **Model Serving**: FastAPI REST API with Prometheus metrics
- ✅ **Monitoring**: Grafana dashboards with drift detection and alerts
- ✅ **Containerization**: Docker deployment ready for production

## 🏗️ Architecture

```
┌─────────────────┐
│  Twelve Data    │
│      API        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Airflow DAG    │ ◄── Scheduled Daily
│   (ETL Pipeline)│
├─────────────────┤
│ 1. Extract      │ → Quality Checks
│ 2. Transform    │ → Feature Engineering
│ 3. Load         │ → MinIO Storage
│ 4. Version      │ → DVC + Google Drive
│ 5. Log          │ → MLflow Artifacts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │
│   (train.py)    │
├─────────────────┤
│ • XGBoost       │
│ • TimeSeriesSplit│
│ • Drift Detection│
│ • MLflow Tracking│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLflow Registry│ ◄── PostgreSQL + MinIO
│  (Production)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Service│
│   (Docker)      │
├─────────────────┤
│ • /predict      │
│ • /health       │
│ • /metrics      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Prometheus    │ ──► │    Grafana      │
│   (Metrics)     │     │  (Dashboard)    │
└─────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git
- Twelve Data API key ([Get free key](https://twelvedata.com/register))
- Google Drive account for DVC storage ([Setup guide](DVC_SETUP.md))

### 1. Clone Repository
```bash
git clone https://github.com/Rayyan9477/Real-Time-MLOps-Pipeline-for-USD-Forecasting.git
cd Real-Time-MLOps-Pipeline-for-USD-Forecasting
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 3. Configure Environment Variables
Edit `.env` file with your credentials:
```bash
# Twelve Data API
TWELVE_DATA_API_KEY=your_api_key_here

# Dagshub
DAGSHUB_REPO_OWNER=your_username
DAGSHUB_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/your_username/Real-Time-MLOps-Pipeline-for-USD-Forecasting.mlflow

# Docker Hub (for deployment)
DOCKER_USERNAME=your_docker_username
```

### 4. Initialize DVC with Google Drive
```bash
# Install DVC with Google Drive support
pip install dvc[gdrive]

# Initialize DVC
dvc init

# Add Google Drive remote
# Replace FOLDER_ID with your Google Drive folder ID
dvc remote add -d gdrive gdrive://FOLDER_ID

# Configure authentication (interactive)
dvc remote modify gdrive gdrive_use_service_account false

# For CI/CD, use service account (see DVC_SETUP.md)
```

For detailed DVC setup with Google Drive, see **[DVC_SETUP.md](DVC_SETUP.md)**.

### 5. Start Infrastructure
```bash
# Start Airflow, MinIO, Prometheus, Grafana
docker-compose up -d

# Check services
docker-compose ps
```

**Access Services:**
- Airflow UI: http://localhost:8080 (airflow/airflow)
- MLflow UI: http://localhost:5000 (no auth)
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
- Prometheus: http://localhost:9090 (no auth)
- Grafana: http://localhost:3000 (admin/admin)
- FastAPI Docs: http://localhost:8000/docs (no auth)

**📊 View Dashboards:**
- **Airflow**: Monitor ETL pipeline execution and DAG status
- **MLflow**: Track experiments, compare models, manage model registry
- **Grafana**: Real-time metrics, alerts, and performance monitoring
  - USD Volatility Prediction Monitoring (`usd-volatility-monitoring`)
  - MLOps Pipeline Overview (`mlops-pipeline-overview`)

For detailed dashboard configuration and usage, see **[DASHBOARD_ACCESS_GUIDE.md](DASHBOARD_ACCESS_GUIDE.md)**.

## 📊 Usage

### Run ETL Pipeline
```bash
# Trigger Airflow DAG manually
curl -X POST "http://localhost:8080/api/v1/dags/usd_volatility_etl_pipeline/dagRuns" \
  -H "Content-Type: application/json" \
  -u "airflow:airflow" \
  -d '{"conf":{}}'
```

### Train Model
```bash
# Train with default hyperparameters
python src/models/train.py

# Train with custom hyperparameters
python src/models/train.py --n-estimators 150 --max-depth 7 --learning-rate 0.05
```

### Start Prediction API
```bash
# Run locally
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t usd-volatility-predictor .
docker run -p 8000:8000 --env-file .env usd-volatility-predictor
```

### Make Predictions
```bash
# Test health endpoint
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "close_lag_1": 1.0854,
      "close_rolling_mean_24": 1.0850,
      "close_rolling_std_24": 0.0015,
      "hour_sin": 0.5,
      "hour_cos": 0.866,
      "log_return": 0.0002
    }
  }'

# Check metrics
curl http://localhost:8000/metrics
```

## 🔄 CI/CD Workflow

### Branch Strategy
```
feature → dev → test → master
          ↓       ↓        ↓
       Lint    Train   Deploy
       Test     CML    Docker
```

### GitHub Actions Workflows

1. **Feature → Dev**: Code quality checks (Black, Flake8, Pylint, PyTest)
2. **Dev → Test**: Full training pipeline + CML metric comparison
3. **Test → Master**: Docker build, push to registry, deployment verification

### Setting Up CI/CD
Add GitHub Secrets:
```
DAGSHUB_TOKEN
DAGSHUB_USERNAME
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
TWELVE_DATA_API_KEY
DOCKER_USERNAME
DOCKER_PASSWORD
```

## 📈 Monitoring & Observability

### Grafana Dashboard
Access: http://localhost:3000

**Panels:**
- Prediction latency (avg, P95, P99)
- Request rate
- Data drift ratio
- Error rate
- Total predictions

**Alerts:**
- High latency (>500ms)
- High drift (>20%)

### Prometheus Metrics
- `predictions_total`: Total predictions made
- `prediction_latency_seconds`: Prediction latency histogram
- `data_drift_ratio`: Current drift ratio
- `prediction_errors_total`: Total errors

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/ -v
```

## 📁 Project Structure

```
├── airflow/
│   └── dags/
│       └── etl_dag.py              # ETL orchestration
├── config/
│   ├── config.py                   # Configuration management
│   ├── prometheus.yml              # Prometheus config
│   └── grafana/                    # Grafana dashboards
├── src/
│   ├── data/
│   │   ├── extraction.py           # Data fetching & validation
│   │   └── transformation.py       # Feature engineering
│   ├── models/
│   │   └── train.py                # Model training
│   ├── api/
│   │   └── app.py                  # FastAPI service
│   └── utils/
│       ├── logger.py               # Logging utilities
│       └── storage.py              # MinIO client
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
├── .github/
│   └── workflows/                  # CI/CD pipelines
├── docker-compose.yml              # Infrastructure stack
├── Dockerfile                      # API container
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | Apache Airflow |
| Data Versioning | DVC + Dagshub |
| Experiment Tracking | MLflow + Dagshub |
| Model Training | XGBoost, scikit-learn |
| API Framework | FastAPI |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker |
| CI/CD | GitHub Actions + CML |
| Object Storage | MinIO |
| Data Source | Twelve Data API |

## 📊 Model Performance

**Typical Metrics (EUR/USD Hourly Volatility):**
- RMSE: ~0.0008 - 0.0012
- MAE: ~0.0005 - 0.0008
- R²: 0.65 - 0.75
- MAPE: 15-25%

*Note: Performance varies based on market conditions and training data.*

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request to `dev` branch

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [Twelve Data](https://twelvedata.com/) for forex data API
- [Dagshub](https://dagshub.com/) for MLOps platform
- [Apache Airflow](https://airflow.apache.org/) for orchestration
- [MLflow](https://mlflow.org/) for experiment tracking

## 📧 Contact

**Rayyan** - [GitHub](https://github.com/Rayyan9477)

**Project Link**: https://github.com/Rayyan9477/Real-Time-MLOps-Pipeline-for-USD-Forecasting

---

**⭐ Star this repo if you find it helpful!**
