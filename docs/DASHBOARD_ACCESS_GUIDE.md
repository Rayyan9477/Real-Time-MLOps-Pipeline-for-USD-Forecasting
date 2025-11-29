# MLOps Dashboard Access Guide

Complete guide for accessing and configuring all dashboards in the Real-Time MLOps Pipeline for USD Volatility Forecasting.

---

## 🎯 Quick Access URLs

### For GitHub Codespaces

Replace `{CODESPACE_NAME}` with your codespace name (e.g., `cuddly-eureka-v4pjvxx7vwg24x9`):

| Service | URL | Default Credentials | Purpose |
|---------|-----|---------------------|---------|
| **Dashboard** | `https://{CODESPACE_NAME}-8000.app.github.dev/dashboard` | No auth | Main prediction dashboard |
| **Airflow** | `https://{CODESPACE_NAME}-8080.app.github.dev` | `airflow` / `airflow` | Workflow orchestration |
| **MLflow** | `https://{CODESPACE_NAME}-5000.app.github.dev` | No auth | Experiment tracking |
| **Grafana** | `https://{CODESPACE_NAME}-3000.app.github.dev` | `admin` / `admin` | Metrics visualization |
| **Prometheus** | `https://{CODESPACE_NAME}-9090.app.github.dev` | No auth | Metrics collection |
| **MinIO** | `https://{CODESPACE_NAME}-9001.app.github.dev` | `minioadmin` / `minioadmin` | Object storage |
| **API Docs** | `https://{CODESPACE_NAME}-8000.app.github.dev/docs` | No auth | API documentation |

### For Local Development

| Service | URL | Default Credentials | Purpose |
|---------|-----|---------------------|---------|
| **Dashboard** | http://localhost:8000/dashboard | No auth | Main prediction dashboard |
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` | Workflow orchestration & DAG monitoring |
| **MLflow** | http://localhost:5000 | No auth required | Experiment tracking & model registry |
| **Grafana** | http://localhost:3000 | `admin` / `admin` | Metrics visualization & alerting |
| **Prometheus** | http://localhost:9090 | No auth required | Metrics collection & querying |
| **MinIO** | http://localhost:9001 | `minioadmin` / `minioadmin` | Object storage management |
| **FastAPI** | http://localhost:8000 | No auth required | Prediction API endpoints |
| **API Docs** | http://localhost:8000/docs | No auth required | Interactive API documentation |

---

## 🚀 Infrastructure Startup

### Prerequisites
```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker compose version

# Navigate to project directory
cd /workspaces/Real-Time-MLOps-Pipeline-for-USD-Forecasting
```

### Start All Services
```bash
# Start all infrastructure services
docker compose up -d

# Check service status
docker compose ps

# View logs for specific service
docker compose logs -f airflow-webserver
docker compose logs -f mlflow
docker compose logs -f grafana
```

### Service Health Check
```bash
# Wait for all services to be healthy (may take 1-2 minutes)
watch docker compose ps

# Test service connectivity
curl http://localhost:8080/health  # Airflow
curl http://localhost:5000/health  # MLflow
curl http://localhost:3000/api/health  # Grafana
curl http://localhost:9090/-/healthy  # Prometheus
```

---

## 📊 Dashboard Configurations

### 1. Airflow Dashboard

**Access:** http://localhost:8080

**Features:**
- DAG (Directed Acyclic Graph) monitoring
- Task execution logs
- Scheduler status
- Database connections
- Variable management

**First-Time Setup:**
```bash
# Create admin user (if not auto-created)
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Enable the ETL DAG
# 1. Go to http://localhost:8080
# 2. Login with airflow/airflow
# 3. Click on "etl_pipeline" DAG
# 4. Toggle the switch to "On"
# 5. Click "Trigger DAG" to run manually
```

**Key Pages:**
- **DAGs**: View all workflows
- **Grid View**: Task execution timeline
- **Graph View**: DAG dependency visualization
- **Code**: View DAG Python code
- **Logs**: Task execution logs

**Monitoring ETL Pipeline:**
1. Go to DAGs → `etl_pipeline`
2. Check last run status (green = success, red = failed)
3. Click on task boxes to view logs
4. View task duration in Graph view

---

### 2. MLflow Dashboard

**Access:** http://localhost:5000

**Features:**
- Experiment tracking
- Model versioning
- Parameter comparison
- Artifact storage
- Model registry

**Configuration:**

Backend: PostgreSQL (metrics, params, tags)
Artifact Storage: MinIO (models, plots, data)

**Using MLflow:**

#### A. Tracking Experiments
```python
import mlflow

# Set tracking URI (in your training script)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("usd-volatility-forecasting")

# Log parameters and metrics
with mlflow.start_run(run_name="RandomForest-Optimized"):
    mlflow.log_param("n_estimators", 443)
    mlflow.log_param("max_depth", 12)
    mlflow.log_metric("rmse", 0.000056)
    mlflow.log_metric("r2_score", 0.8401)
    mlflow.log_metric("mape", 4.80)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("reports/optimization/model_comparison.png")
```

#### B. Viewing Experiments
1. Go to http://localhost:5000
2. Select experiment: "usd-volatility-forecasting"
3. Compare runs in table view
4. Sort by metrics (R² Score, RMSE)
5. Click run to see details

#### C. Model Registry
```python
# Register best model
model_uri = "runs:/<RUN_ID>/model"
mlflow.register_model(model_uri, "usd-volatility-predictor")

# Transition to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="usd-volatility-predictor",
    version=1,
    stage="Production"
)
```

**UI Navigation:**
- **Experiments**: View all runs
- **Models**: Registered models
- **Compare**: Side-by-side run comparison
- **Artifacts**: Download logged files

---

### 3. Grafana Dashboard

**Access:** http://localhost:3000

**Login:** `admin` / `admin` (change on first login)

**Pre-configured Dashboards:**

#### Dashboard 1: USD Volatility Prediction Monitoring
**UID:** `usd-volatility-monitoring`

**Panels:**
- Prediction Latency (avg, P95, P99)
- Request Rate (requests/sec)
- Data Drift Ratio
- Error Rate
- Total Predictions Counter
- Current metrics (latency, drift, errors)

**Alerts:**
- High Latency: > 500ms for 5 minutes
- Data Drift: > 20% for 5 minutes

#### Dashboard 2: MLOps Pipeline Overview
**UID:** `mlops-pipeline-overview`

**Sections:**
1. **Model Performance**: R², RMSE, MAPE, Training Samples
2. **API Performance**: Latency distribution, Request rate
3. **Data Pipeline**: Drift monitoring, Feature statistics
4. **Infrastructure Health**: Service status (Airflow, MLflow, MinIO, Postgres)
5. **System Resources**: CPU/Memory by container

**Accessing Dashboards:**
1. Login to Grafana
2. Click "Dashboards" (left sidebar)
3. Select pre-loaded dashboard
4. Or go directly:
   - http://localhost:3000/d/usd-volatility-monitoring
   - http://localhost:3000/d/mlops-pipeline-overview

**Datasource Configuration:**
- Pre-configured Prometheus datasource at `http://prometheus:9090`
- Auto-provisioned on startup
- Manual check: Configuration → Data Sources → Prometheus

**Creating Alerts:**
1. Edit dashboard panel
2. Click "Alert" tab
3. Define alert rule
4. Configure notification channel
5. Save dashboard

---

### 4. Prometheus Dashboard

**Access:** http://localhost:9090

**Features:**
- Metrics scraping
- Query interface (PromQL)
- Target health monitoring
- Alert rules

**Key Metrics:**

```promql
# Prediction latency (average)
rate(prediction_latency_seconds_sum[5m]) / rate(prediction_latency_seconds_count[5m])

# Request rate
rate(predictions_total[1m])

# Data drift ratio
data_drift_ratio

# Model metrics
model_r2_score
model_rmse
model_mape

# Infrastructure health
up{job="airflow-scheduler"}
up{job="mlflow"}
```

**Useful Queries:**

```promql
# Top 10 slowest endpoints
topk(10, rate(prediction_latency_seconds_sum[5m]))

# Error rate percentage
rate(prediction_errors_total[5m]) / rate(predictions_total[5m]) * 100

# 95th percentile latency
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
```

---

### 5. MinIO Dashboard

**Access:** http://localhost:9001

**Login:** `minioadmin` / `minioadmin`

**Features:**
- Bucket management
- Object browser
- Access policy configuration
- Monitoring & metrics

**Buckets:**
- `processed-data`: Processed datasets
- `mlflow-artifacts`: MLflow model artifacts, plots

**Creating Bucket for MLflow:**
```bash
# Access MinIO container
docker-compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin

# Create bucket
docker-compose exec minio mc mb local/mlflow-artifacts

# Set public read policy (optional for dev)
docker-compose exec minio mc policy set download local/mlflow-artifacts
```

**Viewing Artifacts:**
1. Login to MinIO console
2. Navigate to "Buckets"
3. Click `mlflow-artifacts`
4. Browse by experiment run ID

---

### 6. FastAPI Dashboard

**Access:** http://localhost:8000/docs

**Interactive API Documentation:**

**Available Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/metrics` | Prometheus metrics |
| GET | `/health` | Service health status |
| GET | `/model/info` | Model metadata |
| GET | `/drift/status` | Data drift status |

**Testing Predictions:**

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "open": 1.156,
       "high": 1.158,
       "low": 1.155,
       "close": 1.157
     }'

# Get model info
curl http://localhost:8000/model/info

# Check drift status
curl http://localhost:8000/drift/status
```

**Using Swagger UI:**
1. Go to http://localhost:8000/docs
2. Expand endpoint (e.g., POST `/predict`)
3. Click "Try it out"
4. Enter test data
5. Click "Execute"
6. View response

---

## 🔧 Troubleshooting

### Service Not Starting

```bash
# Check logs
docker-compose logs <service-name>

# Restart specific service
docker-compose restart <service-name>

# Recreate service
docker-compose up -d --force-recreate <service-name>
```

### Airflow Issues

```bash
# Reset Airflow database
docker-compose down -v
docker-compose up -d

# Check scheduler logs
docker-compose logs -f airflow-scheduler

# Enter Airflow container
docker-compose exec airflow-webserver bash
airflow dags list
```

### MLflow Connection Error

```bash
# Check MLflow logs
docker-compose logs -f mlflow

# Verify MinIO connection
docker-compose exec mlflow curl http://minio:9000/minio/health/live

# Check PostgreSQL
docker-compose exec postgres psql -U airflow -d mlflow -c "\dt"
```

### Grafana Datasource Not Working

```bash
# Check Prometheus from Grafana container
docker-compose exec grafana wget -O- http://prometheus:9090/-/healthy

# Re-provision datasources
docker-compose restart grafana

# Manual configuration
# Grafana UI → Configuration → Data Sources → Add Prometheus
# URL: http://prometheus:9090
```

### Port Conflicts

```bash
# Check ports in use
lsof -i :8080  # Airflow
lsof -i :5000  # MLflow
lsof -i :3000  # Grafana

# Modify docker-compose.yml ports section
# "HOST_PORT:CONTAINER_PORT"
```

---

## 📈 Monitoring Best Practices

### 1. Dashboard Organization
- **Airflow**: Monitor daily for DAG failures
- **MLflow**: Review weekly for model performance trends
- **Grafana**: Set up alerts for critical metrics
- **Prometheus**: Use for ad-hoc metric queries

### 2. Alert Configuration

**Critical Alerts (immediate action):**
- Prediction latency > 500ms
- Data drift > 20%
- Service down (Airflow, MLflow)

**Warning Alerts (review soon):**
- Prediction latency > 300ms
- Data drift > 15%
- Error rate > 1%

### 3. Regular Health Checks

**Daily:**
- Check Airflow DAG runs
- Review API error rate in Grafana

**Weekly:**
- Compare model metrics in MLflow
- Review drift trends in Grafana
- Check MinIO storage usage

**Monthly:**
- Analyze prediction latency trends
- Review model registry
- Clean up old MLflow experiments

---

## 🔒 Security Recommendations

### Production Deployment

```bash
# Change default passwords
docker-compose exec airflow-webserver airflow users create \
    --username <new-admin> \
    --password <strong-password> \
    --role Admin

# Update Grafana admin password
# Grafana UI → Configuration → Users → admin → Change Password

# Secure MinIO
docker-compose exec minio mc admin user add local <new-user> <strong-password>

# Enable HTTPS (use reverse proxy like nginx)
# Add SSL certificates
# Update docker-compose.yml with SSL configuration
```

### Environment Variables

```bash
# Create .env from .env.example
cp .env.example .env

# Update credentials
vim .env

# Never commit .env to Git
echo ".env" >> .gitignore
```

---

## 📚 Additional Resources

### Documentation Links
- **Airflow**: https://airflow.apache.org/docs/
- **MLflow**: https://mlflow.org/docs/latest/
- **Grafana**: https://grafana.com/docs/grafana/latest/
- **Prometheus**: https://prometheus.io/docs/

### Query Examples
- **PromQL**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **MLflow API**: https://mlflow.org/docs/latest/rest-api.html

### Community Support
- **Stack Overflow**: Tag with `airflow`, `mlflow`, `grafana`
- **GitHub Issues**: Report bugs in respective repositories

---

## ✅ Verification Checklist

After starting infrastructure, verify:

- [ ] Airflow webserver accessible at :8080
- [ ] MLflow UI accessible at :5000
- [ ] Grafana accessible at :3000 with Prometheus datasource
- [ ] Prometheus accessible at :9090 with targets up
- [ ] MinIO console accessible at :9001
- [ ] FastAPI docs accessible at :8000/docs
- [ ] All 2 Grafana dashboards loaded
- [ ] Airflow ETL DAG visible and enabled
- [ ] MLflow connected to PostgreSQL and MinIO
- [ ] Prometheus scraping all configured targets

**Test Complete Pipeline:**
```bash
# Trigger Airflow DAG
curl -X POST http://localhost:8080/api/v1/dags/etl_pipeline/dagRuns \
     -H "Content-Type: application/json" \
     -u airflow:airflow \
     -d '{"conf": {}}'

# Wait for completion, then check:
# 1. Airflow: DAG run success
# 2. MLflow: New experiment run logged
# 3. MinIO: Artifacts stored
# 4. Grafana: Metrics updated
```

---

**Last Updated:** November 26, 2025  
**Version:** 1.0.0  
**Status:** Production Ready
