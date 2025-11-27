# MLOps Dashboard Quick Reference

**Last Updated:** November 26, 2025

## 🚀 One-Command Startup

```bash
./start_infrastructure.sh
```

## 📊 Dashboard URLs

| Dashboard | URL | Login |
|-----------|-----|-------|
| **Airflow** | http://localhost:8080 | airflow / airflow |
| **MLflow** | http://localhost:5000 | (no auth) |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | (no auth) |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin |
| **API Docs** | http://localhost:8000/docs | (no auth) |

## 🔧 Essential Commands

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f [service-name]

# Stop all services
docker-compose down

# Restart service
docker-compose restart [service-name]
```

## 📚 Documentation

- **Full Guide:** [DASHBOARD_ACCESS_GUIDE.md](DASHBOARD_ACCESS_GUIDE.md)
- **DVC Setup:** [DVC_SETUP.md](DVC_SETUP.md)
- **Changes:** [CONFIGURATION_UPDATE_SUMMARY.md](CONFIGURATION_UPDATE_SUMMARY.md)

## 🎯 Grafana Dashboards

1. **USD Volatility Monitoring** (`usd-volatility-monitoring`)
   - Prediction latency
   - Request rate
   - Data drift
   - Error tracking

2. **MLOps Pipeline Overview** (`mlops-pipeline-overview`)
   - Model metrics (R², RMSE, MAPE)
   - API performance
   - Infrastructure health
   - System resources

## 🔍 Troubleshooting

**Service won't start:**
```bash
docker-compose logs [service-name]
docker-compose restart [service-name]
```

**Port conflict:**
```bash
lsof -i :[port]
# Modify docker-compose.yml ports
```

**Reset everything:**
```bash
docker-compose down -v
./start_infrastructure.sh
```

## ✅ Health Check

```bash
curl http://localhost:8080/health  # Airflow
curl http://localhost:5000/health  # MLflow
curl http://localhost:3000/api/health  # Grafana
curl http://localhost:9090/-/healthy  # Prometheus
```

---

**For complete documentation, see:** [DASHBOARD_ACCESS_GUIDE.md](DASHBOARD_ACCESS_GUIDE.md)
