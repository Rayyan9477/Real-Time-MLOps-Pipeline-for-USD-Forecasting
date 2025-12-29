# 🚀 Production Deployment Guide

## Overview
This guide covers deploying the Real-Time MLOps Pipeline for USD Forecasting to production environments.

## Supported Platforms

### 1. Vercel (Recommended for API)
- ✅ Best for FastAPI backend
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Easy environment variable management

### 2. Railway
- ✅ Full-stack applications
- ✅ Docker support
- ✅ Database hosting
- ✅ Cron jobs support

### 3. Render
- ✅ Free tier available
- ✅ Docker support
- ✅ Cron jobs
- ✅ PostgreSQL hosting

### 4. AWS/GCP/Azure
- ✅ Enterprise-grade
- ✅ Full control
- ✅ Scalable infrastructure

## Pre-Deployment Checklist

### Required API Keys
- [ ] Twelve Data API key ([Get it here](https://twelvedata.com/))
- [ ] DagsHub account and token (optional, for MLflow)
- [ ] Cloud storage credentials (MinIO/S3)

### Environment Variables
Copy `.env.example` to `.env.production` and fill in:
```bash
# Required
TWELVE_DATA_API_KEY=your_api_key_here

# Optional but recommended
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
MLFLOW_TRACKING_URI=your_mlflow_uri
```

## Deployment Instructions

### Vercel Deployment

1. **Install Vercel CLI:**
```bash
npm i -g vercel
```

2. **Login:**
```bash
vercel login
```

3. **Deploy:**
```bash
vercel --prod
```

4. **Set Environment Variables:**
```bash
vercel env add TWELVE_DATA_API_KEY
vercel env add DAGSHUB_USERNAME
vercel env add DAGSHUB_TOKEN
```

5. **Configure Cron (via Vercel Cron):**
- Add cron jobs in `vercel.json` (already configured for 2-hour updates)

### Railway Deployment

1. **Install Railway CLI:**
```bash
npm i -g @railway/cli
```

2. **Login:**
```bash
railway login
```

3. **Initialize Project:**
```bash
railway init
```

4. **Deploy:**
```bash
railway up
```

5. **Set Environment Variables:**
```bash
railway variables set TWELVE_DATA_API_KEY=your_key
```

6. **Add Cron Service:**
```bash
railway service create cron-job
```

### Render Deployment

1. **Connect GitHub Repository:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service:**
   - **Name:** usd-volatility-api
   - **Environment:** Docker
   - **Docker Command:** (use default from Dockerfile)
   - **Plan:** Free or paid

3. **Set Environment Variables:**
   - Add all variables from `.env.example`

4. **Add Cron Job:**
   - Create a new "Cron Job"
   - Schedule: `0 */2 * * *` (every 2 hours)
   - Command: `python -m src.data.data_extraction`

### Docker Deployment (Self-Hosted)

1. **Build Image:**
```bash
docker build -t usd-volatility-predictor:latest .
```

2. **Run Container:**
```bash
docker run -d \
  --name usd-volatility-api \
  -p 8000:8000 \
  --env-file .env.production \
  usd-volatility-predictor:latest
```

3. **Set up Cron (Host Machine):**
```bash
# Add to crontab
0 */2 * * * docker exec usd-volatility-api python -m src.data.data_extraction
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Post-Deployment Verification

### 1. Health Check
```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-29T..."
}
```

### 2. API Test
```bash
curl https://your-domain.com/api/v1/predict
```

### 3. Monitor Logs
```bash
# Vercel
vercel logs

# Railway
railway logs

# Render
# Check dashboard logs

# Docker
docker logs usd-volatility-api
```

## Cron Job Configuration

The project is configured to update data **every 2 hours**:

### Schedule: `0 */2 * * *`
- Runs at: 00:00, 02:00, 04:00, 06:00, ..., 22:00 UTC
- Fetches latest USD/EUR data
- Updates model predictions
- Stores results in database

### Airflow DAG (Alternative)
If using Airflow:
```python
schedule_interval="0 */2 * * *"  # Every 2 hours
```

### Vercel Cron
```json
{
  "crons": [{
    "path": "/api/cron/update-data",
    "schedule": "0 */2 * * *"
  }]
}
```

## Monitoring & Alerting

### Prometheus Metrics
Access at: `https://your-domain.com/metrics`

### Grafana Dashboards
Access at: `https://your-domain.com:3000`
- Default credentials: `admin/admin`

### Health Monitoring
```bash
# Set up monitoring
curl -X POST https://your-domain.com/api/monitoring/enable

# Check drift detection
curl https://your-domain.com/api/monitoring/drift
```

## Scaling Considerations

### API Scaling
```bash
# Increase workers (in .env.production)
API_WORKERS=8

# Enable auto-scaling (platform-specific)
# Vercel: Automatic
# Railway: Configure in dashboard
# Render: Horizontal scaling available
```

### Database Scaling
- Use connection pooling
- Add read replicas
- Implement caching (Redis)

### Model Updates
- Retrain weekly with latest data
- Version models with MLflow
- A/B test before production deployment

## Troubleshooting

### Issue: API Key Invalid
```bash
# Verify API key
curl "https://api.twelvedata.com/time_series?symbol=EUR/USD&apikey=YOUR_KEY"
```

### Issue: Cron Not Running
```bash
# Check cron logs
# Vercel: Check Functions logs
# Railway: Check cron service logs
# Render: Check cron job runs
```

### Issue: Model Not Loading
```bash
# Check model file exists
ls models/latest_model.pkl

# Pull from DVC
dvc pull models/latest_model.pkl.dvc
```

### Issue: High Latency
- Enable caching
- Increase workers
- Optimize model inference
- Use CDN for static assets

## Security Best Practices

1. **Environment Variables:**
   - Never commit `.env` files
   - Use platform secret management
   - Rotate API keys regularly

2. **API Protection:**
   - Enable rate limiting
   - Use API authentication
   - Implement CORS properly

3. **HTTPS:**
   - Always use HTTPS in production
   - Configure SSL certificates
   - Enable HSTS headers

4. **Monitoring:**
   - Set up error tracking (Sentry)
   - Enable audit logging
   - Monitor for anomalies

## Cost Optimization

### Free Tier Recommendations
- **Vercel:** Free for hobby projects
- **Railway:** $5/month credit
- **Render:** Free tier available
- **Twelve Data:** 800 API calls/day free

### Paid Plans
- **Vercel Pro:** $20/month (team features)
- **Railway:** Pay-as-you-go
- **Render:** Starting at $7/month
- **Twelve Data:** Starting at $9.99/month

## Maintenance Schedule

### Daily
- [ ] Monitor API health
- [ ] Check error logs
- [ ] Verify cron execution

### Weekly
- [ ] Review model performance
- [ ] Check data quality
- [ ] Update dependencies

### Monthly
- [ ] Retrain models
- [ ] Review costs
- [ ] Security audit

## Support & Resources

- **Documentation:** [README.md](../README.md)
- **Issues:** [GitHub Issues](https://github.com/Rayyan9477/Real-Time-MLOps-Pipeline-for-USD-Forecasting/issues)
- **Twelve Data API:** [Documentation](https://twelvedata.com/docs)
- **MLflow:** [Documentation](https://mlflow.org/docs/latest/index.html)

## Emergency Contacts

### Platform Support
- Vercel: support@vercel.com
- Railway: help@railway.app
- Render: support@render.com

### Rollback Procedure
```bash
# Vercel
vercel rollback

# Railway
railway rollback

# Docker
docker pull usd-volatility-predictor:previous-tag
docker-compose up -d
```

---

**Last Updated:** December 29, 2025
**Version:** 1.0.0
