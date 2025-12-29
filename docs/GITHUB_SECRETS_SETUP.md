# GitHub Secrets Configuration Guide

## 📝 Required Secrets Setup

To run this project in production with automated 2-hour updates via GitHub Actions, you need to configure the following secrets in your GitHub repository.

### Setting Up Secrets
1. Go to your GitHub repository
2. Click `Settings` → `Secrets and variables` → `Actions`
3. Click `New repository secret`
4. Add each secret below

---

## 🔑 Mandatory Secrets

### 1. TWELVE_DATA_API_KEY
**Purpose:** Fetch USD/EUR forex data every 2 hours

**How to get:**
- Sign up at https://twelvedata.com
- Free tier includes: 800 API calls/day (sufficient for 2-hour updates)
- Copy your API key from the dashboard

**Set in GitHub:**
```
Name: TWELVE_DATA_API_KEY
Value: YOUR_TWELVE_DATA_API_KEY_HERE
```

---

### 2. MLflow Tracking (Optional but Recommended)

#### MLFLOW_TRACKING_URI
```
Name: MLFLOW_TRACKING_URI
Value: https://dagshub.com/YOUR_USERNAME/Real-Time-MLOps-Pipeline-for-USD-Forecasting.mlflow
```

#### MLFLOW_TRACKING_USERNAME
```
Name: MLFLOW_TRACKING_USERNAME
Value: YOUR_DAGSHUB_USERNAME
```

#### MLFLOW_TRACKING_PASSWORD
```
Name: MLFLOW_TRACKING_PASSWORD
Value: YOUR_DAGSHUB_TOKEN
```

#### DAGSHUB_TOKEN
```
Name: DAGSHUB_TOKEN
Value: YOUR_DAGSHUB_TOKEN
```

**How to get DagsHub credentials:**
- Sign up at https://dagshub.com
- Connect your GitHub repository
- Generate access token from Settings → Tokens

---

### 3. Docker Hub (For Container Deployment)

#### DOCKER_USERNAME
```
Name: DOCKER_USERNAME
Value: YOUR_DOCKERHUB_USERNAME
```

#### DOCKER_PASSWORD
```
Name: DOCKER_PASSWORD  
Value: YOUR_DOCKERHUB_ACCESS_TOKEN
```

**How to get:**
- Sign up at https://hub.docker.com
- Go to Account Settings → Security → New Access Token

---

### 4. Railway (For API Deployment - Optional)

#### RAILWAY_TOKEN
```
Name: RAILWAY_TOKEN
Value: YOUR_RAILWAY_API_TOKEN
```

**How to get:**
- Sign up at https://railway.app
- Go to Account → Tokens → Create Token

---

## ✅ Verification Checklist

After setting up secrets, verify in GitHub Actions:

1. ✅ **TWELVE_DATA_API_KEY** - Required for data pipeline
2. ✅ **MLFLOW_TRACKING_URI** - Optional (can comment out in workflows)
3. ✅ **MLFLOW_TRACKING_USERNAME** - Optional
4. ✅ **MLFLOW_TRACKING_PASSWORD** - Optional
5. ✅ **DAGSHUB_TOKEN** - Optional
6. ✅ **DOCKER_USERNAME** - Optional (for Docker builds)
7. ✅ **DOCKER_PASSWORD** - Optional (for Docker builds)
8. ✅ **RAILWAY_TOKEN** - Optional (for deployment)

---

## 🚀 Testing the Setup

### 1. Manual Workflow Trigger
- Go to `Actions` tab
- Select `Data Pipeline - Scheduled Updates`
- Click `Run workflow`
- Check logs for any secret-related errors

### 2. Check Workflow Status
```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/Real-Time-MLOps-Pipeline-for-USD-Forecasting.git
cd Real-Time-MLOps-Pipeline-for-USD-Forecasting

# Check GitHub Actions status
gh run list
```

---

## 🔒 Security Best Practices

1. **Never commit secrets** to the repository
2. **Rotate API keys** regularly (every 90 days)
3. **Use read-only tokens** where possible
4. **Enable 2FA** on all service accounts
5. **Monitor API usage** to detect unauthorized access

---

## 🎯 Minimal Setup (Only Data Pipeline)

If you only want the automated data updates (every 2 hours), you need:

**Required:**
- `TWELVE_DATA_API_KEY`

**Optional (can be removed from workflows):**
- All MLflow secrets
- All Docker secrets
- Railway token

To disable optional features, comment out the relevant steps in:
- `.github/workflows/data-pipeline.yml`
- `.github/workflows/ci-cd.yml`

---

## 🐛 Troubleshooting

### Secret Not Found Error
```
Error: Input required and not supplied: TWELVE_DATA_API_KEY
```
**Solution:** Ensure the secret name exactly matches (case-sensitive)

### API Rate Limit Exceeded
```
Error: API rate limit exceeded
```
**Solution:** Upgrade Twelve Data plan or reduce update frequency in cron schedule

### MLflow Connection Failed
```
Error: Could not connect to MLflow tracking server
```
**Solution:** Comment out MLflow-related steps in workflows (optional feature)

---

## 📞 Support

- **Documentation:** `/docs/PRODUCTION_DEPLOYMENT.md`
- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/Real-Time-MLOps-Pipeline-for-USD-Forecasting/issues)

---

**Last Updated:** December 2025
