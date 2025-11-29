# DVC Configuration for USD Volatility MLOps Pipeline

## Google Drive Storage Setup

### Prerequisites
1. Google account with Google Drive access
2. DVC installed: `pip install dvc[gdrive]`
3. Google Drive folder for data storage

### Setup Instructions

#### 1. Initialize DVC (if not already done)
```bash
dvc init
```

#### 2. Create Google Drive Folder
- Go to https://drive.google.com
- Create a new folder: "MLOps-USD-Forecasting-Data"
- Right-click folder → Share → Copy link
- Extract folder ID from URL: `https://drive.google.com/drive/folders/FOLDER_ID`

#### 3. Add Google Drive as DVC Remote
```bash
# Using folder ID
dvc remote add -d gdrive gdrive://FOLDER_ID

# Example:
# dvc remote add -d gdrive gdrive://1a2b3c4d5e6f7g8h9i0j
```

#### 4. Configure Google Drive Authentication
```bash
# Option A: Interactive authentication (recommended for local)
dvc remote modify gdrive gdrive_use_service_account false

# Option B: Service account (recommended for CI/CD)
# Download service account JSON from Google Cloud Console
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path /path/to/service-account.json
```

#### 5. Add Data Files to DVC Tracking
```bash
# Track processed data
dvc add data/processed/processed_data_*.parquet

# Track raw data
dvc add data/raw/raw_data_*.csv

# Track models
dvc add models/best_model_*.pkl
```

#### 6. Push to Google Drive
```bash
# First push will open browser for authentication
dvc push

# Follow the prompts to authenticate with your Google account
```

#### 7. Commit .dvc Files to Git
```bash
git add data/.gitignore data/processed/*.dvc data/raw/*.dvc models/*.dvc .dvc/config
git commit -m "Add data and models to DVC with Google Drive storage"
git push
```

### Pulling Data from Google Drive

```bash
# Pull all tracked data
dvc pull

# Pull specific file
dvc pull data/processed/processed_data_20251126_130558.parquet.dvc
```

### Google Drive Service Account Setup (For CI/CD)

#### 1. Create Service Account
- Go to https://console.cloud.google.com
- Create new project: "MLOps-USD-Forecasting"
- Enable Google Drive API
- Create Service Account:
  - IAM & Admin → Service Accounts → Create
  - Name: "dvc-storage-service"
  - Role: "Editor" or custom role with Drive access
- Create JSON key → Download

#### 2. Share Drive Folder with Service Account
- Copy service account email: `dvc-storage-service@PROJECT_ID.iam.gserviceaccount.com`
- Go to Google Drive folder
- Right-click → Share → Paste service account email
- Grant "Editor" permissions

#### 3. Configure DVC with Service Account
```bash
# Set service account credentials
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path .secrets/google-service-account.json

# Store credentials securely (don't commit!)
mkdir -p .secrets
cp ~/Downloads/service-account-key.json .secrets/google-service-account.json
echo ".secrets/" >> .gitignore
```

### CI/CD Integration (GitHub Actions)

```yaml
# Add to .github/workflows/ci.yml
steps:
  - name: Set up DVC
    run: pip install dvc[gdrive]
  
  - name: Configure DVC Google Drive
    env:
      GDRIVE_CREDENTIALS_DATA: ${{ secrets.GDRIVE_CREDENTIALS_DATA }}
    run: |
      mkdir -p .secrets
      echo "$GDRIVE_CREDENTIALS_DATA" > .secrets/google-service-account.json
      dvc remote modify gdrive gdrive_use_service_account true
      dvc remote modify gdrive gdrive_service_account_json_file_path .secrets/google-service-account.json
  
  - name: Pull data from Google Drive
    run: dvc pull
```

**GitHub Secret Setup:**
1. Go to repository Settings → Secrets → Actions
2. Add new secret: `GDRIVE_CREDENTIALS_DATA`
3. Paste entire contents of service-account.json

### Alternative: MinIO as Local DVC Remote

```bash
# For local development/testing
dvc remote add -d minio s3://processed-data
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
```

### Verify DVC Configuration

```bash
# Check remote configuration
dvc remote list

# Check DVC status
dvc status

# Verify tracked files
dvc list . --dvc-only
```

### Troubleshooting

**Issue: Authentication failed**
```bash
# Clear cached credentials
rm -rf ~/.config/pydrive2fs/
dvc remote modify gdrive gdrive_acknowledge_abuse true
dvc push
```

**Issue: File not found in Google Drive**
```bash
# Check DVC cache
dvc cache dir

# Verify .dvc file
cat data/processed/processed_data_*.dvc
```

**Issue: Large file upload timeout**
```bash
# Increase timeout
dvc config cache.timeout 600
dvc push
```

### Best Practices

1. **Track Large Binary Files Only**: Only add data/models to DVC, not code
2. **Use .dvcignore**: Exclude unnecessary files
3. **Regular Pushes**: Push after every significant data/model change
4. **Version Control .dvc Files**: Always commit .dvc files to Git
5. **Secure Credentials**: Never commit service account JSON to Git
6. **Monitor Storage**: Check Google Drive storage quota regularly

### Storage Quota Management

```bash
# Check local cache size
du -sh .dvc/cache

# Clean local cache (keeps .dvc files)
dvc gc --workspace

# Remove old versions from remote
dvc gc --cloud --not-in-remote
```
