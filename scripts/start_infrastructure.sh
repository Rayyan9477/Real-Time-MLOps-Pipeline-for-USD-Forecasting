#!/bin/bash

# MLOps Infrastructure Startup Script for GitHub Codespaces
# Ensures all services start correctly and provides dashboard links

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   🚀 MLOps Pipeline - GitHub Codespaces Startup              ║"
echo "║   Real-Time USD Volatility Forecasting                       ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# GitHub Codespace name
CODESPACE_NAME="cuddly-eureka-v4pjvxx7vwg24x9"

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p airflow/dags airflow/logs airflow/plugins
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p reports/optimization
mkdir -p config/grafana/datasources config/grafana/dashboards
mkdir -p logs
mkdir -p .secrets
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Set proper permissions for Airflow
echo "🔐 Setting permissions..."
export AIRFLOW_UID=50000
echo -e "${GREEN}✅ Permissions configured (AIRFLOW_UID=$AIRFLOW_UID)${NC}"
echo ""

# Stop existing services
echo "🛑 Stopping existing services (if any)..."
docker-compose down 2>/dev/null || true
echo ""

# Start services
echo "🚀 Starting all MLOps services..."
echo ""
docker-compose up -d

# Wait for services to start
echo ""
echo "⏳ Waiting for services to start up..."
echo "   This may take 30-60 seconds..."
echo ""
sleep 15

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   ✅ All Services Started Successfully!                       ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   🌐 GitHub Codespaces Dashboard Links                        ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo "🔹 Grafana Dashboard:"
echo "   https://${CODESPACE_NAME}-3000.githubpreview.dev"
echo "   Username: admin"
echo "   Password: admin"
echo ""

echo "🔹 Airflow Orchestration:"
echo "   https://${CODESPACE_NAME}-8080.githubpreview.dev"
echo "   Username: airflow"
echo "   Password: airflow"
echo ""

echo "🔹 MLflow Tracking:"
echo "   https://${CODESPACE_NAME}-5000.githubpreview.dev"
echo ""

echo "🔹 FastAPI Prediction Service:"
echo "   https://${CODESPACE_NAME}-8000.githubpreview.dev"
echo "   Health Check: /health"
echo "   API Docs: /docs"
echo ""

echo "🔹 Prometheus Monitoring:"
echo "   https://${CODESPACE_NAME}-9090.githubpreview.dev"
echo ""

echo "🔹 MinIO Object Storage:"
echo "   Console: https://${CODESPACE_NAME}-9001.githubpreview.dev"
echo "   API: https://${CODESPACE_NAME}-9000.githubpreview.dev"
echo "   Username: minioadmin"
echo "   Password: minioadmin"
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   📊 Service Status                                           ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check service status
docker-compose ps

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   ✨ Ready to use! Click the links above to access dashboards ✨"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
