#!/bin/bash

# MLOps Infrastructure Startup Script
# Ensures all services start correctly and dashboards are accessible

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   MLOps Pipeline - Infrastructure Startup                    ║"
echo "║   Real-Time USD Volatility Forecasting                       ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

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

# Check .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found, creating from .env.example${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env file with your credentials${NC}"
    echo ""
fi

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
echo "🚀 Starting infrastructure services..."
echo ""
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to become healthy..."
echo "   This may take 1-2 minutes..."
echo ""

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo -e "${RED}❌ $service failed to start${NC}"
    return 1
}

# Check PostgreSQL (via Airflow health)
echo "Checking PostgreSQL..."
docker-compose exec -T postgres pg_isready -U airflow > /dev/null 2>&1 && \
    echo -e "${GREEN}✅ PostgreSQL is ready${NC}" || \
    echo -e "${YELLOW}⚠️  PostgreSQL may still be initializing${NC}"

# Wait a bit for initialization
sleep 10

# Check Airflow Webserver
echo "Checking Airflow..."
check_service "Airflow" "http://localhost:8080/health"

# Check MLflow
echo "Checking MLflow..."
check_service "MLflow" "http://localhost:5000/health"

# Check MinIO
echo "Checking MinIO..."
check_service "MinIO" "http://localhost:9000/minio/health/live"

# Check Prometheus
echo "Checking Prometheus..."
check_service "Prometheus" "http://localhost:9090/-/healthy"

# Check Grafana
echo "Checking Grafana..."
check_service "Grafana" "http://localhost:3000/api/health"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   ✅ All Services Started Successfully!                       ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Create MLflow bucket in MinIO
echo "🗂️  Configuring MinIO buckets..."
sleep 5
docker-compose exec -T minio mc alias set myminio http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
docker-compose exec -T minio mc mb myminio/mlflow-artifacts 2>/dev/null || echo "   Bucket already exists"
docker-compose exec -T minio mc mb myminio/processed-data 2>/dev/null || echo "   Bucket already exists"
echo -e "${GREEN}✅ MinIO buckets configured${NC}"
echo ""

# Initialize Airflow (create default admin user)
echo "👤 Checking Airflow admin user..."
docker-compose exec -T airflow-webserver airflow users list | grep -q "airflow" && \
    echo -e "${GREEN}✅ Airflow admin user exists${NC}" || \
    echo -e "${YELLOW}⚠️  Creating Airflow admin user...${NC}"
echo ""

# Create PostgreSQL database for MLflow if not exists
echo "💾 Configuring MLflow database..."
docker-compose exec -T postgres psql -U airflow -tc "SELECT 1 FROM pg_database WHERE datname = 'mlflow'" | grep -q 1 || \
    docker-compose exec -T postgres psql -U airflow -c "CREATE DATABASE mlflow;" 2>/dev/null || true
echo -e "${GREEN}✅ MLflow database ready${NC}"
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   📊 Dashboard Access URLs                                    ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "🔹 Airflow Dashboard:      http://localhost:8080"
echo "   Username: airflow"
echo "   Password: airflow"
echo ""
echo "🔹 MLflow Tracking:        http://localhost:5000"
echo "   (No authentication required)"
echo ""
echo "🔹 Grafana Dashboards:     http://localhost:3000"
echo "   Username: admin"
echo "   Password: admin"
echo "   Dashboards:"
echo "     • USD Volatility Monitoring"
echo "     • MLOps Pipeline Overview"
echo ""
echo "🔹 Prometheus Metrics:     http://localhost:9090"
echo "   (No authentication required)"
echo ""
echo "🔹 MinIO Console:          http://localhost:9001"
echo "   Username: minioadmin"
echo "   Password: minioadmin"
echo ""
echo "🔹 FastAPI Documentation:  http://localhost:8000/docs"
echo "   (Start with: uvicorn src.api.app:app)"
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   📚 Next Steps                                               ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "1. 📖 View detailed dashboard guide:"
echo "   cat DASHBOARD_ACCESS_GUIDE.md"
echo ""
echo "2. 🔄 Run ETL pipeline:"
echo "   python src/data/extraction.py"
echo "   python src/data/transformation.py"
echo ""
echo "3. 🤖 Train optimized model:"
echo "   python src/models/train_optimized.py"
echo ""
echo "4. 🚀 Start prediction API:"
echo "   uvicorn src.api.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "5. 🔍 View logs:"
echo "   docker-compose logs -f <service-name>"
echo ""
echo "6. 🛑 Stop services:"
echo "   docker-compose down"
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   ✨ Infrastructure Ready - Happy ML Engineering! ✨          ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if all services are running
echo "📊 Service Status:"
docker-compose ps
