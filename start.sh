#!/bin/bash

# Simple startup script for MLOps Pipeline
# Starts all Docker services and provides access links

echo "🚀 Starting all MLOps services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to initialize..."
sleep 10

echo ""
echo "✅ Services started! Access links:"
echo ""

# GitHub Codespaces URLs
if [ -z "$CODESPACE_NAME" ]; then
    echo "⚠️  CODESPACE_NAME not set - using localhost URLs"
    CODESPACE_NAME="localhost"
fi

echo "⭐ RECOMMENDED: Access via VS Code Ports Tab"
echo "   1. Click 'PORTS' tab at bottom of VS Code"
echo "   2. Find port 8000, click globe icon"
echo "   3. Add '/dashboard' to the URL"
echo ""
echo "Or try these direct URLs:"
echo ""
echo "📊 Custom UI Dashboard (port 8000):"
echo "   https://${CODESPACE_NAME}-8000.app.github.dev/dashboard"
echo "   ⭐ Interactive UI with predictions, charts & monitoring"
echo ""
echo "📈 Grafana Dashboard (port 3000):"
echo "   https://${CODESPACE_NAME}-3000.githubpreview.dev"
echo "   Username: admin | Password: admin"
echo ""
echo "🎯 Airflow UI (port 8080):"
echo "   https://${CODESPACE_NAME}-8080.githubpreview.dev"
echo "   Username: airflow | Password: airflow"
echo ""
echo "🔍 MLflow Tracking (port 5000):"
echo "   https://${CODESPACE_NAME}-5000.githubpreview.dev"
echo ""
echo "🚀 FastAPI Docs (port 8000):"
echo "   https://${CODESPACE_NAME}-8000.githubpreview.dev/docs"
echo "   Health: https://${CODESPACE_NAME}-8000.githubpreview.dev/health"
echo ""
echo "📈 Prometheus Monitoring (port 9090):"
echo "   https://${CODESPACE_NAME}-9090.githubpreview.dev"
echo ""
echo "💾 MinIO Storage (port 9000/9001):"
echo "   Console: https://${CODESPACE_NAME}-9001.githubpreview.dev"
echo "   API: https://${CODESPACE_NAME}-9000.githubpreview.dev"
echo "   Username: minioadmin | Password: minioadmin"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 TIP: Open port 8000 from the 'Ports' tab to access the dashboard"
echo "Check service status: docker-compose ps"
echo "Detailed status: ./dashboard_status.sh"