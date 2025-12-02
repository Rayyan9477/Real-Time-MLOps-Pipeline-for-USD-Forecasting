/**
 * USD Volatility Prediction Dashboard
 * Main JavaScript file for the UI - Now with Real API Integration
 */

// Configuration
const CONFIG = {
    API_BASE_URL: window.location.origin,
    REFRESH_INTERVAL: 30000, // 30 seconds
    MAX_CHART_POINTS: 24,
    MAX_TABLE_ROWS: 10
};

// Get GitHub Codespaces URLs
function getCodespacesUrl(port) {
    const hostname = window.location.hostname;
    if (hostname.includes('github.dev')) {
        // Extract codespace name (everything before .github.dev)
        const codespaceName = hostname.split('.')[0];
        return `https://${codespaceName}-${port}.app.github.dev`;
    }
    // Fallback to localhost for local development
    return `http://localhost:${port}`;
}

// Service URLs
const SERVICE_URLS = {
    grafana: getCodespacesUrl(3000),
    prometheus: getCodespacesUrl(9090),
    mlflow: getCodespacesUrl(5000),
    airflow: getCodespacesUrl(8080),
    minio: getCodespacesUrl(9001)
};

// State Management
const state = {
    predictions: [],
    modelInfo: null,
    stats: null,
    isApiOnline: false,
    modelLoaded: false,
    currentPage: 'dashboard'
};

// DOM Elements
const elements = {
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    modelVersion: document.getElementById('model-version'),
    predictionsCount: document.getElementById('predictions-count'),
    avgLatency: document.getElementById('avg-latency'),
    driftScore: document.getElementById('drift-score'),
    driftStatus: document.getElementById('drift-status'),
    predictionsTable: document.getElementById('predictions-table'),
    predictionForm: document.getElementById('prediction-form'),
    predictionResult: document.getElementById('prediction-result'),
    predictionSuccess: document.getElementById('prediction-success')
};

// Charts
let volatilityChart = null;
let latencyChart = null;
let driftChart = null;

// ==================== Navigation ====================
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const page = e.currentTarget.dataset.page;
        navigateToPage(page);
    });
});

function navigateToPage(page) {
    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active', 'text-cyan-400', 'bg-cyan-400/10');
        link.classList.add('text-slate-300');
    });
    
    const activeLink = document.querySelector(`[data-page="${page}"]`);
    if (activeLink) {
        activeLink.classList.add('active', 'text-cyan-400', 'bg-cyan-400/10');
        activeLink.classList.remove('text-slate-300');
    }
    
    // Show/hide pages
    document.querySelectorAll('.page-content').forEach(p => {
        p.classList.add('hidden');
    });
    
    const targetPage = document.getElementById(`page-${page}`);
    if (targetPage) {
        targetPage.classList.remove('hidden');
    }
    
    state.currentPage = page;
    
    // Initialize page-specific content
    if (page === 'monitoring') {
        loadMonitoringData();
    }
}

// ==================== API Functions ====================
async function checkApiHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });
        
        if (response.ok) {
            const data = await response.json();
            updateApiStatus(true, data);
            return data;
        } else {
            updateApiStatus(false);
            return null;
        }
    } catch (error) {
        console.error('API health check failed:', error);
        updateApiStatus(false);
        return null;
    }
}

function updateApiStatus(isOnline, data = null) {
    state.isApiOnline = isOnline;
    
    if (isOnline) {
        elements.statusDot?.classList.remove('status-offline');
        elements.statusDot?.classList.add('status-online');
        if (elements.statusText) elements.statusText.textContent = 'Online';
        
        if (data) {
            state.modelLoaded = data.model_loaded;
            if (data.model_version && elements.modelVersion) {
                elements.modelVersion.textContent = `v${data.model_version}`;
            }
            const badge = document.getElementById('model-status-badge');
            if (badge) {
                badge.textContent = data.model_loaded ? 'Active' : 'Loading';
            }
        }
    } else {
        elements.statusDot?.classList.remove('status-online');
        elements.statusDot?.classList.add('status-offline');
        if (elements.statusText) elements.statusText.textContent = 'Offline';
    }
}

async function fetchStats() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/stats`);
        if (response.ok) {
            state.stats = await response.json();
            updateDashboardFromStats();
            return state.stats;
        }
    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
    return null;
}

async function fetchModelInfo() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/model/info`);
        if (response.ok) {
            state.modelInfo = await response.json();
            updateModelMetrics();
            return state.modelInfo;
        }
    } catch (error) {
        console.error('Failed to fetch model info:', error);
    }
    return null;
}

async function fetchRecentPredictions() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/predictions/recent?limit=20`);
        if (response.ok) {
            const data = await response.json();
            state.predictions = data.predictions || [];
            updatePredictionsTable();
            updateVolatilityChart();
            return data;
        }
    } catch (error) {
        console.error('Failed to fetch predictions:', error);
    }
    return null;
}

async function fetchLatencyDistribution() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/latency/distribution`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Failed to fetch latency distribution:', error);
    }
    return null;
}

async function fetchDriftHistory() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/drift/history`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Failed to fetch drift history:', error);
    }
    return null;
}

async function makePrediction(features) {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ features })
        });
        
        if (response.ok) {
            const data = await response.json();
            // Refresh predictions after successful prediction
            await fetchRecentPredictions();
            await fetchStats();
            return { success: true, data };
        } else {
            const error = await response.json();
            return { success: false, error: error.detail || 'Prediction failed' };
        }
    } catch (error) {
        console.error('Prediction error:', error);
        return { success: false, error: error.message };
    }
}

// ==================== Update Dashboard ====================
function updateDashboardFromStats() {
    if (!state.stats) return;
    
    if (elements.predictionsCount) {
        elements.predictionsCount.textContent = state.stats.total_predictions || 0;
    }
    
    if (elements.avgLatency) {
        elements.avgLatency.textContent = (state.stats.avg_latency_ms || 0).toFixed(1);
    }
    
    if (elements.driftScore) {
        elements.driftScore.textContent = state.stats.drift_alerts || 0;
    }
    
    // Update model accuracy card if exists
    const accuracyEl = document.getElementById('model-accuracy');
    if (accuracyEl && state.stats.model_accuracy) {
        accuracyEl.textContent = state.stats.model_accuracy.toFixed(1);
    }
}

function updateModelMetrics() {
    if (!state.modelInfo || !state.modelInfo.metrics) return;
    
    const metrics = state.modelInfo.metrics;
    
    // Update metrics displays
    const rmseEl = document.getElementById('metric-rmse');
    const maeEl = document.getElementById('metric-mae');
    const r2El = document.getElementById('metric-r2');
    const mapeEl = document.getElementById('metric-mape');
    
    if (rmseEl) rmseEl.textContent = (metrics.rmse || 0).toFixed(5);
    if (maeEl) maeEl.textContent = (metrics.mae || 0).toFixed(5);
    if (r2El) r2El.textContent = (metrics.r2 || 0).toFixed(3);
    if (mapeEl) mapeEl.textContent = (metrics.mape || 0).toFixed(2) + '%';
    
    // Update progress bars if they exist
    updateProgressBar('rmse-bar', Math.min(metrics.rmse * 10000, 100));
    updateProgressBar('mae-bar', Math.min(metrics.mae * 10000, 100));
    updateProgressBar('r2-bar', Math.max(0, metrics.r2) * 100);
    updateProgressBar('mape-bar', Math.min(metrics.mape * 10, 100));
}

function updateProgressBar(id, percentage) {
    const bar = document.getElementById(id);
    if (bar) {
        bar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
    }
}

// ==================== Prediction Form ====================
if (elements.predictionForm) {
    elements.predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!state.modelLoaded) {
            showError('Model not loaded. Please wait for model to load.');
            return;
        }
        
        const formData = new FormData(e.target);
        const features = {};
        
        formData.forEach((value, key) => {
            features[key] = parseFloat(value);
        });
        
        // Show loading state
        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Predicting...';
        submitBtn.disabled = true;
        
        const result = await makePrediction(features);
        
        // Restore button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        
        if (result.success) {
            displayPredictionResult(result.data);
        } else {
            showError(result.error);
        }
    });
}

function displayPredictionResult(data) {
    if (elements.predictionResult) elements.predictionResult.classList.add('hidden');
    if (elements.predictionSuccess) elements.predictionSuccess.classList.remove('hidden');
    
    const resultValue = document.getElementById('result-value');
    const resultModel = document.getElementById('result-model');
    const resultLatency = document.getElementById('result-latency');
    const resultDrift = document.getElementById('result-drift');
    const resultTimestamp = document.getElementById('result-timestamp');
    
    if (resultValue) resultValue.textContent = data.prediction.toFixed(6);
    if (resultModel) resultModel.textContent = `v${data.model_version}`;
    if (resultLatency) resultLatency.textContent = `${(data.latency_ms || 0).toFixed(1)}ms`;
    
    if (resultDrift) {
        resultDrift.textContent = data.drift_detected ? 'Yes' : 'No';
        resultDrift.className = data.drift_detected ? 'font-medium text-amber-400' : 'font-medium text-emerald-400';
    }
    
    if (resultTimestamp) resultTimestamp.textContent = new Date(data.timestamp).toLocaleString();
    
    // Update interpretation display
    const interpretationEl = document.getElementById('result-interpretation');
    const riskLevelEl = document.getElementById('result-risk-level');
    const confidenceEl = document.getElementById('result-confidence');
    
    if (interpretationEl) interpretationEl.textContent = data.prediction_interpretation || 'No interpretation available';
    if (riskLevelEl) {
        riskLevelEl.textContent = data.risk_level || 'Unknown';
        // Color code risk level
        riskLevelEl.className = 'font-medium px-2 py-1 rounded text-sm ' + 
            (data.risk_level === 'Low' ? 'bg-green-100 text-green-800' :
             data.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
             data.risk_level === 'High' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800');
    }
    if (confidenceEl) confidenceEl.textContent = data.confidence_score || 'Unknown';
}

// Random values button
document.getElementById('random-btn')?.addEventListener('click', () => {
    const inputs = document.querySelectorAll('#prediction-form input[type="number"]');
    inputs.forEach(input => {
        const name = input.name;
        let value;
        
        switch(name) {
            case 'close_lag_1':
            case 'close_rolling_mean_24':
                value = (1.08 + Math.random() * 0.02).toFixed(4);
                break;
            case 'close_rolling_std_24':
                value = (0.001 + Math.random() * 0.002).toFixed(4);
                break;
            case 'log_return':
                value = ((Math.random() - 0.5) * 0.002).toFixed(5);
                break;
            case 'hour_sin':
            case 'hour_cos':
                value = ((Math.random() - 0.5) * 2).toFixed(3);
                break;
            case 'price_range':
                value = (0.0005 + Math.random() * 0.002).toFixed(4);
                break;
            case 'price_change_pct':
                value = ((Math.random() - 0.5) * 0.001).toFixed(5);
                break;
            default:
                value = Math.random().toFixed(4);
        }
        
        input.value = value;
    });
});

// ==================== Tables ====================
function updatePredictionsTable() {
    if (!elements.predictionsTable) return;
    
    if (state.predictions.length === 0) {
        elements.predictionsTable.innerHTML = `
            <tr class="border-b border-slate-700/50">
                <td class="py-4 text-slate-300" colspan="5">
                    <div class="flex items-center justify-center text-slate-500">
                        <i class="fas fa-inbox mr-2"></i> No predictions yet
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    elements.predictionsTable.innerHTML = state.predictions.slice(0, CONFIG.MAX_TABLE_ROWS).map(pred => `
        <tr class="border-b border-slate-700/50 hover:bg-slate-800/30 transition">
            <td class="py-4 text-slate-300 text-sm">${new Date(pred.timestamp).toLocaleString()}</td>
            <td class="py-4">
                <span class="font-mono text-cyan-400">${pred.prediction.toFixed(6)}</span>
            </td>
            <td class="py-4 text-slate-300">${(pred.latency_ms || 0).toFixed(1)}ms</td>
            <td class="py-4">
                <span class="text-sm ${pred.drift_ratio > 0.2 ? 'text-amber-400' : 'text-emerald-400'}">
                    ${((pred.drift_ratio || 0) * 100).toFixed(1)}%
                </span>
            </td>
            <td class="py-4">
                <span class="px-2 py-1 rounded-full text-xs ${pred.drift_detected ? 'bg-amber-400/10 text-amber-400' : 'bg-emerald-400/10 text-emerald-400'}">
                    ${pred.drift_detected ? 'Drift' : 'Normal'}
                </span>
            </td>
        </tr>
    `).join('');
}

// ==================== Charts ====================
function initVolatilityChart() {
    const ctx = document.getElementById('volatilityChart');
    if (!ctx) return;
    
    volatilityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Predicted Volatility',
                data: [],
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: '#06b6d4'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function updateVolatilityChart() {
    if (!volatilityChart || state.predictions.length === 0) return;
    
    // Use real prediction data
    const recentPreds = state.predictions.slice(0, CONFIG.MAX_CHART_POINTS).reverse();
    
    volatilityChart.data.labels = recentPreds.map(p => {
        const date = new Date(p.timestamp);
        return date.getHours() + ':' + date.getMinutes().toString().padStart(2, '0');
    });
    
    volatilityChart.data.datasets[0].data = recentPreds.map(p => p.prediction);
    volatilityChart.update();
}

async function loadMonitoringData() {
    // Load latency distribution
    const latencyData = await fetchLatencyDistribution();
    if (latencyData && latencyData.buckets.length > 0) {
        updateLatencyChart(latencyData);
    } else {
        initLatencyChartWithDefaults();
    }
    
    // Load drift history
    const driftData = await fetchDriftHistory();
    if (driftData && driftData.timestamps.length > 0) {
        updateDriftChart(driftData);
    } else {
        initDriftChartWithDefaults();
    }
}

function updateLatencyChart(data) {
    const ctx = document.getElementById('latencyChart');
    if (!ctx) return;
    
    if (latencyChart) {
        latencyChart.data.labels = data.buckets;
        latencyChart.data.datasets[0].data = data.counts;
        latencyChart.update();
    } else {
        latencyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.buckets,
                datasets: [{
                    label: 'Request Count',
                    data: data.counts,
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(6, 182, 212, 0.8)',
                        'rgba(124, 58, 237, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(239, 68, 68, 0.8)'
                    ],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    }
}

function initLatencyChartWithDefaults() {
    updateLatencyChart({
        buckets: ['<10ms', '10-25ms', '25-50ms', '50-100ms', '>100ms'],
        counts: [0, 0, 0, 0, 0]
    });
}

function updateDriftChart(data) {
    const ctx = document.getElementById('driftChart');
    if (!ctx) return;
    
    if (driftChart) {
        driftChart.data.labels = data.timestamps.map(t => {
            const date = new Date(t);
            return date.getHours() + ':00';
        });
        driftChart.data.datasets[0].data = data.scores;
        driftChart.update();
    } else {
        driftChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.timestamps.map(t => {
                    const date = new Date(t);
                    return date.getHours() + ':00';
                }),
                datasets: [{
                    label: 'Drift Score (%)',
                    data: data.scores,
                    borderColor: '#7c3aed',
                    backgroundColor: 'rgba(124, 58, 237, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Threshold',
                    data: Array(data.timestamps.length).fill(data.threshold || 20),
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8' } }
                },
                scales: {
                    x: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' }, max: 30 }
                }
            }
        });
    }
}

function initDriftChartWithDefaults() {
    const now = new Date();
    const timestamps = [];
    for (let i = 23; i >= 0; i--) {
        timestamps.push(new Date(now - i * 60 * 60 * 1000).toISOString());
    }
    updateDriftChart({
        timestamps: timestamps,
        scores: Array(24).fill(0),
        threshold: 20
    });
}

// ==================== Refresh Button ====================
document.getElementById('refresh-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('refresh-btn');
    btn.innerHTML = '<i class="fas fa-sync-alt fa-spin text-slate-300"></i>';
    
    await refreshAllData();
    
    setTimeout(() => {
        btn.innerHTML = '<i class="fas fa-sync-alt text-slate-300"></i>';
    }, 1000);
});

async function refreshAllData() {
    await Promise.all([
        checkApiHealth(),
        fetchStats(),
        fetchModelInfo(),
        fetchRecentPredictions()
    ]);
}

// ==================== Error Handling ====================
function showError(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'fixed bottom-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 z-50';
    toast.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'fixed bottom-4 right-4 bg-emerald-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 z-50';
    toast.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 USD Volatility Dashboard Initialized');
    
    // Initialize charts first
    initVolatilityChart();
    
    // Load all data
    await refreshAllData();
    
    // Update model status indicator
    if (state.modelLoaded) {
        showSuccess('Model loaded successfully!');
    }
    
    // Periodic refresh
    setInterval(refreshAllData, CONFIG.REFRESH_INTERVAL);
});
