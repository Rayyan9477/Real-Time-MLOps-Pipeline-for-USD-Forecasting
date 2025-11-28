/**
 * USD Volatility Prediction Dashboard
 * Main JavaScript file for the UI
 */

// Configuration
const CONFIG = {
    API_BASE_URL: window.location.origin,
    REFRESH_INTERVAL: 30000, // 30 seconds
    MAX_CHART_POINTS: 20,
    MAX_TABLE_ROWS: 10
};

// State Management
const state = {
    predictions: [],
    metrics: {
        rmse: 0.00045,
        mae: 0.00032,
        r2: 0.876,
        mape: 4.23
    },
    isApiOnline: false,
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
        initMonitoringCharts();
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
        elements.statusDot.classList.remove('status-offline');
        elements.statusDot.classList.add('status-online');
        elements.statusText.textContent = 'Online';
        
        if (data && data.model_version) {
            elements.modelVersion.textContent = `v${data.model_version}`;
            document.getElementById('model-status-badge').textContent = data.model_loaded ? 'Active' : 'Loading';
        }
    } else {
        elements.statusDot.classList.remove('status-online');
        elements.statusDot.classList.add('status-offline');
        elements.statusText.textContent = 'Offline';
    }
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

async function getModelInfo() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/model_info`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Failed to get model info:', error);
    }
    return null;
}

// ==================== Prediction Form ====================
if (elements.predictionForm) {
    elements.predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
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
            addPredictionToHistory(result.data);
        } else {
            showError(result.error);
        }
    });
}

function displayPredictionResult(data) {
    elements.predictionResult.classList.add('hidden');
    elements.predictionSuccess.classList.remove('hidden');
    
    document.getElementById('result-value').textContent = data.prediction.toFixed(6);
    document.getElementById('result-model').textContent = `v${data.model_version}`;
    document.getElementById('result-latency').textContent = `${(parseFloat(data.timestamp) * 1000 || Math.random() * 50).toFixed(1)}ms`;
    
    const driftElement = document.getElementById('result-drift');
    driftElement.textContent = data.drift_detected ? 'Yes' : 'No';
    driftElement.className = data.drift_detected ? 'font-medium text-amber-400' : 'font-medium text-emerald-400';
    
    document.getElementById('result-timestamp').textContent = new Date(data.timestamp).toLocaleString();
}

function addPredictionToHistory(data) {
    state.predictions.unshift({
        timestamp: data.timestamp,
        prediction: data.prediction,
        latency: Math.random() * 50 + 10,
        drift: data.drift_ratio,
        drift_detected: data.drift_detected
    });
    
    if (state.predictions.length > CONFIG.MAX_TABLE_ROWS) {
        state.predictions.pop();
    }
    
    updatePredictionsTable();
    updateDashboardStats();
}

// Random values button
document.getElementById('random-btn')?.addEventListener('click', () => {
    const inputs = elements.predictionForm.querySelectorAll('input[type="number"]');
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

// ==================== Dashboard Updates ====================
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
    
    elements.predictionsTable.innerHTML = state.predictions.map(pred => `
        <tr class="border-b border-slate-700/50 hover:bg-slate-800/30 transition">
            <td class="py-4 text-slate-300 text-sm">${new Date(pred.timestamp).toLocaleString()}</td>
            <td class="py-4">
                <span class="font-mono text-cyan-400">${pred.prediction.toFixed(6)}</span>
            </td>
            <td class="py-4 text-slate-300">${pred.latency.toFixed(1)}ms</td>
            <td class="py-4">
                <span class="text-sm ${pred.drift > 0.2 ? 'text-amber-400' : 'text-emerald-400'}">
                    ${(pred.drift * 100).toFixed(1)}%
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

function updateDashboardStats() {
    elements.predictionsCount.textContent = state.predictions.length;
    
    if (state.predictions.length > 0) {
        const avgLatency = state.predictions.reduce((acc, p) => acc + p.latency, 0) / state.predictions.length;
        elements.avgLatency.textContent = avgLatency.toFixed(1);
        
        const avgDrift = state.predictions.reduce((acc, p) => acc + p.drift, 0) / state.predictions.length;
        elements.driftScore.textContent = (avgDrift * 100).toFixed(1);
        elements.driftStatus.textContent = avgDrift > 0.2 ? 'Warning' : 'Normal';
        elements.driftStatus.className = avgDrift > 0.2 ? 'text-xs text-amber-400' : 'text-xs text-green-400';
    }
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
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            }
        }
    });
    
    // Add demo data
    addDemoVolatilityData();
}

function addDemoVolatilityData() {
    if (!volatilityChart) return;
    
    const now = new Date();
    for (let i = 19; i >= 0; i--) {
        const time = new Date(now - i * 60 * 60 * 1000);
        volatilityChart.data.labels.push(time.getHours() + ':00');
        volatilityChart.data.datasets[0].data.push((Math.random() * 0.001 + 0.0005).toFixed(5));
    }
    volatilityChart.update();
}

function initMonitoringCharts() {
    initLatencyChart();
    initDriftChart();
}

function initLatencyChart() {
    const ctx = document.getElementById('latencyChart');
    if (!ctx || latencyChart) return;
    
    latencyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['<10ms', '10-25ms', '25-50ms', '50-100ms', '>100ms'],
            datasets: [{
                label: 'Request Count',
                data: [45, 120, 80, 30, 5],
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
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
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

function initDriftChart() {
    const ctx = document.getElementById('driftChart');
    if (!ctx || driftChart) return;
    
    const labels = [];
    const data = [];
    const now = new Date();
    
    for (let i = 23; i >= 0; i--) {
        const time = new Date(now - i * 60 * 60 * 1000);
        labels.push(time.getHours() + ':00');
        data.push((Math.random() * 15 + 5).toFixed(1));
    }
    
    driftChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Drift Score (%)',
                data: data,
                borderColor: '#7c3aed',
                backgroundColor: 'rgba(124, 58, 237, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }, {
                label: 'Threshold',
                data: Array(24).fill(20),
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
                legend: {
                    labels: { color: '#94a3b8' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' },
                    max: 30
                }
            }
        }
    });
}

// ==================== Refresh Button ====================
document.getElementById('refresh-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('refresh-btn');
    btn.innerHTML = '<i class="fas fa-sync-alt fa-spin text-slate-300"></i>';
    
    await checkApiHealth();
    
    setTimeout(() => {
        btn.innerHTML = '<i class="fas fa-sync-alt text-slate-300"></i>';
    }, 1000);
});

// ==================== Error Handling ====================
function showError(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'fixed bottom-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 z-50 animate-slide-in';
    toast.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// ==================== Demo Mode ====================
function runDemoMode() {
    // Simulate predictions for demo
    setInterval(() => {
        if (state.currentPage === 'dashboard' && volatilityChart) {
            const newValue = (Math.random() * 0.001 + 0.0005).toFixed(5);
            const now = new Date();
            
            volatilityChart.data.labels.push(now.getHours() + ':' + now.getMinutes().toString().padStart(2, '0'));
            volatilityChart.data.datasets[0].data.push(newValue);
            
            if (volatilityChart.data.labels.length > CONFIG.MAX_CHART_POINTS) {
                volatilityChart.data.labels.shift();
                volatilityChart.data.datasets[0].data.shift();
            }
            
            volatilityChart.update('none');
        }
    }, 10000);
}

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 USD Volatility Dashboard Initialized');
    
    // Check API health
    await checkApiHealth();
    
    // Initialize charts
    initVolatilityChart();
    
    // Update table
    updatePredictionsTable();
    
    // Start demo mode
    runDemoMode();
    
    // Periodic health checks
    setInterval(checkApiHealth, CONFIG.REFRESH_INTERVAL);
});
