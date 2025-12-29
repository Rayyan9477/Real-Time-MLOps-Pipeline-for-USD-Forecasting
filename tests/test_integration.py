#!/usr/bin/env python3
"""
Comprehensive integration test for all MLOps pipeline components.
Tests data flow, model prediction, API endpoints, and monitoring.
"""

import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class ComponentTester:
    """Tests all pipeline components."""
    
    def __init__(self):
        self.results = []
        self.api_base_url = "http://localhost:8000"
        
    def test(self, name: str, func):
        """Run a test and record result."""
        try:
            func()
            self.results.append((name, True, ""))
            print(f"{GREEN}✓{RESET} {name}")
            return True
        except Exception as e:
            self.results.append((name, False, str(e)))
            print(f"{RED}✗{RESET} {name}: {str(e)}")
            return False
    
    def test_configuration(self):
        """Test configuration loading."""
        from config.config import (
            TWELVE_DATA_CONFIG, MODEL_CONFIG, API_CONFIG,
            MODELS_DIR, DATA_DIR
        )
        assert MODELS_DIR.exists(), "Models directory not found"
        assert DATA_DIR.exists(), "Data directory not found"
        assert API_CONFIG["port"] == 8000, "API port mismatch"
    
    def test_model_artifacts(self):
        """Test model file existence and loading."""
        import joblib
        from config.config import MODELS_DIR
        
        model_path = MODELS_DIR / 'latest_model.pkl'
        scaler_path = MODELS_DIR / 'scaler.pkl'
        metadata_path = MODELS_DIR / 'latest_metadata.json'
        
        assert model_path.exists(), "Model file not found"
        assert scaler_path.exists(), "Scaler file not found"
        assert metadata_path.exists(), "Metadata file not found"
        
        # Load and verify
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert 'model_type' in metadata, "Invalid metadata"
        assert 'metrics' in metadata, "Metrics not in metadata"
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        import joblib
        from config.config import MODELS_DIR
        
        model = joblib.load(MODELS_DIR / 'latest_model.pkl')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        
        # Create dummy input (34 features)
        dummy_input = np.random.rand(1, 34)
        scaled_input = scaler.transform(dummy_input)
        prediction = model.predict(scaled_input)
        
        assert len(prediction) == 1, "Prediction shape mismatch"
        assert not np.isnan(prediction[0]), "Prediction is NaN"
        assert prediction[0] >= 0, "Negative prediction"
    
    def test_data_extraction_module(self):
        """Test data extraction module import."""
        from src.data.data_extraction import (
            TwelveDataClient,
            DataQualityChecker,
            extract_forex_data
        )
        assert callable(extract_forex_data), "extract_forex_data not callable"
    
    def test_data_transformation_module(self):
        """Test data transformation module."""
        from src.data.data_transformation import (
            FeatureEngineer,
            DataCleaner,
            DataTransformer,
            transform_data
        )
        
        # Test class instantiation
        transformer = DataTransformer()
        assert transformer is not None, "DataTransformer not instantiated"
    
    def test_model_trainer_module(self):
        """Test model training module."""
        from src.models.production_trainer import ProductionModelTrainer
        trainer = ProductionModelTrainer()
        assert trainer is not None, "ProductionModelTrainer not instantiated"
    
    def test_api_health_endpoint(self):
        """Test API health endpoint."""
        response = requests.get(f"{self.api_base_url}/health", timeout=5)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        data = response.json()
        assert data['status'] == 'healthy', "API not healthy"
        assert data['model_loaded'] == True, "Model not loaded"
    
    def test_api_metrics_endpoint(self):
        """Test API metrics endpoint."""
        # Try both /api/metrics and /model/info
        try:
            response = requests.get(f"{self.api_base_url}/model/info", timeout=5)
            assert response.status_code == 200, f"Model info endpoint failed: {response.status_code}"
            
            data = response.json()
            assert 'model_version' in data or 'model_type' in data, "Model info not in response"
        except:
            response = requests.get(f"{self.api_base_url}/api/stats", timeout=5)
            assert response.status_code == 200, f"API stats failed: {response.status_code}"
    
    def test_api_model_info_endpoint(self):
        """Test model info endpoint."""
        response = requests.get(f"{self.api_base_url}/model/info", timeout=5)
        assert response.status_code == 200, f"Model info failed: {response.status_code}"
        
        data = response.json()
        assert 'model_version' in data or 'model_type' in data, "Model info not complete"
    
    def test_prometheus_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = requests.get(f"{self.api_base_url}/metrics", timeout=5)
        assert response.status_code == 200, f"Prometheus metrics failed: {response.status_code}"
        assert 'predictions_total' in response.text or 'prediction' in response.text.lower(), "Metrics not found"
    
    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        response = requests.get(f"{self.api_base_url}/", timeout=5)
        assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
    
    def test_logger_functionality(self):
        """Test logging system."""
        from src.utils.logger import get_logger
        test_logger = get_logger("integration_test")
        test_logger.info("Test log message")
        assert test_logger is not None, "Logger not initialized"
    
    def test_github_workflows(self):
        """Test GitHub Actions workflows exist."""
        workflows_dir = Path('.github/workflows')
        assert workflows_dir.exists(), "Workflows directory not found"
        
        data_pipeline = workflows_dir / 'data-pipeline.yml'
        ci_cd = workflows_dir / 'ci-cd.yml'
        
        assert data_pipeline.exists(), "Data pipeline workflow not found"
        assert ci_cd.exists(), "CI/CD workflow not found"
        
        # Check for cron schedule
        content = data_pipeline.read_text()
        assert '0 */2 * * *' in content, "2-hour cron schedule not found"
    
    def test_deployment_configs(self):
        """Test deployment configuration files."""
        configs = ['railway.json', 'render.yaml', 'Dockerfile', 'docker-compose.yml']
        for config in configs:
            path = Path(config)
            assert path.exists(), f"{config} not found"
    
    def test_documentation(self):
        """Test documentation files."""
        docs = [
            'README.md',
            'docs/PRODUCTION_DEPLOYMENT.md',
            'docs/GITHUB_SECRETS_SETUP.md',
            'PRODUCTION_CHECKLIST.md'
        ]
        for doc in docs:
            path = Path(doc)
            assert path.exists(), f"{doc} not found"
    
    def run_all_tests(self):
        """Execute all integration tests."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}MLOps Pipeline - Comprehensive Integration Test{RESET}")
        print(f"{BLUE}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
        # Configuration & Artifacts
        print(f"{BLUE}📦 Configuration & Artifacts{RESET}")
        self.test("Configuration Loading", self.test_configuration)
        self.test("Model Artifacts", self.test_model_artifacts)
        self.test("Model Prediction", self.test_model_prediction)
        
        # Module Imports
        print(f"\n{BLUE}📚 Module Imports{RESET}")
        self.test("Data Extraction Module", self.test_data_extraction_module)
        self.test("Data Transformation Module", self.test_data_transformation_module)
        self.test("Model Trainer Module", self.test_model_trainer_module)
        self.test("Logger Functionality", self.test_logger_functionality)
        
        # API Endpoints (check if API is running)
        print(f"\n{BLUE}🌐 API Endpoints{RESET}")
        try:
            self.test("API Health Endpoint", self.test_api_health_endpoint)
            self.test("API Metrics Endpoint", self.test_api_metrics_endpoint)
            self.test("API Model Info", self.test_api_model_info_endpoint)
            self.test("Prometheus Metrics", self.test_prometheus_metrics_endpoint)
            self.test("API Root Endpoint", self.test_api_root_endpoint)
        except Exception as e:
            print(f"{YELLOW}⚠  API tests skipped (API not running): {e}{RESET}")
        
        # CI/CD & Deployment
        print(f"\n{BLUE}🔄 CI/CD & Deployment{RESET}")
        self.test("GitHub Workflows", self.test_github_workflows)
        self.test("Deployment Configs", self.test_deployment_configs)
        self.test("Documentation", self.test_documentation)
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}Test Summary{RESET}")
        print(f"{BLUE}{'='*70}{RESET}")
        
        total = len(self.results)
        passed = sum(1 for _, success, _ in self.results if success)
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        
        if failed > 0:
            print(f"\n{RED}Failed Tests:{RESET}")
            for name, success, error in self.results:
                if not success:
                    print(f"  {RED}✗{RESET} {name}")
                    if error:
                        print(f"    Error: {error}")
        
        print(f"\n{BLUE}{'='*70}{RESET}")
        
        if failed == 0:
            print(f"{GREEN}✅ ALL TESTS PASSED - SYSTEM OPERATIONAL{RESET}")
            return 0
        else:
            print(f"{YELLOW}⚠  {failed} test(s) failed - Review required{RESET}")
            return 1


def main():
    """Main entry point."""
    tester = ComponentTester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
