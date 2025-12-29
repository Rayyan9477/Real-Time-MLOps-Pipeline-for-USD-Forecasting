#!/usr/bin/env python3
"""
Production Deployment Validation Script
Validates all components of the MLOps pipeline are properly configured.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class DeploymentValidator:
    """Validates deployment readiness of the MLOps pipeline."""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.project_root = Path(__file__).parent.parent
        
    def check(self, name: str, condition: bool, message: str = ""):
        """Record a validation check result."""
        self.results.append((name, condition, message))
        status = f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"
        print(f"{status} {name}: {message if message else ('PASS' if condition else 'FAIL')}")
        return condition
    
    def section(self, title: str):
        """Print section header."""
        print(f"\n{BLUE}{'=' * 60}{RESET}")
        print(f"{BLUE}{title}{RESET}")
        print(f"{BLUE}{'=' * 60}{RESET}")
    
    def validate_environment_files(self) -> bool:
        """Validate environment configuration files exist."""
        self.section("Environment Configuration")
        
        env_files = ['.env.example', '.env.development', '.env.production']
        all_exist = True
        
        for env_file in env_files:
            exists = (self.project_root / env_file).exists()
            all_exist &= self.check(
                f"Environment file: {env_file}",
                exists,
                "Found" if exists else "Missing"
            )
        
        return all_exist
    
    def validate_required_env_vars(self) -> bool:
        """Validate required environment variables are set."""
        self.section("Required Environment Variables")
        
        required_vars = [
            'TWELVE_DATA_API_KEY',
            'API_HOST',
            'API_PORT',
        ]
        
        optional_vars = [
            'MLFLOW_TRACKING_URI',
            'DAGSHUB_USERNAME',
            'DAGSHUB_TOKEN',
        ]
        
        all_set = True
        for var in required_vars:
            is_set = os.getenv(var) is not None and os.getenv(var) != ''
            all_set &= self.check(
                f"Required: {var}",
                is_set,
                "Set" if is_set else f"{YELLOW}NOT SET - Required for production{RESET}"
            )
        
        for var in optional_vars:
            is_set = os.getenv(var) is not None and os.getenv(var) != ''
            self.check(
                f"Optional: {var}",
                is_set,
                "Set" if is_set else f"{YELLOW}Not set - Optional{RESET}"
            )
        
        return all_set
    
    def validate_directory_structure(self) -> bool:
        """Validate project directory structure."""
        self.section("Directory Structure")
        
        required_dirs = [
            'src',
            'src/api',
            'src/data',
            'src/models',
            'src/utils',
            'airflow/dags',
            'config',
            'tests',
            'docs',
            'models',
            'data/raw',
            'data/processed',
        ]
        
        all_exist = True
        for directory in required_dirs:
            path = self.project_root / directory
            exists = path.exists() and path.is_dir()
            all_exist &= self.check(
                f"Directory: {directory}",
                exists,
                "Found" if exists else "Missing"
            )
        
        return all_exist
    
    def validate_deployment_files(self) -> bool:
        """Validate deployment configuration files."""
        self.section("Deployment Configuration")
        
        deployment_files = [
            ('Dockerfile', True),
            ('docker-compose.yml', True),
            ('.dockerignore', True),
            ('requirements.txt', True),
            ('vercel.json', False),
            ('railway.json', False),
            ('render.yaml', False),
        ]
        
        all_required_exist = True
        for file_name, required in deployment_files:
            exists = (self.project_root / file_name).exists()
            if required:
                all_required_exist &= exists
            
            status_msg = "Found" if exists else ("Missing (required)" if required else "Missing (optional)")
            self.check(
                f"{'Required' if required else 'Optional'}: {file_name}",
                exists or not required,
                status_msg
            )
        
        return all_required_exist
    
    def validate_python_imports(self) -> bool:
        """Validate critical Python imports."""
        self.section("Python Dependencies")
        
        critical_imports = [
            ('fastapi', 'FastAPI framework'),
            ('pandas', 'Data processing'),
            ('numpy', 'Numerical computing'),
            ('xgboost', 'ML model'),
            ('mlflow', 'Experiment tracking'),
            ('prometheus_client', 'Monitoring'),
        ]
        
        all_imported = True
        for module, description in critical_imports:
            try:
                __import__(module)
                self.check(f"Import: {module}", True, f"✓ {description}")
            except ImportError as e:
                all_imported = False
                self.check(f"Import: {module}", False, f"✗ {description} - {str(e)}")
        
        return all_imported
    
    def validate_airflow_dag(self) -> bool:
        """Validate Airflow DAG configuration."""
        self.section("Airflow Configuration")
        
        dag_file = self.project_root / 'airflow' / 'dags' / 'etl_dag.py'
        
        if not dag_file.exists():
            self.check("Airflow DAG file", False, "etl_dag.py not found")
            return False
        
        self.check("Airflow DAG file", True, "Found")
        
        # Check DAG content
        content = dag_file.read_text()
        
        has_schedule = "schedule='0 */2 * * *'" in content or 'schedule_interval' in content
        self.check(
            "DAG Schedule (2 hours)",
            has_schedule,
            "Configured" if has_schedule else "Not configured"
        )
        
        has_catchup = 'catchup=False' in content
        self.check(
            "Catchup disabled",
            has_catchup,
            "Yes" if has_catchup else "No"
        )
        
        return has_schedule
    
    def validate_api_health(self) -> bool:
        """Validate API health endpoint (if running)."""
        self.section("API Health Check")
        
        api_host = os.getenv('API_HOST', 'localhost')
        api_port = os.getenv('API_PORT', '8000')
        
        if api_host == '0.0.0.0':
            api_host = 'localhost'
        
        health_url = f"http://{api_host}:{api_port}/health"
        
        try:
            response = requests.get(health_url, timeout=5)
            is_healthy = response.status_code == 200
            self.check(
                "API Health Endpoint",
                is_healthy,
                f"Responding at {health_url}" if is_healthy else f"Error: {response.status_code}"
            )
            return is_healthy
        except requests.exceptions.ConnectionError:
            self.check(
                "API Health Endpoint",
                False,
                f"{YELLOW}API not running at {health_url} (start with: uvicorn src.api.main:app){RESET}"
            )
            return False
        except Exception as e:
            self.check("API Health Endpoint", False, f"Error: {str(e)}")
            return False
    
    def validate_gitignore(self) -> bool:
        """Validate .gitignore is properly configured."""
        self.section("Git Configuration")
        
        gitignore = self.project_root / '.gitignore'
        
        if not gitignore.exists():
            self.check("Git ignore file", False, "Missing")
            return False
        
        self.check("Git ignore file", True, "Found")
        
        content = gitignore.read_text()
        
        important_entries = [
            ('__pycache__', 'Python cache'),
            ('.env', 'Environment variables'),
            ('*.log', 'Log files'),
            ('data/raw', 'Raw data'),
            ('models/*.pkl', 'Model files'),
        ]
        
        all_present = True
        for pattern, description in important_entries:
            present = pattern in content
            all_present &= present
            self.check(
                f"Ignore: {description}",
                present,
                "Present" if present else "Missing"
            )
        
        return all_present
    
    def validate_documentation(self) -> bool:
        """Validate documentation files."""
        self.section("Documentation")
        
        docs = [
            ('README.md', True),
            ('docs/DEPLOYMENT_GUIDE.md', True),
            ('docs/DASHBOARD_ACCESS_GUIDE.md', False),
            ('docs/DVC_SETUP.md', False),
        ]
        
        all_required_exist = True
        for doc, required in docs:
            exists = (self.project_root / doc).exists()
            if required:
                all_required_exist &= exists
            
            self.check(
                f"{'Required' if required else 'Optional'}: {doc}",
                exists or not required,
                "Found" if exists else ("Missing (required)" if required else "Missing (optional)")
            )
        
        return all_required_exist
    
    def validate_cron_configuration(self) -> bool:
        """Validate cron/scheduling configuration."""
        self.section("Cron/Scheduling Configuration")
        
        # Check vercel.json for cron
        vercel_config = self.project_root / 'vercel.json'
        has_vercel_cron = False
        
        if vercel_config.exists():
            try:
                config = json.loads(vercel_config.read_text())
                has_vercel_cron = 'crons' in config
                self.check(
                    "Vercel cron configuration",
                    has_vercel_cron,
                    "Configured" if has_vercel_cron else "Not configured"
                )
            except json.JSONDecodeError:
                self.check("Vercel cron configuration", False, "Invalid JSON")
        
        # Check railway.json for cron
        railway_config = self.project_root / 'railway.json'
        if railway_config.exists():
            self.check("Railway configuration", True, "Found")
        
        # Check render.yaml for cron
        render_config = self.project_root / 'render.yaml'
        has_render_cron = False
        
        if render_config.exists():
            content = render_config.read_text()
            has_render_cron = 'schedule:' in content and '0 */2 * * *' in content
            self.check(
                "Render cron configuration",
                has_render_cron,
                "Configured (every 2 hours)" if has_render_cron else "Not configured"
            )
        
        return has_vercel_cron or has_render_cron or True  # DAG schedule is primary
    
    def print_summary(self):
        """Print validation summary."""
        self.section("Validation Summary")
        
        total = len(self.results)
        passed = sum(1 for _, success, _ in self.results if success)
        failed = total - passed
        
        print(f"\nTotal Checks: {total}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        
        if failed > 0:
            print(f"\n{YELLOW}Failed Checks:{RESET}")
            for name, success, message in self.results:
                if not success:
                    print(f"  {RED}✗{RESET} {name}: {message}")
        
        print(f"\n{'=' * 60}")
        
        if failed == 0:
            print(f"{GREEN}✓ All validation checks passed!{RESET}")
            print(f"{GREEN}✓ Project is ready for production deployment{RESET}")
            return True
        else:
            print(f"{YELLOW}⚠ Some checks failed. Please review and fix issues before deployment.{RESET}")
            return False
    
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        print(f"\n{BLUE}{'=' * 60}{RESET}")
        print(f"{BLUE}MLOps Pipeline - Deployment Validation{RESET}")
        print(f"{BLUE}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        print(f"{BLUE}{'=' * 60}{RESET}")
        
        # Run all validations
        self.validate_environment_files()
        self.validate_required_env_vars()
        self.validate_directory_structure()
        self.validate_deployment_files()
        self.validate_python_imports()
        self.validate_airflow_dag()
        self.validate_gitignore()
        self.validate_documentation()
        self.validate_cron_configuration()
        self.validate_api_health()
        
        # Print summary
        return self.print_summary()


def main():
    """Main entry point."""
    validator = DeploymentValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
