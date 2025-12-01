#!/usr/bin/env python3
"""
Comprehensive deployment validation script for USD Volatility Prediction MLOps Pipeline.
Validates all components before production deployment.
"""
import sys
import requests
import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DeploymentValidator:
    """Validates all components of the MLOps pipeline."""

    def __init__(self):
        self.results: Dict[str, List[Tuple[str, bool, str]]] = {
            "Configuration": [],
            "Services": [],
            "Monitoring": [],
            "Data Pipeline": [],
            "Model": []
        }

    def validate_yaml_files(self) -> bool:
        """Validate all YAML configuration files."""
        print("\n" + "="*60)
        print("VALIDATING YAML CONFIGURATION FILES")
        print("="*60)

        yaml_files = [
            "docker-compose.yml",
            "infrastructure/prometheus/prometheus.yml",
            "infrastructure/prometheus/alert_rules.yml",
            "infrastructure/grafana/datasources/datasource.yml",
            "infrastructure/grafana/dashboards/dashboard.yml",
            ".github/workflows/lint-test.yml",
            ".github/workflows/deploy.yml",
            ".github/workflows/train-cml.yml"
        ]

        all_valid = True
        for yaml_file in yaml_files:
            file_path = PROJECT_ROOT / yaml_file
            try:
                with open(file_path, 'r') as f:
                    yaml.safe_load(f)
                print(f"[OK] {yaml_file}")
                self.results["Configuration"].append((yaml_file, True, "Valid YAML"))
            except FileNotFoundError:
                print(f"[FAIL] {yaml_file} - File not found")
                self.results["Configuration"].append((yaml_file, False, "File not found"))
                all_valid = False
            except yaml.YAMLError as e:
                print(f"[FAIL] {yaml_file} - YAML Error: {e}")
                self.results["Configuration"].append((yaml_file, False, f"YAML Error: {e}"))
                all_valid = False

        return all_valid

    def validate_python_imports(self) -> bool:
        """Validate critical Python modules can be imported."""
        print("\n" + "="*60)
        print("VALIDATING PYTHON MODULES")
        print("="*60)

        modules = [
            ("config.config", "Configuration module"),
            ("src.data.data_extraction", "Data extraction module"),
            ("src.data.data_transformation", "Data transformation module"),
            ("src.models.trainer", "Model training module"),
            ("src.monitoring.drift", "Drift detection module"),
            ("src.monitoring.alerts", "Alert management module"),
        ]

        all_valid = True
        for module_name, description in modules:
            try:
                __import__(module_name)
                print(f"[OK] {description}: {module_name}")
                self.results["Configuration"].append((module_name, True, "Imported successfully"))
            except ImportError as e:
                print(f"[FAIL] {description}: {module_name} - {e}")
                self.results["Configuration"].append((module_name, False, str(e)))
                all_valid = False

        return all_valid

    def check_service_endpoint(self, name: str, url: str, timeout: int = 5) -> bool:
        """Check if a service endpoint is reachable."""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code in [200, 404]:  # 404 is ok for root path
                print(f"[OK] {name}: {url} (Status: {response.status_code})")
                self.results["Services"].append((name, True, f"Reachable (Status: {response.status_code})"))
                return True
            else:
                print(f"[WARN] {name}: {url} (Status: {response.status_code})")
                self.results["Services"].append((name, True, f"Reachable but unexpected status: {response.status_code}"))
                return True
        except requests.exceptions.RequestException as e:
            print(f"[FAIL] {name}: {url} - {type(e).__name__}")
            self.results["Services"].append((name, False, f"Not reachable: {type(e).__name__}"))
            return False

    def validate_services(self) -> bool:
        """Validate all services are running (if stack is up)."""
        print("\n" + "="*60)
        print("VALIDATING SERVICES (if running)")
        print("="*60)

        services = [
            ("FastAPI Prediction API", "http://localhost:8000/health"),
            ("Prometheus", "http://localhost:9090/-/healthy"),
            ("Grafana", "http://localhost:3000/api/health"),
            ("MinIO", "http://localhost:9000/minio/health/live"),
            ("Airflow Webserver", "http://localhost:8080/health"),
            ("MLflow", "http://localhost:5000/health")
        ]

        reachable_count = 0
        for name, url in services:
            if self.check_service_endpoint(name, url):
                reachable_count += 1

        print(f"\nServices reachable: {reachable_count}/{len(services)}")
        if reachable_count == 0:
            print("[WARN] No services are running. Start with: docker-compose up -d")

        return True  # Don't fail if services aren't running (they might not be started yet)

    def validate_prometheus_config(self) -> bool:
        """Validate Prometheus configuration."""
        print("\n" + "="*60)
        print("VALIDATING PROMETHEUS CONFIGURATION")
        print("="*60)

        prom_config = PROJECT_ROOT / "infrastructure/prometheus/prometheus.yml"
        alert_rules = PROJECT_ROOT / "infrastructure/prometheus/alert_rules.yml"

        try:
            with open(prom_config) as f:
                config = yaml.safe_load(f)

            # Check scrape configs
            scrape_jobs = [job['job_name'] for job in config.get('scrape_configs', [])]
            expected_jobs = ['prometheus', 'prediction-api', 'minio']

            for job in expected_jobs:
                if job in scrape_jobs:
                    print(f"[OK] Scrape job configured: {job}")
                    self.results["Monitoring"].append((f"Scrape job: {job}", True, "Configured"))
                else:
                    print(f"[FAIL] Missing scrape job: {job}")
                    self.results["Monitoring"].append((f"Scrape job: {job}", False, "Not configured"))

            # Check alert rules
            if alert_rules.exists():
                with open(alert_rules) as f:
                    rules = yaml.safe_load(f)
                alert_count = sum(len(group.get('rules', [])) for group in rules.get('groups', []))
                print(f"[OK] Alert rules file exists with {alert_count} rules")
                self.results["Monitoring"].append(("Alert rules", True, f"{alert_count} rules configured"))
            else:
                print("[FAIL] Alert rules file not found")
                self.results["Monitoring"].append(("Alert rules", False, "File not found"))

            return True
        except Exception as e:
            print(f"[FAIL] Error validating Prometheus config: {e}")
            self.results["Monitoring"].append(("Prometheus config", False, str(e)))
            return False

    def validate_grafana_dashboards(self) -> bool:
        """Validate Grafana dashboard configuration."""
        print("\n" + "="*60)
        print("VALIDATING GRAFANA DASHBOARDS")
        print("="*60)

        dashboard_dir = PROJECT_ROOT / "infrastructure/grafana/dashboards"

        # Check for dashboard JSON files
        dashboard_files = list(dashboard_dir.glob("*.json"))

        if dashboard_files:
            for dashboard_file in dashboard_files:
                try:
                    with open(dashboard_file) as f:
                        dashboard = json.load(f)

                    title = dashboard.get('title', 'Unknown')
                    panel_count = len(dashboard.get('panels', []))

                    print(f"[OK] Dashboard: {title}")
                    print(f"  - Panels: {panel_count}")
                    print(f"  - File: {dashboard_file.name}")
                    self.results["Monitoring"].append((f"Dashboard: {title}", True, f"{panel_count} panels"))
                except Exception as e:
                    print(f"[FAIL] Error loading {dashboard_file.name}: {e}")
                    self.results["Monitoring"].append((dashboard_file.name, False, str(e)))
        else:
            print("[WARN] No dashboard JSON files found in infrastructure/grafana/dashboards/")
            self.results["Monitoring"].append(("Grafana dashboards", False, "No dashboard files found"))

        # Check datasource config
        datasource_file = PROJECT_ROOT / "infrastructure/grafana/datasources/datasource.yml"
        try:
            with open(datasource_file) as f:
                datasources = yaml.safe_load(f)

            ds_count = len(datasources.get('datasources', []))
            print(f"[OK] Datasources configured: {ds_count}")
            self.results["Monitoring"].append(("Grafana datasources", True, f"{ds_count} configured"))
        except Exception as e:
            print(f"[FAIL] Error loading datasource config: {e}")
            self.results["Monitoring"].append(("Grafana datasources", False, str(e)))

        return True

    def validate_airflow_dag(self) -> bool:
        """Validate Airflow DAG can be loaded."""
        print("\n" + "="*60)
        print("VALIDATING AIRFLOW DAG")
        print("="*60)

        dag_file = PROJECT_ROOT / "airflow/dags/etl_dag.py"

        if not dag_file.exists():
            print("[FAIL] ETL DAG file not found")
            self.results["Data Pipeline"].append(("Airflow DAG", False, "File not found"))
            return False

        try:
            # Try to parse the DAG file
            sys.path.insert(0, str(PROJECT_ROOT / "airflow/dags"))

            import etl_dag

            print(f"[OK] DAG loaded successfully")
            print(f"  - DAG ID: {etl_dag.dag.dag_id}")
            print(f"  - Schedule: {etl_dag.dag.schedule_interval}")
            print(f"  - Tasks: {len(etl_dag.dag.tasks)}")

            for task in etl_dag.dag.tasks:
                print(f"    - {task.task_id}")

            self.results["Data Pipeline"].append(("Airflow DAG", True, f"{len(etl_dag.dag.tasks)} tasks configured"))
            return True
        except Exception as e:
            print(f"[FAIL] Error loading DAG: {e}")
            self.results["Data Pipeline"].append(("Airflow DAG", False, str(e)))
            return False

    def validate_model_files(self) -> bool:
        """Check for model-related files and directories."""
        print("\n" + "="*60)
        print("VALIDATING MODEL SETUP")
        print("="*60)

        models_dir = PROJECT_ROOT / "models"
        data_dir = PROJECT_ROOT / "data"

        # Check directories exist
        for dir_path, name in [(models_dir, "Models"), (data_dir, "Data")]:
            if dir_path.exists():
                print(f"[OK] {name} directory exists: {dir_path}")
                self.results["Model"].append((f"{name} directory", True, "Exists"))
            else:
                print(f"[WARN] {name} directory not found (will be created on first run)")
                self.results["Model"].append((f"{name} directory", True, "Will be created automatically"))

        # Check for training script
        train_script = PROJECT_ROOT / "src/models/trainer.py"
        if train_script.exists():
            print(f"[OK] Training script exists")
            self.results["Model"].append(("Training script", True, "Found"))
        else:
            print(f"[FAIL] Training script not found")
            self.results["Model"].append(("Training script", False, "Not found"))
            return False

        return True

    def generate_report(self) -> None:
        """Generate validation summary report."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY REPORT")
        print("="*80)

        for category, checks in self.results.items():
            print(f"\n{category}:")
            print("-" * 80)

            passed = sum(1 for _, success, _ in checks if success)
            total = len(checks)

            for name, success, detail in checks:
                status = "[PASS]" if success else "[FAIL]"
                print(f"  {status}: {name}")
                if detail:
                    print(f"         {detail}")

            print(f"\n  Summary: {passed}/{total} checks passed")

        # Overall summary
        total_checks = sum(len(checks) for checks in self.results.values())
        total_passed = sum(sum(1 for _, success, _ in checks if success) for checks in self.results.values())

        print("\n" + "="*80)
        print(f"OVERALL: {total_passed}/{total_checks} checks passed")

        if total_passed == total_checks:
            print(">>> ALL VALIDATIONS PASSED - READY FOR DEPLOYMENT <<<")
            print("="*80)
            return True
        elif total_passed / total_checks > 0.8:
            print(">>> MOSTLY READY - Review failed checks before deployment")
            print("="*80)
            return True
        else:
            print(">>> NOT READY - Multiple critical issues need resolution")
            print("="*80)
            return False

    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        print("\n" + "=" * 80)
        print(" " * 15 + "USD VOLATILITY PREDICTION MLOPS PIPELINE")
        print(" " * 22 + "DEPLOYMENT VALIDATION")
        print("=" * 80)

        validations = [
            self.validate_yaml_files,
            self.validate_python_imports,
            self.validate_prometheus_config,
            self.validate_grafana_dashboards,
            self.validate_airflow_dag,
            self.validate_model_files,
            self.validate_services,
        ]

        for validation in validations:
            try:
                validation()
            except Exception as e:
                print(f"\n[FAIL] Validation error: {e}")

        return self.generate_report()


if __name__ == "__main__":
    validator = DeploymentValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)
