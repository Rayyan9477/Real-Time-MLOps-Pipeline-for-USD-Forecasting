"""
ETL DAG for USD Volatility Prediction Pipeline.
Runs every 2 hours to extract, transform, load, and version forex data.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))

try:
    from src.data.data_extraction import extract_forex_data, DataQualityChecker
    from src.data.data_transformation import transform_data, generate_data_profile
    from src.utils.storage import MinIOClient
    from src.utils.logger import get_logger
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger = None
    IMPORTS_SUCCESSFUL = False
    print(f"Warning: Some imports failed: {e}. DAG will not function properly.")

logger = get_logger("etl_dag") if IMPORTS_SUCCESSFUL else None

# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 29),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(minutes=30),
}


def extract_data_task(**context):
    """
    Task 1: Extract data from Twelve Data API with quality checks.
    Raises AirflowException if quality checks fail.
    """
    if not IMPORTS_SUCCESSFUL:
        raise AirflowException("Required modules not available. Please ensure all dependencies are installed.")
    
    logger.info("=" * 60)
    logger.info("TASK 1: DATA EXTRACTION")
    logger.info("=" * 60)
    
    try:
        # Extract data (includes quality checks)
        df = extract_forex_data(save_raw=True)
        
        # Push dataset info to XCom
        context['ti'].xcom_push(key='raw_data_shape', value=df.shape)
        context['ti'].xcom_push(key='extraction_timestamp', value=datetime.now().isoformat())
        
        logger.info(f"✓ Data extraction successful: {df.shape[0]} rows, {df.shape[1]} columns")
        return "extraction_success"
        
    except Exception as e:
        logger.error(f"✗ Data extraction FAILED: {str(e)}")
        raise AirflowException(f"Data extraction failed: {str(e)}")


def transform_data_task(**context):
    """
    Task 2: Transform data with feature engineering.
    """
    if not IMPORTS_SUCCESSFUL:
        raise AirflowException("Required modules not available. Please ensure all dependencies are installed.")
    
    logger.info("=" * 60)
    logger.info("TASK 2: DATA TRANSFORMATION")
    logger.info("=" * 60)
    
    try:
        # Get the latest raw data
        from config.config import RAW_DATA_DIR
        import pandas as pd
        
        # Find most recent raw data file
        raw_files = sorted(RAW_DATA_DIR.glob("raw_data_*.csv"))
        if not raw_files:
            raise AirflowException("No raw data files found")
        
        latest_raw = raw_files[-1]
        logger.info(f"Loading raw data from {latest_raw}")
        
        df_raw = pd.read_csv(latest_raw, parse_dates=['datetime'])
        
        # Transform data
        df_transformed = transform_data(df_raw, save_processed=True)
        
        # Generate data profile report
        report_path = generate_data_profile(df_transformed)
        
        # Push info to XCom
        context['ti'].xcom_push(key='processed_data_shape', value=df_transformed.shape)
        context['ti'].xcom_push(key='data_profile_path', value=report_path)
        context['ti'].xcom_push(key='transformation_timestamp', value=datetime.now().isoformat())
        
        logger.info(f"✓ Data transformation successful: {df_transformed.shape[0]} rows, {df_transformed.shape[1]} columns")
        return "transformation_success"
        
    except Exception as e:
        logger.error(f"✗ Data transformation FAILED: {str(e)}")
        raise AirflowException(f"Data transformation failed: {str(e)}")


def load_to_minio_task(**context):
    """
    Task 3: Upload processed data to MinIO object storage.
    """
    if not IMPORTS_SUCCESSFUL:
        raise AirflowException("Required modules not available. Please ensure all dependencies are installed.")
    
    logger.info("=" * 60)
    logger.info("TASK 3: LOAD TO MINIO")
    logger.info("=" * 60)
    
    try:
        from config.config import PROCESSED_DATA_DIR
        
        # Get latest processed file
        processed_files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.parquet"))
        if not processed_files:
            raise AirflowException("No processed data files found")
        
        latest_processed = processed_files[-1]
        logger.info(f"Uploading {latest_processed} to MinIO")
        
        # Upload to MinIO
        minio_client = MinIOClient()
        object_name = minio_client.upload_file(str(latest_processed))
        
        # Push info to XCom
        context['ti'].xcom_push(key='minio_object_name', value=object_name)
        context['ti'].xcom_push(key='load_timestamp', value=datetime.now().isoformat())
        
        logger.info(f"✓ Data loaded to MinIO: {object_name}")
        return "load_success"
        
    except Exception as e:
        logger.error(f"✗ Load to MinIO FAILED: {str(e)}")
        raise AirflowException(f"Load to MinIO failed: {str(e)}")


def version_with_dvc_task(**context):
    """
    Task 4: Version data with DVC and push to Dagshub remote.
    """
    if not IMPORTS_SUCCESSFUL:
        raise AirflowException("Required modules not available. Please ensure all dependencies are installed.")
    
    logger.info("=" * 60)
    logger.info("TASK 4: VERSION WITH DVC")
    logger.info("=" * 60)
    
    try:
        import subprocess
        from config.config import PROCESSED_DATA_DIR
        
        # Get latest processed file
        processed_files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.parquet"))
        if not processed_files:
            raise AirflowException("No processed data files found")
        
        latest_processed = processed_files[-1]
        
        # DVC add
        logger.info(f"Adding {latest_processed} to DVC")
        subprocess.run(
            ["dvc", "add", str(latest_processed)],
            check=True,
            cwd=str(project_root)
        )
        
        # DVC push
        logger.info("Pushing to DVC remote")
        subprocess.run(
            ["dvc", "push"],
            check=True,
            cwd=str(project_root)
        )
        
        # Git add .dvc file
        dvc_file = latest_processed.with_suffix('.parquet.dvc')
        subprocess.run(
            ["git", "add", str(dvc_file)],
            check=True,
            cwd=str(project_root)
        )
        
        # Push info to XCom
        context['ti'].xcom_push(key='dvc_file', value=str(dvc_file))
        context['ti'].xcom_push(key='versioning_timestamp', value=datetime.now().isoformat())
        
        logger.info(f"✓ Data versioned with DVC")
        return "versioning_success"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ DVC versioning FAILED: {str(e)}")
        raise AirflowException(f"DVC versioning failed: {str(e)}")
    except Exception as e:
        logger.error(f"✗ DVC versioning FAILED: {str(e)}")
        raise AirflowException(f"DVC versioning failed: {str(e)}")


def log_mlflow_artifacts_task(**context):
    """
    Task 5: Log data profile report to MLflow.
    """
    if not IMPORTS_SUCCESSFUL:
        raise AirflowException("Required modules not available. Please ensure all dependencies are installed.")
    
    logger.info("=" * 60)
    logger.info("TASK 5: LOG ARTIFACTS TO MLFLOW")
    logger.info("=" * 60)
    
    try:
        import mlflow
        from config.config import MLFLOW_CONFIG
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
        
        # Get data profile path from previous task
        report_path = context['ti'].xcom_pull(key='data_profile_path', task_ids='transform_data')
        
        if report_path and Path(report_path).exists():
            with mlflow.start_run(run_name=f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("data_source", "Twelve Data API")
                mlflow.log_param("symbol", "EUR/USD")
                mlflow.log_param("interval", "1h")
                
                # Log metrics
                raw_shape = context['ti'].xcom_pull(key='raw_data_shape', task_ids='extract_data')
                processed_shape = context['ti'].xcom_pull(key='processed_data_shape', task_ids='transform_data')
                
                if raw_shape:
                    mlflow.log_metric("raw_data_rows", raw_shape[0])
                    mlflow.log_metric("raw_data_columns", raw_shape[1])
                
                if processed_shape:
                    mlflow.log_metric("processed_data_rows", processed_shape[0])
                    mlflow.log_metric("processed_data_columns", processed_shape[1])
                
                # Log artifact
                mlflow.log_artifact(report_path, artifact_path="data_quality_reports")
                
                logger.info(f"✓ Artifacts logged to MLflow")
        else:
            logger.warning("No data profile report found to log")
        
        return "mlflow_logging_success"
        
    except Exception as e:
        logger.error(f"✗ MLflow logging FAILED: {str(e)}")
        # Don't fail the DAG for logging issues
        logger.warning("Continuing despite MLflow logging failure")
        return "mlflow_logging_failed"

# Define the DAG
with DAG(
    dag_id='usd_volatility_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for USD volatility prediction - Runs every 2 hours',
    schedule='0 */2 * * *',  # Every 2 hours
    catchup=False,
    max_active_runs=1,
    tags=['etl', 'forex', 'mlops', 'production'],
) as dag:['etl', 'forex', 'mlops'],
) as dag:
    
    # Task 1: Extract data
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_task,
    )
    
    # Task 2: Transform data
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data_task,
    )
    
    # Task 3: Load to MinIO
    load_task = PythonOperator(
        task_id='load_to_minio',
        python_callable=load_to_minio_task,
    )
    
    # Task 4: Version with DVC
    version_task = PythonOperator(
        task_id='version_with_dvc',
        python_callable=version_with_dvc_task,
    )
    
    # Task 5: Log to MLflow
    mlflow_task = PythonOperator(
        task_id='log_mlflow_artifacts',
        python_callable=log_mlflow_artifacts_task,
    )
    
    # Define task dependencies
    extract_task >> transform_task >> load_task >> version_task >> mlflow_task
