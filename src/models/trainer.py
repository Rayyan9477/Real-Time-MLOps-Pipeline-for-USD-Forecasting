"""
Model training script with MLflow tracking and drift detection.
Trains XGBoost model for volatility prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.xgboost
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
import xgboost as xgb

from config import (
    MLFLOW_CONFIG,
    MODEL_CONFIG,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.utils.logger import get_logger
from src.utils.storage import MinIOClient

logger = get_logger("model_training")


class ModelTrainer:
    """Handles model training, evaluation, and drift detection."""

    def __init__(self, experiment_name: str = None):
        """
        Initialize model trainer with MLflow tracking.

        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name or MLFLOW_CONFIG["experiment_name"]
        self.model_config = MODEL_CONFIG

        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        mlflow.set_experiment(self.experiment_name)

        logger.info(f"Initialized ModelTrainer with experiment: {self.experiment_name}")

    @staticmethod
    def load_latest_data() -> pd.DataFrame:
        """
        Load the most recent processed data.

        Returns:
            DataFrame with processed features
        """
        # Try to get from MinIO first
        try:
            minio_client = MinIOClient()
            latest_object = minio_client.get_latest_object(prefix="processed_data")

            local_path = PROCESSED_DATA_DIR / latest_object
            if not local_path.exists():
                minio_client.download_file(latest_object, str(local_path))

            df = pd.read_parquet(local_path)
            logger.info(f"Loaded data from MinIO: {latest_object}")

        except Exception as e:
            logger.warning(
                f"Failed to load from MinIO: {e}. Loading from local storage."
            )

            # Fallback to local
            processed_files = sorted(
                PROCESSED_DATA_DIR.glob("processed_data_*.parquet")
            )
            if not processed_files:
                raise FileNotFoundError("No processed data files found")

            df = pd.read_parquet(processed_files[-1])
            logger.info(f"Loaded data from local: {processed_files[-1].name}")

        return df

    def prepare_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare features and target for training.

        Args:
            df: Processed DataFrame

        Returns:
            Tuple of (features, target, feature_names)
        """
        # Exclude non-feature columns
        exclude_cols = [
            "datetime",
            "target_volatility",
            "volatility",
            "open",
            "high",
            "low",
            "close",  # Keep only engineered features
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df["target_volatility"].copy()

        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(
            f"Feature columns: {feature_cols[:10]}... ({len(feature_cols)} total)"
        )

        return X, y, feature_cols

    def split_data(
        self, X: pd.DataFrame, y: pd.Series, train_ratio: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data for time series (no shuffle).

        Args:
            X: Features
            y: Target
            train_ratio: Ratio for train/test split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        train_ratio = train_ratio or self.model_config["train_test_split"]

        split_idx = int(len(X) * train_ratio)

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Dict = None,
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model with cross-validation.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            params: Model hyperparameters

        Returns:
            Trained XGBoost model
        """
        if params is None:
            params = self.model_config["xgboost_params"]

        logger.info("Training XGBoost model...")
        logger.info(f"Hyperparameters: {params}")

        # Create model
        model = xgb.XGBRegressor(**params)

        # Train with early stopping
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        logger.info("Model training complete")
        return model

    def evaluate_model(
        self, model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        }

        logger.info("Model Evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.6f}")

        return metrics

    def detect_drift(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, significance: float = 0.05
    ) -> Dict[str, float]:
        """
        Detect data drift using Kolmogorov-Smirnov test.

        Args:
            X_train: Training features
            X_test: Test features
            significance: Significance level for KS test

        Returns:
            Dictionary with drift scores
        """
        logger.info("Detecting data drift...")

        drift_scores = {}
        drifted_features = []

        for col in X_train.columns:
            try:
                statistic, p_value = ks_2samp(X_train[col], X_test[col])
                drift_scores[col] = p_value

                if p_value < significance:
                    drifted_features.append(col)
            except Exception as e:
                logger.warning(f"Could not compute drift for {col}: {e}")

        drift_ratio = len(drifted_features) / len(X_train.columns)

        logger.info(
            f"Drift detected in {len(drifted_features)}/{len(X_train.columns)} features"
        )
        logger.info(f"Drift ratio: {drift_ratio:.2%}")

        if drifted_features:
            logger.info(f"Drifted features: {drifted_features[:5]}...")

        return {
            "drift_ratio": drift_ratio,
            "num_drifted_features": len(drifted_features),
            "total_features": len(X_train.columns),
        }

    def plot_feature_importance(
        self, model: xgb.XGBRegressor, feature_names: list, top_n: int = 20
    ) -> str:
        """
        Plot and save feature importance.

        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to plot

        Returns:
            Path to saved plot
        """
        importance = model.feature_importances_
        indices = np.argsort(importance)[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = REPORTS_DIR / f"feature_importance_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved to {plot_path}")
        return str(plot_path)

    def plot_predictions(
        self, y_true: pd.Series, y_pred: np.ndarray, sample_size: int = 200
    ) -> str:
        """
        Plot actual vs predicted values.

        Args:
            y_true: True values
            y_pred: Predicted values
            sample_size: Number of samples to plot

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 6))

        # Plot subset for clarity
        plot_indices = range(min(sample_size, len(y_true)))

        plt.plot(plot_indices, y_true.iloc[plot_indices], label="Actual", alpha=0.7)
        plt.plot(plot_indices, y_pred[plot_indices], label="Predicted", alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Volatility")
        plt.title("Actual vs Predicted Volatility")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = REPORTS_DIR / f"predictions_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Predictions plot saved to {plot_path}")
        return str(plot_path)

    def train_and_log(self, hyperparams: Dict = None) -> str:
        """
        Complete training pipeline with MLflow logging.

        Args:
            hyperparams: Optional custom hyperparameters

        Returns:
            MLflow run ID
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 60)

        with mlflow.start_run(
            run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            try:
                # Load data
                df = self.load_latest_data()

                # Prepare features and target
                X, y, feature_names = self.prepare_features_target(df)

                # Split data
                X_train, X_test, y_train, y_test = self.split_data(X, y)

                # Log dataset info
                mlflow.log_param("total_samples", len(X))
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("num_features", X.shape[1])

                # Train model
                params = hyperparams or self.model_config["xgboost_params"]
                model = self.train_xgboost(X_train, y_train, X_test, y_test, params)

                # Log hyperparameters
                mlflow.log_params(params)

                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)
                mlflow.log_metrics(metrics)

                # Detect drift
                drift_metrics = self.detect_drift(X_train, X_test)
                mlflow.log_metrics(drift_metrics)

                # Generate plots
                importance_plot = self.plot_feature_importance(model, feature_names)
                mlflow.log_artifact(importance_plot, artifact_path="plots")

                y_pred = model.predict(X_test)
                pred_plot = self.plot_predictions(y_test, y_pred)
                mlflow.log_artifact(pred_plot, artifact_path="plots")

                # Log model
                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=MLFLOW_CONFIG["model_name"],
                )

                logger.info("=" * 60)
                logger.info("MODEL TRAINING COMPLETE ✓")
                logger.info(f"Run ID: {run.info.run_id}")
                logger.info("=" * 60)

                return run.info.run_id

            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise


def main():
    """Main entry point for training script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train USD volatility prediction model"
    )
    parser.add_argument("--experiment", type=str, help="MLflow experiment name")
    parser.add_argument("--n-estimators", type=int, help="Number of trees")
    parser.add_argument("--max-depth", type=int, help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")

    args = parser.parse_args()

    # Custom hyperparameters
    hyperparams = {}
    if args.n_estimators:
        hyperparams["n_estimators"] = args.n_estimators
    if args.max_depth:
        hyperparams["max_depth"] = args.max_depth
    if args.learning_rate:
        hyperparams["learning_rate"] = args.learning_rate

    # Train model
    trainer = ModelTrainer(experiment_name=args.experiment)
    run_id = trainer.train_and_log(hyperparams if hyperparams else None)

    print(f"\n✓ Training complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()
