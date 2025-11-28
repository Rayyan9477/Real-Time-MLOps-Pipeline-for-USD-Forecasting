"""
Local model training script - trains and saves model locally without MLflow dependency.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import PROCESSED_DATA_DIR, MODELS_DIR, MODEL_CONFIG
from src.utils.logger import get_logger

logger = get_logger("local_training")


class LocalModelTrainer:
    """Train and save model locally without MLflow."""
    
    def __init__(self):
        self.model_dir = MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.metrics = {}
        self.feature_names = []
        
    def load_latest_data(self) -> pd.DataFrame:
        """Load the most recent processed data."""
        processed_files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.parquet"))
        
        if not processed_files:
            raise FileNotFoundError("No processed data files found in data/processed/")
        
        latest_file = processed_files[-1]
        df = pd.read_parquet(latest_file)
        logger.info(f"Loaded data from: {latest_file.name}, shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Prepare features and target."""
        # Define feature columns (exclude non-features)
        exclude_cols = [
            'datetime', 'target_volatility', 'volatility',
            'open', 'high', 'low', 'close', 'volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Check if target exists
        target_col = 'target_volatility'
        if target_col not in df.columns:
            # Create target as next period volatility
            if 'volatility' in df.columns:
                df['target_volatility'] = df['volatility'].shift(-1)
                target_col = 'target_volatility'
            else:
                # Calculate volatility from returns
                if 'log_return' in df.columns:
                    df['volatility'] = df['log_return'].rolling(window=24).std()
                    df['target_volatility'] = df['volatility'].shift(-1)
                else:
                    raise ValueError("Cannot create target - no volatility or log_return column")
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove NaN rows
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = feature_cols
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        
        return X, y, feature_cols
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        """Train XGBoost model."""
        # Split data (time series - no shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Model parameters
        params = MODEL_CONFIG.get("xgboost_params", {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        })
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(self.feature_names)
        }
        
        logger.info("Model Metrics:")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.6f}")
            else:
                logger.info(f"  {k}: {v}")
        
        return self.model
    
    def save_model(self) -> Dict[str, str]:
        """Save model and metadata to local files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model as pickle
        model_path = self.model_dir / f"xgboost_model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Also save as JSON (XGBoost native format)
        model_json_path = self.model_dir / f"xgboost_model_{timestamp}.json"
        self.model.save_model(str(model_json_path))
        
        # Save metadata
        metadata = {
            "model_type": "XGBRegressor",
            "timestamp": timestamp,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "model_path": str(model_path),
            "model_json_path": str(model_json_path)
        }
        
        metadata_path = self.model_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create symlinks to latest
        latest_model = self.model_dir / "latest_model.pkl"
        latest_json = self.model_dir / "latest_model.json"
        latest_metadata = self.model_dir / "latest_metadata.json"
        
        # Remove old symlinks/files
        for p in [latest_model, latest_json, latest_metadata]:
            if p.exists() or p.is_symlink():
                p.unlink()
        
        # Copy instead of symlink for compatibility
        import shutil
        shutil.copy(model_path, latest_model)
        shutil.copy(model_json_path, latest_json)
        shutil.copy(metadata_path, latest_metadata)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "latest_model": str(latest_model),
            "latest_metadata": str(latest_metadata)
        }
    
    def run_training_pipeline(self) -> Dict:
        """Run complete training pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING LOCAL MODEL TRAINING")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_latest_data()
        
        # Prepare features
        X, y, features = self.prepare_features(df)
        
        # Train model
        self.train(X, y)
        
        # Save model
        paths = self.save_model()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "metrics": self.metrics,
            "paths": paths
        }


def main():
    """Main entry point."""
    trainer = LocalModelTrainer()
    result = trainer.run_training_pipeline()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
