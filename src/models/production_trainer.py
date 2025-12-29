"""
Production-ready model trainer with optimized hyperparameters and ensemble methods.
Designed for high accuracy USD volatility prediction.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any
import json
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler
import joblib

from config.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("production_model_trainer")


class ProductionModelTrainer:
    """Production-ready model trainer with state-of-the-art algorithms."""
    
    def __init__(self):
        """Initialize the production trainer."""
        self.scaler = RobustScaler()  # More robust to outliers
        self.best_model = None
        self.feature_names = None
        self.metadata = {}
        
        logger.info("Initialized ProductionModelTrainer")
    
    def load_latest_data(self) -> pd.DataFrame:
        """Load the most recent processed data."""
        processed_files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.parquet"))
        
        if not processed_files:
            raise FileNotFoundError("No processed data files found")
        
        latest_file = processed_files[-1]
        df = pd.read_parquet(latest_file)
        
        logger.info(f"Loaded data from {latest_file.name}: {df.shape}")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Prepare features and target with advanced feature engineering.
        
        Args:
            df: Input DataFrame
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Exclude non-feature columns
        exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'target_volatility']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['target_volatility'].values
        
        # Remove any NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Time series split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.feature_names = feature_cols
        
        logger.info(f"Prepared data - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Features: {feature_cols}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def build_optimized_model(self) -> Any:
        """
        Build an optimized ensemble model with best hyperparameters.
        
        Returns:
            Trained ensemble model
        """
        # Base models with optimized hyperparameters
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Stack models with Ridge meta-learner (use cv=5 instead of TimeSeriesSplit)
        ensemble_model = StackingRegressor(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5,  # Use integer splits instead of TimeSeriesSplit for compatibility
            n_jobs=-1
        )
        
        logger.info("Built optimized ensemble model")
        return ensemble_model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        logger.info("Training ensemble model...")
        
        model = self.build_optimized_model()
        model.fit(X_train, y_train)
        
        # Cross-validation score
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        logger.info(f"Cross-validation RMSE: {cv_rmse:.6f}")
        
        self.best_model = model
        return model
    
    def evaluate(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            # Training metrics
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'train_mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            
            # Test metrics
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
            
            # Additional metrics
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'n_features': X_train.shape[1]
        }
        
        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info("=" * 60)
        logger.info(f"Training Set:")
        logger.info(f"  RMSE: {metrics['train_rmse']:.6f}")
        logger.info(f"  MAE:  {metrics['train_mae']:.6f}")
        logger.info(f"  R²:   {metrics['train_r2']:.4f}")
        logger.info(f"  MAPE: {metrics['train_mape']:.2f}%")
        logger.info(f"\nTest Set:")
        logger.info(f"  RMSE: {metrics['test_rmse']:.6f}")
        logger.info(f"  MAE:  {metrics['test_mae']:.6f}")
        logger.info(f"  R²:   {metrics['test_r2']:.4f}")
        logger.info(f"  MAPE: {metrics['test_mape']:.2f}%")
        logger.info("=" * 60)
        
        return metrics
    
    def save_model(self, model: Any, metrics: Dict[str, float], feature_names: list):
        """
        Save model and metadata.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            feature_names: List of feature names
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = MODELS_DIR / "latest_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'StackingRegressor_Ensemble',
            'timestamp': timestamp,
            'features': feature_names,
            'metrics': metrics,
            'feature_names': feature_names,
            'base_models': ['XGBoost', 'RandomForest', 'GradientBoosting'],
            'meta_learner': 'Ridge'
        }
        
        metadata_path = MODELS_DIR / "latest_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        self.metadata = metadata
    
    def run_training_pipeline(self):
        """Execute the complete training pipeline."""
        logger.info("Starting production model training pipeline...")
        
        try:
            # Load data
            df = self.load_latest_data()
            
            # Prepare data
            X_train, X_test, y_train, y_test, feature_names = self.prepare_data(df)
            
            # Train model
            model = self.train(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate(model, X_train, y_train, X_test, y_test)
            
            # Save
            self.save_model(model, metrics, feature_names)
            
            logger.info("✓ Training pipeline completed successfully!")
            logger.info(f"Final Test R²: {metrics['test_r2']:.4f}")
            logger.info(f"Final Test MAPE: {metrics['test_mape']:.2f}%")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run training."""
    trainer = ProductionModelTrainer()
    model, metrics = trainer.run_training_pipeline()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Test R² Score: {metrics['test_r2']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.6f}")
    print(f"Test MAPE: {metrics['test_mape']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
