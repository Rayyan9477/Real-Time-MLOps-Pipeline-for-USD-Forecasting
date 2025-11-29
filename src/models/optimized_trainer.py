"""
Optimized model training with hyperparameter tuning and ensemble methods.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger('model_training_optimized', 'logs/train_optimized.log')


class OptimizedModelTrainer:
    """Advanced model trainer with hyperparameter tuning and ensemble methods."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to processed data file
        """
        self.data_path = data_path
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Create directories
        Path('models').mkdir(exist_ok=True)
        Path('reports').mkdir(exist_ok=True)
        Path('reports/optimization').mkdir(exist_ok=True)
        
        logger.info("Initialized OptimizedModelTrainer")
    
    def load_data(self) -> pd.DataFrame:
        """Load the latest processed data."""
        if self.data_path and Path(self.data_path).exists():
            df = pd.read_parquet(self.data_path)
            logger.info(f"Loaded data from {self.data_path}: {df.shape}")
        else:
            processed_files = sorted(list(Path('data/processed').glob('*.parquet')))
            if not processed_files:
                raise FileNotFoundError("No processed data files found")
            df = pd.read_parquet(processed_files[-1])
            logger.info(f"Loaded latest processed data: {df.shape}")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and target, split into train/test.
        
        Args:
            df: Input DataFrame
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Exclude non-feature columns
        exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'target_volatility']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['target_volatility'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split (80/20)
        split_idx = int(len(df) * 0.8)
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Prepared data: {len(feature_cols)} features")
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost_tuned(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train XGBoost with hyperparameter tuning.
        
        Args:
            X_train, y_train, X_test, y_test: Training and test data
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Starting XGBoost hyperparameter tuning...")
        
        # Define parameter grid
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5)
        }
        
        # Base model
        xgb_model = xgb.XGBRegressor(
            random_state=42,
            tree_method='hist',
            verbosity=0
        )
        
        # Randomized search with time series CV
        tscv = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=20,
            scoring='neg_mean_squared_error',
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"XGBoost tuned - Best params: {random_search.best_params_}")
        logger.info(f"XGBoost metrics: RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")
        
        return {
            'model': best_model,
            'predictions': y_pred,
            'metrics': metrics,
            'best_params': random_search.best_params_
        }
    
    def train_random_forest_tuned(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train Random Forest with hyperparameter tuning.
        
        Args:
            X_train, y_train, X_test, y_test: Training and test data
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Starting Random Forest hyperparameter tuning...")
        
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.5, 0.7]
        }
        
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        tscv = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            rf_model,
            param_distributions=param_dist,
            n_iter=15,
            scoring='neg_mean_squared_error',
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"Random Forest tuned - Best params: {random_search.best_params_}")
        logger.info(f"Random Forest metrics: RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")
        
        return {
            'model': best_model,
            'predictions': y_pred,
            'metrics': metrics,
            'best_params': random_search.best_params_
        }
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Gradient Boosting Regressor."""
        logger.info("Training Gradient Boosting...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb_model = GradientBoostingRegressor(random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            gb_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=tscv,
            verbose=0,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"Gradient Boosting - Best params: {grid_search.best_params_}")
        logger.info(f"Gradient Boosting metrics: RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")
        
        return {
            'model': best_model,
            'predictions': y_pred,
            'metrics': metrics,
            'best_params': grid_search.best_params_
        }
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      individual_models: Dict) -> Dict[str, Any]:
        """
        Create ensemble model from best individual models.
        
        Args:
            X_train, y_train, X_test, y_test: Training and test data
            individual_models: Dictionary of trained models
            
        Returns:
            Dictionary with ensemble model and metrics
        """
        logger.info("Creating ensemble model...")
        
        # Select top 3 models based on R² score
        sorted_models = sorted(
            individual_models.items(),
            key=lambda x: x[1]['metrics']['r2'],
            reverse=True
        )[:3]
        
        estimators = [(name, result['model']) for name, result in sorted_models]
        
        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(X_train, y_train)
        
        y_pred = ensemble.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"Ensemble metrics: RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")
        logger.info(f"Ensemble members: {[name for name, _ in estimators]}")
        
        return {
            'model': ensemble,
            'predictions': y_pred,
            'metrics': metrics,
            'members': [name for name, _ in estimators]
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare all model results.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison = []
        for name, result in results.items():
            metrics = result['metrics']
            comparison.append({
                'Model': name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('R²', ascending=False)
        
        logger.info(f"\n{df.to_string()}")
        
        return df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Plot model comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['RMSE', 'MAE', 'R²', 'MAPE']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (metric, ax, color) in enumerate(zip(metrics, axes.flatten(), colors)):
            df_sorted = comparison_df.sort_values(metric, ascending=(metric != 'R²'))
            ax.barh(df_sorted['Model'], df_sorted[metric], color=color, alpha=0.7)
            ax.set_xlabel(metric, fontsize=12)
            ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(df_sorted[metric]):
                ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {save_path}")
        plt.close()
    
    def plot_predictions(self, y_test: np.ndarray, results: Dict[str, Dict], 
                        save_path: str = None):
        """Plot predictions for all models."""
        n_models = len(results)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(results.items()):
            y_pred = result['predictions']
            metrics = result['metrics']
            
            axes[idx].plot(y_test, label='Actual', marker='o', alpha=0.7, linewidth=2)
            axes[idx].plot(y_pred, label='Predicted', marker='x', alpha=0.7, linewidth=2)
            axes[idx].set_title(
                f'{name} - RMSE: {metrics["rmse"]:.6f}, R²: {metrics["r2"]:.4f}',
                fontsize=12, fontweight='bold'
            )
            axes[idx].set_xlabel('Sample Index')
            axes[idx].set_ylabel('Volatility')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")
        plt.close()
    
    def save_best_model(self, name: str, model: Any, metrics: Dict, params: Dict = None):
        """Save the best model and its metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        if hasattr(model, 'save_model'):
            model_path = f'models/best_model_{name}_{timestamp}.json'
            model.save_model(model_path)
        else:
            import joblib
            model_path = f'models/best_model_{name}_{timestamp}.pkl'
            joblib.dump(model, model_path)
        
        logger.info(f"Saved best model to {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': name,
            'timestamp': timestamp,
            'metrics': metrics,
            'parameters': params or {},
            'features': self.feature_names,
            'model_path': model_path
        }
        
        metadata_path = f'models/best_model_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main training function."""
    print("=" * 80)
    print("OPTIMIZED MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 80)
    
    trainer = OptimizedModelTrainer()
    
    # Load and prepare data
    print("\n[1] Loading and preparing data...")
    df = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train multiple models
    results = {}
    
    print("\n[2] Training XGBoost with hyperparameter tuning...")
    results['XGBoost_Tuned'] = trainer.train_xgboost_tuned(X_train, y_train, X_test, y_test)
    print(f"✓ RMSE: {results['XGBoost_Tuned']['metrics']['rmse']:.6f}, "
          f"R²: {results['XGBoost_Tuned']['metrics']['r2']:.4f}")
    
    print("\n[3] Training Random Forest with hyperparameter tuning...")
    results['RandomForest_Tuned'] = trainer.train_random_forest_tuned(X_train, y_train, X_test, y_test)
    print(f"✓ RMSE: {results['RandomForest_Tuned']['metrics']['rmse']:.6f}, "
          f"R²: {results['RandomForest_Tuned']['metrics']['r2']:.4f}")
    
    print("\n[4] Training Gradient Boosting...")
    results['GradientBoosting'] = trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
    print(f"✓ RMSE: {results['GradientBoosting']['metrics']['rmse']:.6f}, "
          f"R²: {results['GradientBoosting']['metrics']['r2']:.4f}")
    
    print("\n[5] Creating ensemble model...")
    results['Ensemble'] = trainer.train_ensemble(X_train, y_train, X_test, y_test, results)
    print(f"✓ RMSE: {results['Ensemble']['metrics']['rmse']:.6f}, "
          f"R²: {results['Ensemble']['metrics']['r2']:.4f}")
    
    # Compare models
    print("\n[6] Comparing all models...")
    comparison_df = trainer.compare_models(results)
    
    # Get best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = results[best_model_name]
    print(f"\n✓ Best model: {best_model_name}")
    print(f"  RMSE: {best_result['metrics']['rmse']:.6f}")
    print(f"  R²: {best_result['metrics']['r2']:.4f}")
    print(f"  MAE: {best_result['metrics']['mae']:.6f}")
    print(f"  MAPE: {best_result['metrics']['mape']:.2f}%")
    
    # Save visualizations
    print("\n[7] Generating visualizations...")
    trainer.plot_model_comparison(
        comparison_df,
        'reports/optimization/model_comparison.png'
    )
    trainer.plot_predictions(
        y_test,
        results,
        'reports/optimization/all_predictions.png'
    )
    print("✓ Saved plots")
    
    # Save best model
    print("\n[8] Saving best model...")
    trainer.save_best_model(
        best_model_name,
        best_result['model'],
        best_result['metrics'],
        best_result.get('best_params')
    )
    
    print("\n" + "=" * 80)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Performance Improvement: R² = {best_result['metrics']['r2']:.4f}")
    print(f"Reports saved to: reports/optimization/")


if __name__ == "__main__":
    main()
