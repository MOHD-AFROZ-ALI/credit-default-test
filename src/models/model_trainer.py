"""
Model Trainer Module for Credit Default Prediction System

This module handles training of multiple machine learning models including:
- XGBoost
- LightGBM  
- CatBoost
- Random Forest
- Logistic Regression

Features:
- Hyperparameter tuning with Optuna
- Cross-validation
- Model persistence
- Performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Import our utilities
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.utils.helpers import ConfigManager, FileManager, MetricsCalculator


class ModelTrainer:
    """
    Comprehensive model trainer for credit default prediction.

    This class handles training, hyperparameter tuning, and evaluation
    of multiple machine learning models.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the model trainer.

        Args:
            config_path (str): Path to configuration file
        """
        self.logger = get_logger()
        self.config = ConfigManager(config_path)

        # Training parameters
        self.random_state = self.config.get('data.random_state', 42)
        self.cv_folds = self.config.get('evaluation.cv_folds', 5)

        # Hyperparameter tuning parameters
        self.n_trials = self.config.get('hyperparameter_tuning.n_trials', 50)
        self.timeout = self.config.get('hyperparameter_tuning.timeout', 3600)

        # Model storage
        self.models = {}
        self.model_configs = {}
        self.training_results = {}

        # Model availability check
        self.available_models = self._check_model_availability()

        self.logger.info(f"ModelTrainer initialized. Available models: {list(self.available_models.keys())}")

    def _check_model_availability(self) -> Dict[str, bool]:
        """Check which ML libraries are available."""
        availability = {
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE,
            'random_forest': True,  # Always available with sklearn
            'logistic_regression': True  # Always available with sklearn
        }

        unavailable = [name for name, available in availability.items() if not available]
        if unavailable:
            self.logger.warning(f"Unavailable models: {unavailable}")

        return availability

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a model.

        Args:
            model_name (str): Name of the model

        Returns:
            Dict[str, Any]: Default parameters
        """
        default_params = {
            'xgboost': {
                'n_estimators': self.config.get('models.xgboost.n_estimators', 100),
                'max_depth': self.config.get('models.xgboost.max_depth', 6),
                'learning_rate': self.config.get('models.xgboost.learning_rate', 0.1),
                'subsample': self.config.get('models.xgboost.subsample', 0.8),
                'colsample_bytree': self.config.get('models.xgboost.colsample_bytree', 0.8),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            },
            'lightgbm': {
                'n_estimators': self.config.get('models.lightgbm.n_estimators', 100),
                'max_depth': self.config.get('models.lightgbm.max_depth', 6),
                'learning_rate': self.config.get('models.lightgbm.learning_rate', 0.1),
                'subsample': self.config.get('models.lightgbm.subsample', 0.8),
                'colsample_bytree': self.config.get('models.lightgbm.colsample_bytree', 0.8),
                'random_state': self.random_state,
                'verbose': -1
            },
            'catboost': {
                'iterations': self.config.get('models.catboost.iterations', 100),
                'depth': self.config.get('models.catboost.depth', 6),
                'learning_rate': self.config.get('models.catboost.learning_rate', 0.1),
                'random_state': self.random_state,
                'verbose': False,
                'loss_function': 'Logloss'
            },
            'random_forest': {
                'n_estimators': self.config.get('models.random_forest.n_estimators', 100),
                'max_depth': self.config.get('models.random_forest.max_depth', 10),
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'logistic_regression': {
                'random_state': self.random_state,
                'max_iter': self.config.get('models.logistic_regression.max_iter', 1000),
                'solver': 'liblinear'
            }
        }

        return default_params.get(model_name, {})

    def create_model(self, model_name: str, params: Dict[str, Any] = None) -> Any:
        """
        Create a model instance.

        Args:
            model_name (str): Name of the model
            params (Dict[str, Any]): Model parameters

        Returns:
            Any: Model instance
        """
        if not self.available_models.get(model_name, False):
            raise ValueError(f"Model {model_name} is not available")

        if params is None:
            params = self.get_default_params(model_name)

        if model_name == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(**params)
        elif model_name == 'catboost':
            return cb.CatBoostClassifier(**params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame = None, y_val: pd.Series = None,
                   params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train a single model.

        Args:
            model_name (str): Name of the model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            params (Dict[str, Any]): Model parameters

        Returns:
            Dict[str, Any]: Training results
        """
        self.logger.info(f"Training {model_name} model")

        # Create model
        model = self.create_model(model_name, params)

        # Train model
        if model_name in ['xgboost', 'lightgbm', 'catboost'] and X_val is not None:
            # Use validation set for early stopping
            if model_name == 'xgboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            elif model_name == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            elif model_name == 'catboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
        else:
            # Standard training
            model.fit(X_train, y_train)

        # Store model
        self.models[model_name] = model
        self.model_configs[model_name] = params or self.get_default_params(model_name)

        # Evaluate model
        train_results = self._evaluate_model(model, X_train, y_train, "train")

        results = {
            'model': model,
            'params': self.model_configs[model_name],
            'train_metrics': train_results
        }

        if X_val is not None:
            val_results = self._evaluate_model(model, X_val, y_val, "validation")
            results['val_metrics'] = val_results

        self.training_results[model_name] = results
        self.logger.info(f"{model_name} training completed")

        return results

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate a model on given data.

        Args:
            model: Trained model
            X (pd.DataFrame): Features
            y (pd.Series): Target
            dataset_name (str): Name of the dataset

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y, y_prob),
            'log_loss': log_loss(y, y_prob)
        }

        self.logger.info(f"{dataset_name} metrics: {metrics}")
        return metrics

    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame = None, y_val: pd.Series = None,
                                metric: str = 'roc_auc') -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            model_name (str): Name of the model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            metric (str): Optimization metric

        Returns:
            Dict[str, Any]: Optimization results
        """
        self.logger.info(f"Optimizing hyperparameters for {model_name}")

        def objective(trial):
            # Define hyperparameter search space
            params = self._get_hyperparameter_space(trial, model_name)

            # Create and train model
            model = self.create_model(model_name, params)

            if X_val is not None:
                # Use validation set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                if metric == 'roc_auc':
                    score = roc_auc_score(y_val, y_prob)
                elif metric == 'f1_score':
                    score = f1_score(y_val, y_pred, average='weighted')
                elif metric == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = roc_auc_score(y_val, y_prob)
            else:
                # Use cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                    scoring=metric,
                    n_jobs=-1
                )
                score = cv_scores.mean()

            return score

        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"Best {metric} for {model_name}: {best_score:.4f}")
        self.logger.info(f"Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }

    def _get_hyperparameter_space(self, trial, model_name: str) -> Dict[str, Any]:
        """
        Define hyperparameter search space for each model.

        Args:
            trial: Optuna trial object
            model_name (str): Name of the model

        Returns:
            Dict[str, Any]: Hyperparameter space
        """
        if model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }

        elif model_name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'random_state': self.random_state,
                'verbose': -1
            }

        elif model_name == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': self.random_state,
                'verbose': False,
                'loss_function': 'Logloss'
            }

        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'n_jobs': -1
            }

        elif model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'liblinear',
                'random_state': self.random_state,
                'max_iter': 1000
            }

        else:
            return {}

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame = None, y_val: pd.Series = None,
                        optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train all available models.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            optimize_hyperparams (bool): Whether to optimize hyperparameters

        Returns:
            Dict[str, Any]: Training results for all models
        """
        self.logger.info("Training all available models")

        all_results = {}

        for model_name in self.available_models:
            if not self.available_models[model_name]:
                continue

            try:
                if optimize_hyperparams:
                    # Optimize hyperparameters first
                    opt_results = self.optimize_hyperparameters(
                        model_name, X_train, y_train, X_val, y_val
                    )
                    best_params = opt_results['best_params']
                else:
                    best_params = None

                # Train model with best parameters
                results = self.train_model(
                    model_name, X_train, y_train, X_val, y_val, best_params
                )

                if optimize_hyperparams:
                    results['optimization'] = opt_results

                all_results[model_name] = results

            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                continue

        self.logger.info(f"Completed training {len(all_results)} models")
        return all_results

    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any, Dict[str, float]]:
        """
        Get the best performing model.

        Args:
            metric (str): Metric to use for comparison

        Returns:
            Tuple[str, Any, Dict[str, float]]: Best model name, model, and metrics
        """
        if not self.training_results:
            raise ValueError("No models have been trained yet")

        best_score = -np.inf
        best_model_name = None
        best_model = None
        best_metrics = None

        for model_name, results in self.training_results.items():
            # Use validation metrics if available, otherwise training metrics
            metrics = results.get('val_metrics', results.get('train_metrics', {}))

            if metric in metrics:
                score = metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = results['model']
                    best_metrics = metrics

        if best_model_name is None:
            raise ValueError(f"No models found with metric {metric}")

        self.logger.info(f"Best model: {best_model_name} with {metric}: {best_score:.4f}")
        return best_model_name, best_model, best_metrics

    def save_models(self, models_dir: str = "data/models") -> None:
        """
        Save all trained models.

        Args:
            models_dir (str): Directory to save models
        """
        FileManager.ensure_directory(models_dir)

        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            FileManager.save_model(model, model_path)
            self.logger.info(f"Saved {model_name} to {model_path}")

        # Save training results
        results_path = os.path.join(models_dir, "training_results.json")

        # Convert results to serializable format
        serializable_results = {}
        for model_name, results in self.training_results.items():
            serializable_results[model_name] = {
                'params': results['params'],
                'train_metrics': results['train_metrics'],
                'val_metrics': results.get('val_metrics', {})
            }

        FileManager.save_json(serializable_results, results_path)
        self.logger.info(f"Saved training results to {results_path}")

    def load_models(self, models_dir: str = "data/models") -> None:
        """
        Load saved models.

        Args:
            models_dir (str): Directory containing saved models
        """
        for model_name in self.available_models:
            if not self.available_models[model_name]:
                continue

            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")

            if os.path.exists(model_path):
                try:
                    model = FileManager.load_model(model_path)
                    self.models[model_name] = model
                    self.logger.info(f"Loaded {model_name} from {model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load {model_name}: {str(e)}")

        # Load training results
        results_path = os.path.join(models_dir, "training_results.json")
        if os.path.exists(results_path):
            try:
                self.training_results = FileManager.load_json(results_path)
                self.logger.info(f"Loaded training results from {results_path}")
            except Exception as e:
                self.logger.error(f"Failed to load training results: {str(e)}")

    def get_feature_importance(self, model_name: str, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance for a model.

        Args:
            model_name (str): Name of the model
            feature_names (List[str]): List of feature names

        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")

        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = None, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation for a model.

        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Number of CV folds
            scoring (str): Scoring metric

        Returns:
            Dict[str, Any]: Cross-validation results
        """
        if cv_folds is None:
            cv_folds = self.cv_folds

        # Create model with default parameters
        model = self.create_model(model_name)

        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            n_jobs=-1
        )

        results = {
            'scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scoring_metric': scoring,
            'cv_folds': cv_folds
        }

        self.logger.info(f"{model_name} CV {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")

        return results


# Convenience functions
def train_credit_models(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None,
                       config_path: str = "config.yaml") -> ModelTrainer:
    """
    Convenience function to train credit default models.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        config_path (str): Path to configuration file

    Returns:
        ModelTrainer: Trained model trainer instance
    """
    trainer = ModelTrainer(config_path)
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    return trainer


# Example usage and testing
if __name__ == "__main__":
    print("Testing Model Trainer...")

    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.choice([0, 1], n_samples))

    X_val = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.choice([0, 1], 200))

    # Test model trainer
    trainer = ModelTrainer()

    # Train a single model
    results = trainer.train_model('logistic_regression', X_train, y_train, X_val, y_val)
    print(f"Logistic Regression trained successfully!")
    print(f"Validation ROC-AUC: {results['val_metrics']['roc_auc']:.4f}")

    # Train all available models
    all_results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    print(f"Trained {len(all_results)} models successfully!")

    # Get best model
    best_name, best_model, best_metrics = trainer.get_best_model()
    print(f"Best model: {best_name} with ROC-AUC: {best_metrics['roc_auc']:.4f}")

    print("Model trainer testing completed successfully!")
