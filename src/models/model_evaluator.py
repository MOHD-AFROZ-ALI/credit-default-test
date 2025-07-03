"""
Model Evaluation Module

This module provides essential evaluation capabilities for credit default prediction models,
including metrics calculation, performance analysis, and comparison tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    classification_report, confusion_matrix, precision_recall_curve,
    roc_curve, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
from datetime import datetime

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.helpers import Timer, ensure_dir, save_json

logger = get_logger(__name__)

class ModelEvaluator:
    """Essential model evaluation for credit default prediction."""

    def __init__(self):
        """Initialize ModelEvaluator."""
        self.config = {
            'cv_folds': get_config('models.cv_folds', 5),
            'random_state': get_config('models.random_state', 42),
            'min_auc_score': get_config('models.min_auc_score', 0.75)
        }
        self.evaluation_results = {}
        logger.info("ModelEvaluator initialized successfully")

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate model performance with essential metrics.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")

        with Timer(f"Model evaluation for {model_name}"):
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)

            # Cross-validation
            cv_scores = self._cross_validate(model, X_test, y_test)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Feature importance
            feature_importance = self._get_feature_importance(model, X_test.columns)

            # Compile results
            evaluation = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'metrics': metrics,
                'cross_validation': cv_scores,
                'confusion_matrix': cm.tolist(),
                'feature_importance': feature_importance,
                'n_samples': len(y_test),
                'n_features': X_test.shape[1],
                'timestamp': datetime.now().isoformat()
            }

            # Add curves if probabilities available
            if y_proba is not None:
                curves = self._calculate_curves(y_test, y_proba)
                evaluation['curves'] = curves

        self.evaluation_results[model_name] = evaluation
        logger.info(f"Evaluation completed. ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

        return evaluation

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate essential classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }

        if y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba),
                'average_precision': average_precision_score(y_true, y_proba),
                'log_loss': log_loss(y_true, y_proba)
            })

        return metrics

    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )

        cv_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                cv_results[metric] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
            except:
                cv_results[metric] = {'mean': 0.0, 'std': 0.0, 'scores': []}

        return cv_results

    def _calculate_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC and PR curves."""
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

        return {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }

    def _get_feature_importance(self, model, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Get feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        else:
            return []

        feature_importance = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(feature_names, importances)
        ]

        return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

    def compare_models(self, evaluation_results: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            evaluation_results: Optional dict of evaluation results

        Returns:
            DataFrame with model comparison
        """
        results = evaluation_results or self.evaluation_results

        if not results:
            return pd.DataFrame()

        comparison_data = []
        for model_name, eval_result in results.items():
            metrics = eval_result.get('metrics', {})
            cv_results = eval_result.get('cross_validation', {})

            row = {
                'model': model_name,
                'model_type': eval_result.get('model_type', 'Unknown'),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'cv_roc_auc_mean': cv_results.get('roc_auc', {}).get('mean', 0),
                'cv_roc_auc_std': cv_results.get('roc_auc', {}).get('std', 0),
                'n_samples': eval_result.get('n_samples', 0),
                'n_features': eval_result.get('n_features', 0)
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values('roc_auc', ascending=False)

        return df

    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, evaluation_results)
        """
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated")

        best_score = -1
        best_model = None
        best_results = None

        for model_name, results in self.evaluation_results.items():
            score = results.get('metrics', {}).get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
                best_results = results

        return best_model, best_results

    def generate_report(self, model_name: str) -> Dict[str, Any]:
        """
        Generate evaluation report for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with formatted report
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results for model: {model_name}")

        results = self.evaluation_results[model_name]
        metrics = results.get('metrics', {})

        # Performance assessment
        roc_auc = metrics.get('roc_auc', 0)
        if roc_auc >= 0.85:
            performance = "Excellent"
        elif roc_auc >= 0.75:
            performance = "Good"
        elif roc_auc >= 0.65:
            performance = "Fair"
        else:
            performance = "Poor"

        report = {
            'model_name': model_name,
            'model_type': results.get('model_type', 'Unknown'),
            'performance_level': performance,
            'key_metrics': {
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.3f}",
                'Precision': f"{metrics.get('precision', 0):.3f}",
                'Recall': f"{metrics.get('recall', 0):.3f}",
                'F1-Score': f"{metrics.get('f1_score', 0):.3f}",
                'Accuracy': f"{metrics.get('accuracy', 0):.3f}"
            },
            'cross_validation': results.get('cross_validation', {}),
            'sample_info': {
                'test_samples': results.get('n_samples', 0),
                'features': results.get('n_features', 0)
            },
            'top_features': results.get('feature_importance', [])[:10],
            'recommendations': self._generate_recommendations(metrics),
            'timestamp': results.get('timestamp', '')
        }

        return report

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        roc_auc = metrics.get('roc_auc', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)

        if roc_auc < self.config['min_auc_score']:
            recommendations.append(f"ROC-AUC ({roc_auc:.3f}) below target ({self.config['min_auc_score']}). Consider feature engineering.")

        if precision < 0.7:
            recommendations.append(f"Low precision ({precision:.3f}). Consider adjusting threshold or improving features.")

        if recall < 0.65:
            recommendations.append(f"Low recall ({recall:.3f}). Consider class balancing or threshold adjustment.")

        if abs(precision - recall) > 0.2:
            recommendations.append("Significant precision-recall imbalance. Review threshold optimization.")

        if not recommendations:
            recommendations.append("Model performance meets criteria. Ready for deployment.")

        return recommendations

    def save_results(self, filepath: str, model_name: Optional[str] = None) -> None:
        """
        Save evaluation results to file.

        Args:
            filepath: Path to save results
            model_name: Specific model to save (if None, saves all)
        """
        data = self.evaluation_results
        if model_name:
            if model_name not in self.evaluation_results:
                raise ValueError(f"No results for model: {model_name}")
            data = {model_name: self.evaluation_results[model_name]}

        ensure_dir(Path(filepath).parent)
        save_json(data, filepath)
        logger.info(f"Evaluation results saved to {filepath}")

    def load_results(self, filepath: str) -> None:
        """Load evaluation results from file."""
        with open(filepath, 'r') as f:
            self.evaluation_results = json.load(f)
        logger.info(f"Evaluation results loaded from {filepath}")

# Convenience functions
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                  model_name: str = "Model") -> Dict[str, Any]:
    """
    Quick model evaluation function.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model

    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X_test, y_test, model_name)

def compare_multiple_models(models: Dict[str, Any], X_test: pd.DataFrame, 
                           y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        models: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test target

    Returns:
        Comparison DataFrame
    """
    evaluator = ModelEvaluator()

    for model_name, model in models.items():
        evaluator.evaluate_model(model, X_test, y_test, model_name)

    return evaluator.compare_models()

def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Get detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Classification report dictionary
    """
    return classification_report(y_true, y_pred, output_dict=True)

def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate business-specific metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Business metrics dictionary
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total = len(y_true)
    actual_defaults = np.sum(y_true)
    predicted_defaults = np.sum(y_pred)

    metrics = {
        'total_customers': int(total),
        'actual_defaults': int(actual_defaults),
        'predicted_defaults': int(predicted_defaults),
        'actual_default_rate': float(actual_defaults / total),
        'predicted_default_rate': float(predicted_defaults / total),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'correctly_identified_defaults': int(tp),
        'missed_defaults': int(fn),
        'false_alarms': int(fp)
    }

    if y_proba is not None:
        # Risk distribution
        low_risk = np.sum(y_proba < 0.3)
        medium_risk = np.sum((y_proba >= 0.3) & (y_proba < 0.7))
        high_risk = np.sum(y_proba >= 0.7)

        metrics['risk_distribution'] = {
            'low_risk_count': int(low_risk),
            'medium_risk_count': int(medium_risk),
            'high_risk_count': int(high_risk),
            'low_risk_pct': float(low_risk / total * 100),
            'medium_risk_pct': float(medium_risk / total * 100),
            'high_risk_pct': float(high_risk / total * 100)
        }

    return metrics
