"""
Model Explainability Components with SHAP and LIME Integration

This module provides comprehensive explainability tools for machine learning models,
including SHAP and LIME integration, feature importance visualization, and
interactive explanation interfaces.

Author: AI Assistant
Created: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import logging
from pathlib import Path

# Core ML libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime import lime_tabular
    from lime.lime_text import LimeTextExplainer
    from lime.lime_image import LimeImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplainabilityConfig:
    """Configuration class for explainability settings."""

    # SHAP Configuration
    shap_explainer_type: str = "auto"  # auto, tree, linear, kernel, deep, gradient
    shap_background_samples: int = 100
    shap_max_evals: int = 1000
    shap_batch_size: int = 50

    # LIME Configuration
    lime_mode: str = "tabular"  # tabular, text, image
    lime_num_features: int = 10
    lime_num_samples: int = 5000
    lime_discretize_continuous: bool = True

    # Visualization Configuration
    plot_style: str = "plotly"  # plotly, matplotlib, seaborn
    color_scheme: str = "RdBu"
    figure_size: Tuple[int, int] = (12, 8)
    save_plots: bool = True
    output_dir: str = "/home/user/output/"

    # Feature Importance Configuration
    importance_method: str = "shap"  # shap, lime, permutation
    top_k_features: int = 20
    normalize_importance: bool = True

    # Performance Configuration
    parallel_processing: bool = True
    n_jobs: int = -1
    random_state: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not SHAP_AVAILABLE and self.importance_method == "shap":
            logger.warning("SHAP not available, switching to permutation importance")
            self.importance_method = "permutation"

        if not LIME_AVAILABLE and self.importance_method == "lime":
            logger.warning("LIME not available, switching to permutation importance")
            self.importance_method = "permutation"

        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

class BaseExplainer(ABC):
    """Abstract base class for all explainers."""

    def __init__(self, model: Any, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.explainer = None
        self.feature_names = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'BaseExplainer':
        """Fit the explainer to the data."""
        pass

    @abstractmethod
    def explain_instance(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain a single instance."""
        pass

    @abstractmethod
    def explain_batch(self, X: np.ndarray) -> Dict[str, Any]:
        """Explain a batch of instances."""
        pass

    def _validate_fitted(self):
        """Check if explainer is fitted."""
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted before use. Call fit() first.")

class SHAPExplainer(BaseExplainer):
    """SHAP-based explainer wrapper."""

    def __init__(self, model: Any, config: ExplainabilityConfig):
        super().__init__(model, config)
        self.explainer_type = config.shap_explainer_type
        self.background_data = None
        self.shap_values = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'SHAPExplainer':
        """Fit SHAP explainer to the data."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required but not installed")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Select background data
        if len(X) > self.config.shap_background_samples:
            indices = np.random.choice(len(X), self.config.shap_background_samples, replace=False)
            self.background_data = X[indices]
        else:
            self.background_data = X

        # Initialize appropriate SHAP explainer
        try:
            if self.explainer_type == "auto":
                self.explainer = self._get_auto_explainer(X)
            elif self.explainer_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
            elif self.explainer_type == "kernel":
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
            elif self.explainer_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
            elif self.explainer_type == "gradient":
                self.explainer = shap.GradientExplainer(self.model, self.background_data)
            else:
                raise ValueError(f"Unknown SHAP explainer type: {self.explainer_type}")

            self.is_fitted = True
            logger.info(f"SHAP {self.explainer_type} explainer fitted successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
            self.explainer_type = "kernel"
            self.is_fitted = True
            logger.info("Fallback to SHAP KernelExplainer")

        return self

    def _get_auto_explainer(self, X: np.ndarray):
        """Automatically select the best SHAP explainer."""
        # Check if model has tree-based methods
        if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
            return shap.TreeExplainer(self.model)

        # Check if model is linear
        if hasattr(self.model, 'coef_'):
            return shap.LinearExplainer(self.model, self.background_data)

        # Default to KernelExplainer
        return shap.KernelExplainer(self.model.predict, self.background_data)

    def explain_instance(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain a single instance using SHAP."""
        self._validate_fitted()

        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        try:
            shap_values = self.explainer.shap_values(instance)

            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for now

            return {
                'shap_values': shap_values[0] if shap_values.ndim > 1 else shap_values,
                'base_value': self.explainer.expected_value,
                'feature_names': self.feature_names,
                'instance_values': instance[0],
                'prediction': self.model.predict(instance)[0]
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return {'error': str(e)}

    def explain_batch(self, X: np.ndarray) -> Dict[str, Any]:
        """Explain a batch of instances using SHAP."""
        self._validate_fitted()

        try:
            # Process in batches to avoid memory issues
            batch_size = self.config.shap_batch_size
            all_shap_values = []

            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                batch_shap_values = self.explainer.shap_values(batch)

                # Handle multi-class case
                if isinstance(batch_shap_values, list):
                    batch_shap_values = batch_shap_values[0]

                all_shap_values.append(batch_shap_values)

            shap_values = np.vstack(all_shap_values)

            return {
                'shap_values': shap_values,
                'base_value': self.explainer.expected_value,
                'feature_names': self.feature_names,
                'instance_values': X,
                'predictions': self.model.predict(X)
            }

        except Exception as e:
            logger.error(f"SHAP batch explanation failed: {str(e)}")
            return {'error': str(e)}

class LIMEExplainer(BaseExplainer):
    """LIME-based explainer wrapper."""

    def __init__(self, model: Any, config: ExplainabilityConfig):
        super().__init__(model, config)
        self.mode = config.lime_mode
        self.training_data = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'LIMEExplainer':
        """Fit LIME explainer to the data."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required but not installed")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.training_data = X

        if self.mode == "tabular":
            self.explainer = lime_tabular.LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                discretize_continuous=self.config.lime_discretize_continuous,
                random_state=self.config.random_state
            )
        elif self.mode == "text":
            self.explainer = LimeTextExplainer()
        elif self.mode == "image":
            self.explainer = LimeImageExplainer()
        else:
            raise ValueError(f"Unknown LIME mode: {self.mode}")

        self.is_fitted = True
        logger.info(f"LIME {self.mode} explainer fitted successfully")
        return self

    def explain_instance(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain a single instance using LIME."""
        self._validate_fitted()

        if self.mode != "tabular":
            raise NotImplementedError(f"Instance explanation for {self.mode} mode not implemented")

        try:
            if instance.ndim == 1:
                instance_to_explain = instance
            else:
                instance_to_explain = instance[0]

            explanation = self.explainer.explain_instance(
                instance_to_explain,
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=self.config.lime_num_features,
                num_samples=self.config.lime_num_samples
            )

            # Extract feature importance
            feature_importance = dict(explanation.as_list())

            return {
                'lime_explanation': explanation,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'instance_values': instance_to_explain,
                'prediction': self.model.predict(instance_to_explain.reshape(1, -1))[0]
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            return {'error': str(e)}

    def explain_batch(self, X: np.ndarray) -> Dict[str, Any]:
        """Explain a batch of instances using LIME."""
        self._validate_fitted()

        explanations = []
        feature_importances = []

        for i, instance in enumerate(X):
            try:
                result = self.explain_instance(instance)
                if 'error' not in result:
                    explanations.append(result['lime_explanation'])
                    feature_importances.append(result['feature_importance'])
                else:
                    logger.warning(f"Failed to explain instance {i}: {result['error']}")

            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {str(e)}")

        return {
            'lime_explanations': explanations,
            'feature_importances': feature_importances,
            'feature_names': self.feature_names,
            'instance_values': X,
            'predictions': self.model.predict(X)
        }


class FeatureImportanceCalculator:
    """Calculate feature importance using various methods."""

    def __init__(self, model: Any, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.importance_scores = {}

    def calculate_shap_importance(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate feature importance using SHAP values."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP importance calculation")

        explainer = SHAPExplainer(self.model, self.config)
        explainer.fit(X, feature_names)

        result = explainer.explain_batch(X)
        if 'error' in result:
            logger.error(f"SHAP importance calculation failed: {result['error']}")
            return {}

        shap_values = result['shap_values']
        feature_names = result['feature_names']

        # Calculate mean absolute SHAP values
        importance = np.mean(np.abs(shap_values), axis=0)

        if self.config.normalize_importance:
            importance = importance / np.sum(importance)

        return dict(zip(feature_names, importance))

    def calculate_lime_importance(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate feature importance using LIME explanations."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIME importance calculation")

        explainer = LIMEExplainer(self.model, self.config)
        explainer.fit(X, feature_names)

        # Sample subset for LIME (computationally expensive)
        sample_size = min(100, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]

        result = explainer.explain_batch(X_sample)
        feature_importances = result['feature_importances']

        if not feature_importances:
            logger.error("No LIME explanations generated")
            return {}

        # Aggregate importance scores
        all_features = set()
        for imp_dict in feature_importances:
            all_features.update(imp_dict.keys())

        aggregated_importance = {}
        for feature in all_features:
            scores = [imp_dict.get(feature, 0) for imp_dict in feature_importances]
            aggregated_importance[feature] = np.mean(np.abs(scores))

        if self.config.normalize_importance:
            total = sum(aggregated_importance.values())
            if total > 0:
                aggregated_importance = {k: v/total for k, v in aggregated_importance.items()}

        return aggregated_importance

    def calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                       feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate permutation feature importance."""
        from sklearn.inspection import permutation_importance

        feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        try:
            perm_importance = permutation_importance(
                self.model, X, y, 
                n_repeats=10, 
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs if self.config.parallel_processing else 1
            )

            importance = perm_importance.importances_mean

            if self.config.normalize_importance:
                importance = importance / np.sum(importance)

            return dict(zip(feature_names, importance))

        except Exception as e:
            logger.error(f"Permutation importance calculation failed: {str(e)}")
            return {}

    def get_feature_importance(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                             feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance using the configured method."""
        method = self.config.importance_method

        if method == "shap":
            return self.calculate_shap_importance(X, feature_names)
        elif method == "lime":
            return self.calculate_lime_importance(X, feature_names)
        elif method == "permutation":
            if y is None:
                raise ValueError("Target values (y) required for permutation importance")
            return self.calculate_permutation_importance(X, y, feature_names)
        else:
            raise ValueError(f"Unknown importance method: {method}")

class ExplanationVisualizer:
    """Visualization utilities for model explanations."""

    def __init__(self, config: ExplainabilityConfig):
        self.config = config

    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                              title: str = "Feature Importance") -> go.Figure:
        """Create feature importance plot."""
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:self.config.top_k_features]

        features, importances = zip(*top_features)

        if self.config.plot_style == "plotly":
            fig = go.Figure(data=[
                go.Bar(
                    x=list(importances),
                    y=list(features),
                    orientation='h',
                    marker_color=['red' if imp < 0 else 'blue' for imp in importances]
                )
            ])

            fig.update_layout(
                title=title,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(top_features) * 25),
                showlegend=False
            )

            return fig
        else:
            # Matplotlib fallback
            plt.figure(figsize=self.config.figure_size)
            colors = ['red' if imp < 0 else 'blue' for imp in importances]
            plt.barh(range(len(features)), importances, color=colors)
            plt.yticks(range(len(features)), features)
            plt.xlabel("Importance Score")
            plt.title(title)
            plt.tight_layout()

            if self.config.save_plots:
                plt.savefig(f"{self.config.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')

            return plt.gcf()

    def plot_shap_summary(self, shap_values: np.ndarray, X: np.ndarray, 
                         feature_names: List[str]) -> go.Figure:
        """Create SHAP summary plot."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP summary plots")

        # Calculate feature importance
        importance = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(importance)[-self.config.top_k_features:]

        # Create violin plot data
        plot_data = []
        for i, idx in enumerate(sorted_idx):
            feature_name = feature_names[idx]
            shap_vals = shap_values[:, idx]
            feature_vals = X[:, idx]

            plot_data.append({
                'feature': feature_name,
                'shap_values': shap_vals,
                'feature_values': feature_vals,
                'y_pos': i
            })

        fig = go.Figure()

        for data in plot_data:
            fig.add_trace(go.Violin(
                y=[data['y_pos']] * len(data['shap_values']),
                x=data['shap_values'],
                name=data['feature'],
                orientation='h',
                showlegend=False,
                line_color='blue'
            ))

        fig.update_layout(
            title="SHAP Summary Plot",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(plot_data))),
                ticktext=[data['feature'] for data in plot_data]
            ),
            height=max(400, len(plot_data) * 40)
        )

        return fig

    def plot_shap_waterfall(self, shap_values: np.ndarray, base_value: float, 
                           feature_names: List[str], instance_values: np.ndarray,
                           prediction: float) -> go.Figure:
        """Create SHAP waterfall plot for a single instance."""
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:self.config.top_k_features]

        # Prepare data
        features = [feature_names[i] for i in sorted_idx]
        values = [shap_values[i] for i in sorted_idx]

        # Calculate cumulative values
        cumulative = [base_value]
        for val in values:
            cumulative.append(cumulative[-1] + val)

        # Create waterfall chart
        fig = go.Figure()

        # Base value
        fig.add_trace(go.Bar(
            x=['Base Value'],
            y=[base_value],
            name='Base Value',
            marker_color='gray'
        ))

        # Feature contributions
        for i, (feature, value) in enumerate(zip(features, values)):
            fig.add_trace(go.Bar(
                x=[feature],
                y=[value],
                name=feature,
                marker_color='red' if value < 0 else 'blue',
                showlegend=False
            ))

        # Final prediction
        fig.add_trace(go.Bar(
            x=['Prediction'],
            y=[prediction],
            name='Prediction',
            marker_color='green'
        ))

        fig.update_layout(
            title="SHAP Waterfall Plot",
            xaxis_title="Features",
            yaxis_title="Value",
            barmode='relative'
        )

        return fig

    def plot_lime_explanation(self, lime_explanation, title: str = "LIME Explanation") -> go.Figure:
        """Create LIME explanation plot."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIME explanation plots")

        # Extract feature importance from LIME explanation
        feature_importance = dict(lime_explanation.as_list())

        # Sort by absolute importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:self.config.top_k_features]

        features, importances = zip(*top_features)

        fig = go.Figure(data=[
            go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color=['red' if imp < 0 else 'blue' for imp in importances]
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="LIME Importance Score",
            yaxis_title="Features",
            height=max(400, len(top_features) * 25),
            showlegend=False
        )

        return fig

class InteractiveExplainer:
    """Interactive explanation interface."""

    def __init__(self, model: Any, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.shap_explainer = None
        self.lime_explainer = None
        self.visualizer = ExplanationVisualizer(config)
        self.importance_calculator = FeatureImportanceCalculator(model, config)

    def setup_explainers(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Initialize explainers with training data."""
        if SHAP_AVAILABLE:
            self.shap_explainer = SHAPExplainer(self.model, self.config)
            self.shap_explainer.fit(X, feature_names)

        if LIME_AVAILABLE:
            self.lime_explainer = LIMEExplainer(self.model, self.config)
            self.lime_explainer.fit(X, feature_names)

    def explain_prediction(self, instance: np.ndarray, 
                         explanation_type: str = "both") -> Dict[str, Any]:
        """Explain a single prediction with visualizations."""
        results = {}

        if explanation_type in ["shap", "both"] and self.shap_explainer:
            shap_result = self.shap_explainer.explain_instance(instance)
            if 'error' not in shap_result:
                results['shap'] = shap_result

                # Create SHAP waterfall plot
                waterfall_fig = self.visualizer.plot_shap_waterfall(
                    shap_result['shap_values'],
                    shap_result['base_value'],
                    shap_result['feature_names'],
                    shap_result['instance_values'],
                    shap_result['prediction']
                )
                results['shap_waterfall'] = waterfall_fig

        if explanation_type in ["lime", "both"] and self.lime_explainer:
            lime_result = self.lime_explainer.explain_instance(instance)
            if 'error' not in lime_result:
                results['lime'] = lime_result

                # Create LIME plot
                lime_fig = self.visualizer.plot_lime_explanation(
                    lime_result['lime_explanation']
                )
                results['lime_plot'] = lime_fig

        return results

    def create_explanation_dashboard(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """Create comprehensive explanation dashboard."""
        dashboard = {}

        # Feature importance plot
        importance_scores = self.importance_calculator.get_feature_importance(X, y, feature_names)
        if importance_scores:
            importance_fig = self.visualizer.plot_feature_importance(
                importance_scores, 
                f"Feature Importance ({self.config.importance_method.upper()})"
            )
            dashboard['feature_importance'] = importance_fig

        # SHAP summary plot
        if self.shap_explainer:
            shap_result = self.shap_explainer.explain_batch(X[:min(100, len(X))])  # Sample for performance
            if 'error' not in shap_result:
                summary_fig = self.visualizer.plot_shap_summary(
                    shap_result['shap_values'],
                    shap_result['instance_values'],
                    shap_result['feature_names']
                )
                dashboard['shap_summary'] = summary_fig

        return dashboard

    def save_dashboard(self, dashboard: Dict[str, go.Figure], prefix: str = "explanation"):
        """Save dashboard plots to files."""
        if not self.config.save_plots:
            return

        for plot_name, fig in dashboard.items():
            filename = f"{self.config.output_dir}/{prefix}_{plot_name}.html"
            fig.write_html(filename)
            logger.info(f"Saved {plot_name} plot to {filename}")

def create_explainer_pipeline(model: Any, config: Optional[ExplainabilityConfig] = None) -> InteractiveExplainer:
    """Factory function to create a complete explainer pipeline."""
    if config is None:
        config = ExplainabilityConfig()

    return InteractiveExplainer(model, config)

def explain_model_predictions(model: Any, X: np.ndarray, y: Optional[np.ndarray] = None,
                            feature_names: Optional[List[str]] = None,
                            config: Optional[ExplainabilityConfig] = None) -> Dict[str, Any]:
    """High-level function to explain model predictions."""
    if config is None:
        config = ExplainabilityConfig()

    explainer = create_explainer_pipeline(model, config)
    explainer.setup_explainers(X, feature_names)

    # Create dashboard
    dashboard = explainer.create_explanation_dashboard(X, y, feature_names)

    # Save plots if configured
    explainer.save_dashboard(dashboard)

    return {
        'explainer': explainer,
        'dashboard': dashboard,
        'config': config
    }


class ModelInterpretationDashboard:
    """Comprehensive model interpretation dashboard."""

    def __init__(self, model: Any, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.explainer = InteractiveExplainer(model, config)
        self.visualizer = ExplanationVisualizer(config)
        self.dashboard_data = {}

    def setup_dashboard(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       feature_names: Optional[List[str]] = None):
        """Initialize dashboard with data."""
        self.explainer.setup_explainers(X, feature_names)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.X = X
        self.y = y

    def create_model_overview(self) -> Dict[str, Any]:
        """Create model overview section."""
        overview = {
            'model_type': type(self.model).__name__,
            'n_features': self.X.shape[1] if hasattr(self, 'X') else 0,
            'n_samples': self.X.shape[0] if hasattr(self, 'X') else 0,
            'feature_names': self.feature_names if hasattr(self, 'feature_names') else [],
            'explainer_config': {
                'shap_available': SHAP_AVAILABLE,
                'lime_available': LIME_AVAILABLE,
                'importance_method': self.config.importance_method
            }
        }
        return overview

    def create_global_explanations(self) -> Dict[str, go.Figure]:
        """Create global model explanations."""
        global_plots = {}

        if not hasattr(self, 'X'):
            logger.warning("Dashboard not initialized with data")
            return global_plots

        # Feature importance
        importance_calc = FeatureImportanceCalculator(self.model, self.config)
        importance_scores = importance_calc.get_feature_importance(self.X, self.y, self.feature_names)

        if importance_scores:
            global_plots['feature_importance'] = self.visualizer.plot_feature_importance(
                importance_scores, "Global Feature Importance"
            )

        # SHAP summary if available
        if SHAP_AVAILABLE and self.explainer.shap_explainer:
            sample_size = min(200, len(self.X))
            sample_idx = np.random.choice(len(self.X), sample_size, replace=False)
            X_sample = self.X[sample_idx]

            shap_result = self.explainer.shap_explainer.explain_batch(X_sample)
            if 'error' not in shap_result:
                global_plots['shap_summary'] = self.visualizer.plot_shap_summary(
                    shap_result['shap_values'],
                    shap_result['instance_values'],
                    shap_result['feature_names']
                )

        return global_plots

    def create_local_explanation(self, instance_idx: int) -> Dict[str, Any]:
        """Create local explanation for specific instance."""
        if not hasattr(self, 'X'):
            return {'error': 'Dashboard not initialized'}

        if instance_idx >= len(self.X):
            return {'error': 'Instance index out of range'}

        instance = self.X[instance_idx]
        return self.explainer.explain_prediction(instance)

    def create_cohort_analysis(self, cohort_column: str, cohort_values: List[Any]) -> Dict[str, go.Figure]:
        """Create cohort-based explanation analysis."""
        if not hasattr(self, 'X'):
            return {}

        cohort_plots = {}

        # Assume cohort_column is an index in feature_names
        try:
            cohort_idx = self.feature_names.index(cohort_column)
        except ValueError:
            logger.error(f"Cohort column '{cohort_column}' not found in features")
            return cohort_plots

        for cohort_value in cohort_values:
            # Filter data for this cohort
            cohort_mask = self.X[:, cohort_idx] == cohort_value
            if not np.any(cohort_mask):
                continue

            X_cohort = self.X[cohort_mask]
            y_cohort = self.y[cohort_mask] if self.y is not None else None

            # Calculate feature importance for this cohort
            importance_calc = FeatureImportanceCalculator(self.model, self.config)
            cohort_importance = importance_calc.get_feature_importance(X_cohort, y_cohort, self.feature_names)

            if cohort_importance:
                cohort_plots[f'cohort_{cohort_value}'] = self.visualizer.plot_feature_importance(
                    cohort_importance, f"Feature Importance - {cohort_column}={cohort_value}"
                )

        return cohort_plots

class CreditExplanationUtilities:
    """Credit-specific explanation utilities."""

    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.credit_features = {
            'financial': ['income', 'debt_to_income', 'credit_utilization', 'payment_history'],
            'demographic': ['age', 'employment_length', 'home_ownership'],
            'credit_history': ['credit_length', 'num_accounts', 'recent_inquiries'],
            'loan_specific': ['loan_amount', 'loan_purpose', 'interest_rate']
        }

    def categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features into credit-relevant groups."""
        categorized = {category: [] for category in self.credit_features.keys()}
        categorized['other'] = []

        for feature in feature_names:
            feature_lower = feature.lower()
            assigned = False

            for category, keywords in self.credit_features.items():
                if any(keyword in feature_lower for keyword in keywords):
                    categorized[category].append(feature)
                    assigned = True
                    break

            if not assigned:
                categorized['other'].append(feature)

        return categorized

    def create_credit_risk_explanation(self, shap_values: np.ndarray, feature_names: List[str],
                                     instance_values: np.ndarray, prediction: float) -> Dict[str, Any]:
        """Create credit risk-specific explanation."""
        categorized_features = self.categorize_features(feature_names)

        # Calculate category-wise contributions
        category_contributions = {}
        for category, features in categorized_features.items():
            if not features:
                continue

            feature_indices = [feature_names.index(f) for f in features if f in feature_names]
            if feature_indices:
                category_shap = shap_values[feature_indices]
                category_contributions[category] = {
                    'total_contribution': np.sum(category_shap),
                    'features': {
                        features[i]: {
                            'shap_value': category_shap[i],
                            'feature_value': instance_values[feature_indices[i]]
                        }
                        for i in range(len(feature_indices))
                    }
                }

        # Risk assessment
        risk_level = self._assess_risk_level(prediction)
        key_risk_factors = self._identify_key_risk_factors(shap_values, feature_names)

        return {
            'prediction': prediction,
            'risk_level': risk_level,
            'category_contributions': category_contributions,
            'key_risk_factors': key_risk_factors,
            'explanation_summary': self._generate_explanation_summary(
                category_contributions, risk_level, key_risk_factors
            )
        }

    def _assess_risk_level(self, prediction: float) -> str:
        """Assess risk level based on prediction."""
        if prediction < 0.3:
            return "Low Risk"
        elif prediction < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

    def _identify_key_risk_factors(self, shap_values: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Identify key risk factors."""
        # Get top contributing features (positive SHAP values increase risk)
        risk_factors = []
        sorted_indices = np.argsort(shap_values)[::-1]

        for i in sorted_indices[:5]:  # Top 5 risk factors
            if shap_values[i] > 0:  # Only positive contributions (increasing risk)
                risk_factors.append({
                    'feature': feature_names[i],
                    'contribution': shap_values[i],
                    'impact': 'Increases Risk' if shap_values[i] > 0 else 'Decreases Risk'
                })

        return risk_factors

    def _generate_explanation_summary(self, category_contributions: Dict[str, Any],
                                    risk_level: str, key_risk_factors: List[Dict[str, Any]]) -> str:
        """Generate human-readable explanation summary."""
        summary_parts = [f"Risk Assessment: {risk_level}"]

        # Category analysis
        if category_contributions:
            top_category = max(category_contributions.items(), 
                             key=lambda x: abs(x[1]['total_contribution']))
            summary_parts.append(
                f"Primary risk driver: {top_category[0].replace('_', ' ').title()} factors"
            )

        # Key factors
        if key_risk_factors:
            top_factor = key_risk_factors[0]
            summary_parts.append(
                f"Most significant factor: {top_factor['feature']} ({top_factor['impact']})"
            )

        return ". ".join(summary_parts) + "."

class StreamlitExplanationComponents:
    """Interactive components for Streamlit dashboard."""

    def __init__(self, dashboard: ModelInterpretationDashboard):
        self.dashboard = dashboard
        self.credit_utils = CreditExplanationUtilities(dashboard.config)

    def render_model_overview(self):
        """Render model overview section."""
        try:
            import streamlit as st

            overview = self.dashboard.create_model_overview()

            st.header("Model Overview")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Model Type", overview['model_type'])
            with col2:
                st.metric("Features", overview['n_features'])
            with col3:
                st.metric("Training Samples", overview['n_samples'])

            st.subheader("Explainability Configuration")
            config_df = pd.DataFrame([
                {"Component": "SHAP", "Available": overview['explainer_config']['shap_available']},
                {"Component": "LIME", "Available": overview['explainer_config']['lime_available']},
                {"Component": "Importance Method", "Available": overview['explainer_config']['importance_method']}
            ])
            st.dataframe(config_df)

        except ImportError:
            logger.warning("Streamlit not available for interactive components")

    def render_global_explanations(self):
        """Render global explanation plots."""
        try:
            import streamlit as st

            st.header("Global Model Explanations")
            global_plots = self.dashboard.create_global_explanations()

            for plot_name, fig in global_plots.items():
                st.subheader(plot_name.replace('_', ' ').title())
                st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            logger.warning("Streamlit not available for interactive components")

    def render_instance_explanation(self):
        """Render interactive instance explanation."""
        try:
            import streamlit as st

            st.header("Individual Prediction Explanation")

            if hasattr(self.dashboard, 'X'):
                instance_idx = st.slider(
                    "Select Instance", 
                    0, len(self.dashboard.X) - 1, 0
                )

                explanation = self.dashboard.create_local_explanation(instance_idx)

                if 'error' in explanation:
                    st.error(explanation['error'])
                    return

                # Display prediction
                if 'shap' in explanation:
                    prediction = explanation['shap']['prediction']
                    st.metric("Prediction", f"{prediction:.3f}")

                # Display SHAP explanation
                if 'shap_waterfall' in explanation:
                    st.subheader("SHAP Waterfall Plot")
                    st.plotly_chart(explanation['shap_waterfall'], use_container_width=True)

                # Display LIME explanation
                if 'lime_plot' in explanation:
                    st.subheader("LIME Explanation")
                    st.plotly_chart(explanation['lime_plot'], use_container_width=True)

                # Credit-specific explanation
                if 'shap' in explanation:
                    shap_data = explanation['shap']
                    credit_explanation = self.credit_utils.create_credit_risk_explanation(
                        shap_data['shap_values'],
                        shap_data['feature_names'],
                        shap_data['instance_values'],
                        shap_data['prediction']
                    )

                    st.subheader("Credit Risk Analysis")
                    st.write(credit_explanation['explanation_summary'])

                    # Risk factors table
                    if credit_explanation['key_risk_factors']:
                        risk_df = pd.DataFrame(credit_explanation['key_risk_factors'])
                        st.dataframe(risk_df)
            else:
                st.warning("Dashboard not initialized with data")

        except ImportError:
            logger.warning("Streamlit not available for interactive components")

class AdvancedExplanationFeatures:
    """Advanced explanation features and utilities."""

    def __init__(self, model: Any, config: ExplainabilityConfig):
        self.model = model
        self.config = config

    def counterfactual_explanation(self, instance: np.ndarray, feature_names: List[str],
                                 target_prediction: float = 0.5) -> Dict[str, Any]:
        """Generate counterfactual explanations."""
        try:
            from sklearn.neighbors import NearestNeighbors

            # Simple counterfactual: find nearest instance with different prediction
            current_pred = self.model.predict(instance.reshape(1, -1))[0]

            # This is a simplified approach - in practice, you'd use more sophisticated methods
            # like DICE (Diverse Counterfactual Explanations)

            return {
                'original_prediction': current_pred,
                'target_prediction': target_prediction,
                'counterfactual_found': False,
                'message': 'Counterfactual generation requires specialized libraries like DICE'
            }

        except Exception as e:
            logger.error(f"Counterfactual explanation failed: {str(e)}")
            return {'error': str(e)}

    def stability_analysis(self, instance: np.ndarray, feature_names: List[str],
                          noise_level: float = 0.1, n_samples: int = 100) -> Dict[str, Any]:
        """Analyze explanation stability with noise."""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP required for stability analysis'}

        try:
            # Create SHAP explainer
            explainer = SHAPExplainer(self.model, self.config)
            explainer.fit(instance.reshape(1, -1), feature_names)

            # Generate noisy versions
            noise = np.random.normal(0, noise_level, (n_samples, len(instance)))
            noisy_instances = instance + noise

            # Get explanations for all noisy instances
            explanations = []
            for noisy_instance in noisy_instances:
                result = explainer.explain_instance(noisy_instance)
                if 'error' not in result:
                    explanations.append(result['shap_values'])

            if not explanations:
                return {'error': 'No valid explanations generated'}

            explanations = np.array(explanations)

            # Calculate stability metrics
            mean_explanation = np.mean(explanations, axis=0)
            std_explanation = np.std(explanations, axis=0)
            stability_score = 1 - np.mean(std_explanation / (np.abs(mean_explanation) + 1e-8))

            return {
                'stability_score': stability_score,
                'mean_explanation': mean_explanation,
                'std_explanation': std_explanation,
                'feature_names': feature_names,
                'n_samples': len(explanations)
            }

        except Exception as e:
            logger.error(f"Stability analysis failed: {str(e)}")
            return {'error': str(e)}

    def feature_interaction_analysis(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature interactions using SHAP interaction values."""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP required for interaction analysis'}

        try:
            explainer = SHAPExplainer(self.model, self.config)
            explainer.fit(X, feature_names)

            # Get interaction values (if supported by explainer)
            if hasattr(explainer.explainer, 'shap_interaction_values'):
                sample_size = min(100, len(X))
                sample_idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[sample_idx]

                interaction_values = explainer.explainer.shap_interaction_values(X_sample)

                # Calculate mean interaction strengths
                mean_interactions = np.mean(np.abs(interaction_values), axis=0)

                # Find top interactions
                n_features = len(feature_names)
                top_interactions = []

                for i in range(n_features):
                    for j in range(i+1, n_features):
                        interaction_strength = mean_interactions[i, j]
                        top_interactions.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'interaction_strength': interaction_strength
                        })

                # Sort by interaction strength
                top_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)

                return {
                    'interaction_matrix': mean_interactions,
                    'top_interactions': top_interactions[:10],
                    'feature_names': feature_names
                }
            else:
                return {'error': 'Explainer does not support interaction values'}

        except Exception as e:
            logger.error(f"Interaction analysis failed: {str(e)}")
            return {'error': str(e)}

# Utility functions for easy integration
def create_credit_explanation_dashboard(model: Any, X: np.ndarray, y: Optional[np.ndarray] = None,
                                      feature_names: Optional[List[str]] = None) -> ModelInterpretationDashboard:
    """Create a complete credit explanation dashboard."""
    config = ExplainabilityConfig(
        importance_method="shap",
        top_k_features=15,
        save_plots=True
    )

    dashboard = ModelInterpretationDashboard(model, config)
    dashboard.setup_dashboard(X, y, feature_names)

    return dashboard

def generate_model_explanation_report(model: Any, X: np.ndarray, y: Optional[np.ndarray] = None,
                                    feature_names: Optional[List[str]] = None,
                                    output_path: str = "/home/user/output/") -> str:
    """Generate comprehensive model explanation report."""
    dashboard = create_credit_explanation_dashboard(model, X, y, feature_names)

    # Create all visualizations
    global_plots = dashboard.create_global_explanations()

    # Save plots
    report_files = []
    for plot_name, fig in global_plots.items():
        filename = f"{output_path}/explanation_{plot_name}.html"
        fig.write_html(filename)
        report_files.append(filename)

    # Create summary report
    overview = dashboard.create_model_overview()
    summary_path = f"{output_path}/explanation_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("Model Explanation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Type: {overview['model_type']}\n")
        f.write(f"Number of Features: {overview['n_features']}\n")
        f.write(f"Number of Samples: {overview['n_samples']}\n")
        f.write(f"Explainability Method: {dashboard.config.importance_method}\n\n")
        f.write("Generated Visualizations:\n")
        for filename in report_files:
            f.write(f"- {filename}\n")

    return summary_path
