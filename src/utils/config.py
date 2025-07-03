"""
Configuration Management Module

This module provides centralized configuration management for the Credit Default Prediction application.
It handles loading configuration from YAML files, environment variables, and provides easy access
to configuration parameters throughout the application.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime

# Configure basic logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration dataclass"""
    name: str = "Credit Default Prediction System"
    version: str = "1.0.0"
    description: str = "Advanced ML-powered credit risk assessment platform"
    author: str = "Credit Risk Analytics Team"


@dataclass
class StreamlitConfig:
    """Streamlit configuration dataclass"""
    page_title: str = "Credit Default Prediction"
    page_icon: str = "💳"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


@dataclass
class DataConfig:
    """Data configuration dataclass"""
    dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    dataset_name: str = "default of credit card clients.xls"
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    models_path: str = "data/models/"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    feature_selection_threshold: float = 0.01
    correlation_threshold: float = 0.95


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    available_models: list = field(default_factory=lambda: [
        "RandomForest", "XGBoost", "LightGBM", "CatBoost", "LogisticRegression"
    ])
    default_model: str = "XGBoost"
    random_state: int = 42


@dataclass
class VisualizationConfig:
    """Visualization configuration dataclass"""
    colors: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ff7f0e",
        "info": "#17a2b8"
    })
    figure_size: list = field(default_factory=lambda: [12, 8])
    dpi: int = 100
    style: str = "whitegrid"
    max_features_plot: int = 20
    correlation_threshold_plot: float = 0.3


class ConfigManager:
    """
    Central configuration manager for the application.

    This class handles loading configuration from YAML files, environment variables,
    and provides easy access to configuration parameters.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path or self._find_config_file()
        self._config_data = {}
        self._load_config()

        # Initialize configuration dataclasses
        self.app = self._create_app_config()
        self.streamlit = self._create_streamlit_config()
        self.data = self._create_data_config()
        self.models = self._create_model_config()
        self.visualization = self._create_visualization_config()

    def _find_config_file(self) -> str:
        """Find the configuration file in common locations"""
        possible_paths = [
            "config.yaml",
            "config.yml",
            "../config.yaml",
            "../../config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # If no config file found, create a default one
        logger.warning("No config file found. Using default configuration.")
        return None

    def _load_config(self):
        """Load configuration from YAML file"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    self._config_data = yaml.safe_load(file) or {}
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self._config_data = {}
        else:
            logger.warning("Using default configuration")
            self._config_data = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if no config file is found"""
        return {
            'app': {
                'name': 'Credit Default Prediction System',
                'version': '1.0.0',
                'description': 'Advanced ML-powered credit risk assessment platform',
                'author': 'Credit Risk Analytics Team'
            },
            'streamlit': {
                'page_title': 'Credit Default Prediction',
                'page_icon': '💳',
                'layout': 'wide',
                'initial_sidebar_state': 'expanded'
            },
            'data': {
                'dataset_url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
                'dataset_name': 'default of credit card clients.xls',
                'raw_data_path': 'data/raw/',
                'processed_data_path': 'data/processed/',
                'models_path': 'data/models/',
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42
            },
            'models': {
                'available_models': ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'LogisticRegression'],
                'default_model': 'XGBoost',
                'random_state': 42
            }
        }

    def _create_app_config(self) -> AppConfig:
        """Create application configuration"""
        app_config = self._config_data.get('app', {})
        return AppConfig(
            name=app_config.get('name', 'Credit Default Prediction System'),
            version=app_config.get('version', '1.0.0'),
            description=app_config.get('description', 'Advanced ML-powered credit risk assessment platform'),
            author=app_config.get('author', 'Credit Risk Analytics Team')
        )

    def _create_streamlit_config(self) -> StreamlitConfig:
        """Create Streamlit configuration"""
        streamlit_config = self._config_data.get('streamlit', {})
        return StreamlitConfig(
            page_title=streamlit_config.get('page_title', 'Credit Default Prediction'),
            page_icon=streamlit_config.get('page_icon', '💳'),
            layout=streamlit_config.get('layout', 'wide'),
            initial_sidebar_state=streamlit_config.get('initial_sidebar_state', 'expanded')
        )

    def _create_data_config(self) -> DataConfig:
        """Create data configuration"""
        data_config = self._config_data.get('data', {})
        return DataConfig(
            dataset_url=data_config.get('dataset_url', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'),
            dataset_name=data_config.get('dataset_name', 'default of credit card clients.xls'),
            raw_data_path=data_config.get('raw_data_path', 'data/raw/'),
            processed_data_path=data_config.get('processed_data_path', 'data/processed/'),
            models_path=data_config.get('models_path', 'data/models/'),
            test_size=data_config.get('test_size', 0.2),
            validation_size=data_config.get('validation_size', 0.2),
            random_state=data_config.get('random_state', 42),
            feature_selection_threshold=data_config.get('feature_selection_threshold', 0.01),
            correlation_threshold=data_config.get('correlation_threshold', 0.95)
        )

    def _create_model_config(self) -> ModelConfig:
        """Create model configuration"""
        model_config = self._config_data.get('models', {})
        return ModelConfig(
            available_models=model_config.get('available_models', ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'LogisticRegression']),
            default_model=model_config.get('default_model', 'XGBoost'),
            random_state=model_config.get('random_state', 42)
        )

    def _create_visualization_config(self) -> VisualizationConfig:
        """Create visualization configuration"""
        viz_config = self._config_data.get('visualization', {})
        return VisualizationConfig(
            colors=viz_config.get('colors', {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "success": "#2ca02c", 
                "danger": "#d62728",
                "warning": "#ff7f0e",
                "info": "#17a2b8"
            }),
            figure_size=viz_config.get('figure_size', [12, 8]),
            dpi=viz_config.get('dpi', 100),
            style=viz_config.get('style', 'whitegrid'),
            max_features_plot=viz_config.get('max_features_plot', 20),
            correlation_threshold_plot=viz_config.get('correlation_threshold_plot', 0.3)
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation.

        Args:
            key: Configuration key (e.g., 'app.name', 'data.test_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value by key using dot notation.

        Args:
            key: Configuration key (e.g., 'app.name', 'data.test_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config_data

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get model parameters for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of model parameters
        """
        model_key = model_name.lower().replace(' ', '_')
        return self._config_data.get('models', {}).get(model_key, {})

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration"""
        return self._config_data.get('features', {})

    def get_business_config(self) -> Dict[str, Any]:
        """Get business intelligence configuration"""
        return self._config_data.get('business_intelligence', {})

    def get_compliance_config(self) -> Dict[str, Any]:
        """Get compliance configuration"""
        return self._config_data.get('compliance', {})

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return self._config_data.get('ui', {})

    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save the configuration file
        """
        output_path = output_path or self.config_path or 'config.yaml'

        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self._config_data, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()

        # Reinitialize configuration dataclasses
        self.app = self._create_app_config()
        self.streamlit = self._create_streamlit_config()
        self.data = self._create_data_config()
        self.models = self._create_model_config()
        self.visualization = self._create_visualization_config()

        logger.info("Configuration reloaded")

    def validate_config(self) -> bool:
        """
        Validate the configuration for required fields and correct types.

        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['app', 'data', 'models']

        for section in required_sections:
            if section not in self._config_data:
                logger.error(f"Missing required configuration section: {section}")
                return False

        # Validate data types
        try:
            if not isinstance(self.data.test_size, (int, float)) or not 0 < self.data.test_size < 1:
                logger.error("Invalid test_size: must be a float between 0 and 1")
                return False

            if not isinstance(self.data.random_state, int):
                logger.error("Invalid random_state: must be an integer")
                return False

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

        logger.info("Configuration validation passed")
        return True

    def get_environment_override(self, key: str) -> Optional[str]:
        """
        Get environment variable override for configuration key.

        Args:
            key: Configuration key

        Returns:
            Environment variable value if exists, None otherwise
        """
        env_key = f"CREDIT_DEFAULT_{key.upper().replace('.', '_')}"
        return os.getenv(env_key)

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(config_path='{self.config_path}', sections={list(self._config_data.keys())})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """
    Get the global configuration instance.

    Returns:
        ConfigManager instance
    """
    return config


def reload_config():
    """Reload the global configuration"""
    global config
    config.reload_config()


# Utility functions for common configuration access
def get_data_paths() -> Dict[str, str]:
    """Get all data paths from configuration"""
    return {
        'raw': config.data.raw_data_path,
        'processed': config.data.processed_data_path,
        'models': config.data.models_path
    }


def get_model_list() -> list:
    """Get list of available models"""
    return config.models.available_models


def get_default_model() -> str:
    """Get default model name"""
    return config.models.default_model


def get_random_state() -> int:
    """Get random state for reproducibility"""
    return config.data.random_state


def get_test_size() -> float:
    """Get test size for train-test split"""
    return config.data.test_size


if __name__ == "__main__":
    # Example usage and testing
    print("Configuration Manager Test")
    print("=" * 50)

    # Initialize configuration
    config_manager = ConfigManager()

    # Test configuration access
    print(f"App Name: {config_manager.app.name}")
    print(f"App Version: {config_manager.app.version}")
    print(f"Default Model: {config_manager.models.default_model}")
    print(f"Test Size: {config_manager.data.test_size}")

    # Test dot notation access
    print(f"\nUsing dot notation:")
    print(f"App Name: {config_manager.get('app.name')}")
    print(f"Data Test Size: {config_manager.get('data.test_size')}")

    # Test model parameters
    print(f"\nModel Parameters:")
    xgb_params = config_manager.get_model_params('XGBoost')
    print(f"XGBoost params: {xgb_params}")

    # Validate configuration
    is_valid = config_manager.validate_config()
    print(f"\nConfiguration valid: {is_valid}")

    print(f"\nConfiguration: {config_manager}")
