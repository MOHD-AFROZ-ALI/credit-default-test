"""
Application Settings

This module provides environment-specific configuration settings,
feature flags, and application-specific parameters for the
Credit Default Prediction system.
"""

import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging


class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseSettings:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    name: str = "credit_default"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class ModelSettings:
    """Machine learning model settings."""
    # Model selection
    default_model: str = "xgboost"
    available_models: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest", "logistic_regression"
    ])

    # Training parameters
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5

    # Performance thresholds
    min_auc_score: float = 0.75
    min_precision: float = 0.70
    min_recall: float = 0.65

    # Model persistence
    model_storage_path: str = "models/"
    model_registry_enabled: bool = True
    auto_model_versioning: bool = True

    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    max_tuning_iterations: int = 100
    tuning_timeout_minutes: int = 60


@dataclass
class DataSettings:
    """Data processing and storage settings."""
    # Paths
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    feature_store_path: str = "data/features/"
    backup_path: str = "data/backups/"

    # File formats
    input_format: str = "csv"
    output_format: str = "parquet"
    compression: str = "snappy"

    # Data quality
    max_missing_percentage: float = 0.3
    outlier_detection_enabled: bool = True
    data_validation_enabled: bool = True

    # Feature engineering
    auto_feature_engineering: bool = True
    feature_selection_enabled: bool = True
    feature_importance_threshold: float = 0.01

    # Data versioning
    enable_data_versioning: bool = True
    max_data_versions: int = 10


@dataclass
class APISettings:
    """API server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 4

    # Security
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100

    # Request handling
    max_request_size: int = 16777216  # 16MB
    request_timeout: int = 30

    # Authentication
    enable_authentication: bool = False
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_expiration_hours: int = 24


@dataclass
class MonitoringSettings:
    """Monitoring and observability settings."""
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Health checks
    enable_health_checks: bool = True
    health_check_interval: int = 30

    # Alerting
    enable_alerting: bool = False
    alert_email: str = ""
    alert_webhook_url: str = ""

    # Performance monitoring
    enable_performance_monitoring: bool = True
    slow_query_threshold: float = 1.0
    memory_usage_threshold: float = 0.8


@dataclass
class FeatureFlags:
    """Feature flags for enabling/disabling functionality."""
    # Model features
    enable_ensemble_models: bool = True
    enable_deep_learning: bool = False
    enable_auto_ml: bool = False

    # Data features
    enable_real_time_features: bool = False
    enable_streaming_data: bool = False
    enable_data_lineage: bool = True

    # API features
    enable_batch_predictions: bool = True
    enable_model_explanations: bool = True
    enable_a_b_testing: bool = False

    # Experimental features
    enable_experimental_features: bool = False
    enable_beta_features: bool = False


@dataclass
class CacheSettings:
    """Caching configuration."""
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Cache behavior
    enable_caching: bool = True
    default_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000

    # Cache keys
    model_cache_ttl: int = 86400  # 24 hours
    feature_cache_ttl: int = 1800  # 30 minutes
    prediction_cache_ttl: int = 300  # 5 minutes


class ApplicationSettings:
    """Main application settings manager."""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        """
        Initialize application settings.

        Args:
            environment: Application environment
        """
        self.environment = environment
        self._load_settings()

    def _load_settings(self) -> None:
        """Load environment-specific settings."""
        if self.environment == Environment.DEVELOPMENT:
            self._load_development_settings()
        elif self.environment == Environment.TESTING:
            self._load_testing_settings()
        elif self.environment == Environment.STAGING:
            self._load_staging_settings()
        elif self.environment == Environment.PRODUCTION:
            self._load_production_settings()

        # Override with environment variables
        self._load_from_environment()

    def _load_development_settings(self) -> None:
        """Load development environment settings."""
        self.database = DatabaseSettings(
            host="localhost",
            name="credit_default_dev",
            echo=True
        )

        self.model = ModelSettings(
            enable_hyperparameter_tuning=False,
            max_tuning_iterations=10
        )

        self.data = DataSettings(
            enable_data_versioning=False
        )

        self.api = APISettings(
            debug=True,
            reload=True,
            workers=1,
            enable_authentication=False
        )

        self.monitoring = MonitoringSettings(
            enable_alerting=False
        )

        self.feature_flags = FeatureFlags(
            enable_experimental_features=True,
            enable_beta_features=True
        )

        self.cache = CacheSettings(
            enable_caching=False
        )

        self.logging_level = LogLevel.DEBUG

    def _load_testing_settings(self) -> None:
        """Load testing environment settings."""
        self.database = DatabaseSettings(
            host="localhost",
            name="credit_default_test",
            echo=False
        )

        self.model = ModelSettings(
            enable_hyperparameter_tuning=False,
            max_tuning_iterations=5
        )

        self.data = DataSettings(
            enable_data_versioning=False
        )

        self.api = APISettings(
            debug=False,
            workers=1,
            enable_authentication=False
        )

        self.monitoring = MonitoringSettings(
            enable_alerting=False,
            enable_performance_monitoring=False
        )

        self.feature_flags = FeatureFlags(
            enable_experimental_features=False,
            enable_beta_features=False
        )

        self.cache = CacheSettings(
            enable_caching=False
        )

        self.logging_level = LogLevel.INFO

    def _load_staging_settings(self) -> None:
        """Load staging environment settings."""
        self.database = DatabaseSettings(
            host=os.getenv("DB_HOST", "staging-db"),
            name="credit_default_staging"
        )

        self.model = ModelSettings()

        self.data = DataSettings()

        self.api = APISettings(
            debug=False,
            workers=2,
            enable_authentication=True
        )

        self.monitoring = MonitoringSettings(
            enable_alerting=True
        )

        self.feature_flags = FeatureFlags(
            enable_experimental_features=False,
            enable_beta_features=True
        )

        self.cache = CacheSettings()

        self.logging_level = LogLevel.INFO

    def _load_production_settings(self) -> None:
        """Load production environment settings."""
        self.database = DatabaseSettings(
            host=os.getenv("DB_HOST", "prod-db"),
            name="credit_default_prod",
            pool_size=20,
            max_overflow=40
        )

        self.model = ModelSettings(
            min_auc_score=0.80,
            min_precision=0.75,
            min_recall=0.70
        )

        self.data = DataSettings()

        self.api = APISettings(
            debug=False,
            workers=8,
            enable_authentication=True,
            rate_limit_per_minute=1000
        )

        self.monitoring = MonitoringSettings(
            enable_alerting=True,
            enable_performance_monitoring=True
        )

        self.feature_flags = FeatureFlags(
            enable_experimental_features=False,
            enable_beta_features=False
        )

        self.cache = CacheSettings(
            default_ttl=7200,  # 2 hours
            max_cache_size=10000
        )

        self.logging_level = LogLevel.WARNING

    def _load_from_environment(self) -> None:
        """Override settings with environment variables."""
        # Database overrides
        if os.getenv("DB_HOST"):
            self.database.host = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"):
            self.database.port = int(os.getenv("DB_PORT"))
        if os.getenv("DB_NAME"):
            self.database.name = os.getenv("DB_NAME")
        if os.getenv("DB_USER"):
            self.database.username = os.getenv("DB_USER")
        if os.getenv("DB_PASSWORD"):
            self.database.password = os.getenv("DB_PASSWORD")

        # API overrides
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        if os.getenv("API_WORKERS"):
            self.api.workers = int(os.getenv("API_WORKERS"))

        # Logging override
        if os.getenv("LOG_LEVEL"):
            try:
                self.logging_level = LogLevel(os.getenv("LOG_LEVEL").upper())
            except ValueError:
                pass

    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self.database.url

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            feature_name: Name of the feature flag

        Returns:
            True if feature is enabled
        """
        return getattr(self.feature_flags, feature_name, False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'environment': self.environment.value,
            'database': self.database.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__,
            'feature_flags': self.feature_flags.__dict__,
            'cache': self.cache.__dict__,
            'logging_level': self.logging_level.value
        }


# Global settings instance
_settings = None


def get_settings(environment: Optional[Environment] = None) -> ApplicationSettings:
    """
    Get global settings instance.

    Args:
        environment: Application environment (optional)

    Returns:
        Application settings instance
    """
    global _settings

    if _settings is None:
        env = environment or Environment(os.getenv("APP_ENV", "development"))
        _settings = ApplicationSettings(env)

    return _settings


def reload_settings(environment: Optional[Environment] = None) -> ApplicationSettings:
    """
    Reload settings with new environment.

    Args:
        environment: Application environment

    Returns:
        New application settings instance
    """
    global _settings
    env = environment or Environment(os.getenv("APP_ENV", "development"))
    _settings = ApplicationSettings(env)
    return _settings
