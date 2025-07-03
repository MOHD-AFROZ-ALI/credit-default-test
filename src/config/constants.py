"""
Application Constants

This module defines all constant values used throughout the Credit Default
Prediction application, including UCI dataset feature definitions, risk
categories, model thresholds, and UI constants.
"""

from typing import Dict, List, Tuple, Any
from enum import Enum


# =============================================================================
# UCI CREDIT DEFAULT DATASET CONSTANTS
# =============================================================================

class UCIFeatures:
    """UCI Credit Default Dataset feature definitions."""

    # Feature names and their descriptions
    FEATURE_DESCRIPTIONS = {
        'LIMIT_BAL': 'Amount of given credit (NT dollar)',
        'SEX': 'Gender (1=male, 2=female)',
        'EDUCATION': 'Education level (1=graduate school, 2=university, 3=high school, 4=others)',
        'MARRIAGE': 'Marital status (1=married, 2=single, 3=others)',
        'AGE': 'Age in years',
        'PAY_0': 'Repayment status in September 2005',
        'PAY_2': 'Repayment status in August 2005',
        'PAY_3': 'Repayment status in July 2005',
        'PAY_4': 'Repayment status in June 2005',
        'PAY_5': 'Repayment status in May 2005',
        'PAY_6': 'Repayment status in April 2005',
        'BILL_AMT1': 'Amount of bill statement in September 2005 (NT dollar)',
        'BILL_AMT2': 'Amount of bill statement in August 2005 (NT dollar)',
        'BILL_AMT3': 'Amount of bill statement in July 2005 (NT dollar)',
        'BILL_AMT4': 'Amount of bill statement in June 2005 (NT dollar)',
        'BILL_AMT5': 'Amount of bill statement in May 2005 (NT dollar)',
        'BILL_AMT6': 'Amount of bill statement in April 2005 (NT dollar)',
        'PAY_AMT1': 'Amount of previous payment in September 2005 (NT dollar)',
        'PAY_AMT2': 'Amount of previous payment in August 2005 (NT dollar)',
        'PAY_AMT3': 'Amount of previous payment in July 2005 (NT dollar)',
        'PAY_AMT4': 'Amount of previous payment in June 2005 (NT dollar)',
        'PAY_AMT5': 'Amount of previous payment in May 2005 (NT dollar)',
        'PAY_AMT6': 'Amount of previous payment in April 2005 (NT dollar)',
        'default.payment.next.month': 'Default payment (1=yes, 0=no)'
    }

    # Categorical feature mappings
    SEX_MAPPING = {1: 'Male', 2: 'Female'}

    EDUCATION_MAPPING = {
        1: 'Graduate School',
        2: 'University',
        3: 'High School',
        4: 'Others',
        5: 'Unknown',
        6: 'Unknown'
    }

    MARRIAGE_MAPPING = {
        1: 'Married',
        2: 'Single',
        3: 'Others',
        0: 'Others'
    }

    # Payment status mappings
    PAYMENT_STATUS_MAPPING = {
        -2: 'No consumption',
        -1: 'Paid in full',
        0: 'Use of revolving credit',
        1: 'Payment delay for 1 month',
        2: 'Payment delay for 2 months',
        3: 'Payment delay for 3 months',
        4: 'Payment delay for 4 months',
        5: 'Payment delay for 5 months',
        6: 'Payment delay for 6 months',
        7: 'Payment delay for 7 months',
        8: 'Payment delay for 8 months',
        9: 'Payment delay for 9+ months'
    }

    # Feature groups for analysis
    DEMOGRAPHIC_FEATURES = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
    PAYMENT_HISTORY_FEATURES = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    BILL_AMOUNT_FEATURES = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    PAYMENT_AMOUNT_FEATURES = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    # Target variable
    TARGET_COLUMN = 'default.payment.next.month'

    # Expected data types
    NUMERIC_FEATURES = ['LIMIT_BAL', 'AGE'] + BILL_AMOUNT_FEATURES + PAYMENT_AMOUNT_FEATURES
    CATEGORICAL_FEATURES = ['SEX', 'EDUCATION', 'MARRIAGE'] + PAYMENT_HISTORY_FEATURES


# =============================================================================
# RISK CATEGORIES AND THRESHOLDS
# =============================================================================

class RiskCategory(Enum):
    """Risk category classifications."""
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"
    VERY_HIGH = "Very High Risk"


class RiskThresholds:
    """Risk assessment thresholds."""

    # Probability thresholds for risk categories
    RISK_PROBABILITY_THRESHOLDS = {
        RiskCategory.LOW: (0.0, 0.25),
        RiskCategory.MEDIUM: (0.25, 0.50),
        RiskCategory.HIGH: (0.50, 0.75),
        RiskCategory.VERY_HIGH: (0.75, 1.0)
    }

    # Credit score ranges (if applicable)
    CREDIT_SCORE_RANGES = {
        'Excellent': (800, 850),
        'Very Good': (740, 799),
        'Good': (670, 739),
        'Fair': (580, 669),
        'Poor': (300, 579)
    }

    # Default probability thresholds for decision making
    DEFAULT_THRESHOLD = 0.5
    CONSERVATIVE_THRESHOLD = 0.3
    AGGRESSIVE_THRESHOLD = 0.7


# =============================================================================
# MODEL PERFORMANCE THRESHOLDS
# =============================================================================

class ModelThresholds:
    """Model performance thresholds and benchmarks."""

    # Minimum acceptable performance metrics
    MIN_AUC_SCORE = 0.75
    MIN_PRECISION = 0.70
    MIN_RECALL = 0.65
    MIN_F1_SCORE = 0.67
    MIN_ACCURACY = 0.75

    # Target performance metrics
    TARGET_AUC_SCORE = 0.85
    TARGET_PRECISION = 0.80
    TARGET_RECALL = 0.75
    TARGET_F1_SCORE = 0.77
    TARGET_ACCURACY = 0.82

    # Model comparison thresholds
    SIGNIFICANT_IMPROVEMENT = 0.02  # 2% improvement considered significant
    MODEL_DEGRADATION_THRESHOLD = 0.05  # 5% degradation triggers alert

    # Cross-validation parameters
    CV_FOLDS = 5
    CV_SCORING = 'roc_auc'

    # Early stopping parameters
    EARLY_STOPPING_ROUNDS = 50
    EARLY_STOPPING_TOLERANCE = 0.001


# =============================================================================
# DATA PROCESSING CONSTANTS
# =============================================================================

class DataConstants:
    """Data processing and validation constants."""

    # Missing value handling
    MAX_MISSING_PERCENTAGE = 0.30
    MISSING_VALUE_INDICATORS = [-999, -9999, 'NULL', 'null', 'N/A', 'n/a', '']

    # Outlier detection
    OUTLIER_METHODS = ['iqr', 'zscore', 'isolation_forest']
    IQR_MULTIPLIER = 1.5
    ZSCORE_THRESHOLD = 3.0

    # Feature engineering
    CORRELATION_THRESHOLD = 0.95
    VARIANCE_THRESHOLD = 0.01
    FEATURE_IMPORTANCE_THRESHOLD = 0.001

    # Data validation rules
    AGE_MIN = 18
    AGE_MAX = 100
    LIMIT_BAL_MIN = 0
    LIMIT_BAL_MAX = 10000000  # 10M NT dollars

    # Sampling parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2


# =============================================================================
# API CONSTANTS
# =============================================================================

class APIConstants:
    """API-related constants."""

    # HTTP status codes
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_400_BAD_REQUEST = 400
    HTTP_401_UNAUTHORIZED = 401
    HTTP_404_NOT_FOUND = 404
    HTTP_422_UNPROCESSABLE_ENTITY = 422
    HTTP_500_INTERNAL_SERVER_ERROR = 500

    # API response messages
    SUCCESS_MESSAGE = "Request processed successfully"
    VALIDATION_ERROR_MESSAGE = "Input validation failed"
    MODEL_ERROR_MESSAGE = "Model prediction failed"
    SERVER_ERROR_MESSAGE = "Internal server error"

    # Request limits
    MAX_BATCH_SIZE = 1000
    MAX_REQUEST_SIZE_MB = 16
    REQUEST_TIMEOUT_SECONDS = 30

    # API versions
    CURRENT_API_VERSION = "v1"
    SUPPORTED_API_VERSIONS = ["v1"]


# =============================================================================
# UI/VISUALIZATION CONSTANTS
# =============================================================================

class ColorSchemes:
    """Color schemes for visualizations and UI."""

    # Risk category colors
    RISK_COLORS = {
        RiskCategory.LOW: '#2E8B57',      # Sea Green
        RiskCategory.MEDIUM: '#FFD700',   # Gold
        RiskCategory.HIGH: '#FF8C00',     # Dark Orange
        RiskCategory.VERY_HIGH: '#DC143C' # Crimson
    }

    # Model performance colors
    PERFORMANCE_COLORS = {
        'excellent': '#228B22',  # Forest Green
        'good': '#32CD32',       # Lime Green
        'fair': '#FFD700',       # Gold
        'poor': '#FF6347',       # Tomato
        'very_poor': '#DC143C'   # Crimson
    }

    # General color palette
    PRIMARY_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Gradient colors
    GRADIENT_COLORS = {
        'blue_gradient': ['#e3f2fd', '#1976d2'],
        'green_gradient': ['#e8f5e8', '#2e7d32'],
        'red_gradient': ['#ffebee', '#c62828'],
        'orange_gradient': ['#fff3e0', '#ef6c00']
    }


class ChartConstants:
    """Chart and visualization constants."""

    # Default figure sizes
    FIGURE_SIZE_SMALL = (8, 6)
    FIGURE_SIZE_MEDIUM = (12, 8)
    FIGURE_SIZE_LARGE = (16, 10)

    # Font sizes
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 12
    TICK_FONT_SIZE = 10
    LEGEND_FONT_SIZE = 10

    # Chart types
    SUPPORTED_CHART_TYPES = [
        'bar', 'line', 'scatter', 'histogram', 'box', 'violin',
        'heatmap', 'pie', 'area', 'density'
    ]

    # Default chart parameters
    DPI = 300
    ALPHA = 0.7
    LINE_WIDTH = 2
    MARKER_SIZE = 6


# =============================================================================
# FILE AND PATH CONSTANTS
# =============================================================================

class FileConstants:
    """File handling constants."""

    # Supported file formats
    SUPPORTED_INPUT_FORMATS = ['csv', 'xlsx', 'json', 'parquet']
    SUPPORTED_OUTPUT_FORMATS = ['csv', 'xlsx', 'json', 'parquet', 'pickle']

    # File size limits
    MAX_FILE_SIZE_MB = 100
    MAX_BATCH_FILE_SIZE_MB = 500

    # Default file names
    DEFAULT_MODEL_FILENAME = 'credit_default_model.pkl'
    DEFAULT_SCALER_FILENAME = 'feature_scaler.pkl'
    DEFAULT_ENCODER_FILENAME = 'label_encoder.pkl'
    DEFAULT_CONFIG_FILENAME = 'config.yaml'

    # File extensions
    MODEL_EXTENSIONS = ['.pkl', '.joblib', '.h5', '.onnx']
    DATA_EXTENSIONS = ['.csv', '.xlsx', '.json', '.parquet']
    CONFIG_EXTENSIONS = ['.yaml', '.yml', '.json']


class PathConstants:
    """Path-related constants."""

    # Default directory structure
    DEFAULT_DIRECTORIES = [
        'data/raw',
        'data/processed',
        'data/features',
        'models',
        'logs',
        'reports',
        'config',
        'output'
    ]

    # Environment-specific paths
    DEVELOPMENT_DATA_PATH = 'data/dev/'
    TESTING_DATA_PATH = 'data/test/'
    PRODUCTION_DATA_PATH = 'data/prod/'


# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

class LoggingConstants:
    """Logging-related constants."""

    # Log levels
    LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    # Log formats
    STANDARD_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
    JSON_FORMAT = 'json'

    # Log file settings
    MAX_LOG_FILE_SIZE_MB = 10
    LOG_BACKUP_COUNT = 5

    # Logger names
    MAIN_LOGGER = 'credit_default'
    MODEL_LOGGER = 'credit_default.model'
    DATA_LOGGER = 'credit_default.data'
    API_LOGGER = 'credit_default.api'


# =============================================================================
# ERROR MESSAGES
# =============================================================================

class ErrorMessages:
    """Standard error messages."""

    # Data validation errors
    INVALID_DATA_TYPE = "Invalid data type for field '{field}'. Expected {expected}, got {actual}"
    MISSING_REQUIRED_FIELD = "Missing required field: '{field}'"
    VALUE_OUT_OF_RANGE = "Value for '{field}' is out of valid range [{min_val}, {max_val}]"
    INVALID_CATEGORICAL_VALUE = "Invalid value '{value}' for categorical field '{field}'"

    # Model errors
    MODEL_NOT_FOUND = "Model not found: {model_name}"
    MODEL_LOAD_ERROR = "Failed to load model: {error}"
    PREDICTION_ERROR = "Prediction failed: {error}"
    MODEL_TRAINING_ERROR = "Model training failed: {error}"

    # File errors
    FILE_NOT_FOUND = "File not found: {filepath}"
    FILE_READ_ERROR = "Failed to read file: {filepath}"
    FILE_WRITE_ERROR = "Failed to write file: {filepath}"
    INVALID_FILE_FORMAT = "Invalid file format. Supported formats: {formats}"

    # API errors
    INVALID_REQUEST_FORMAT = "Invalid request format"
    AUTHENTICATION_FAILED = "Authentication failed"
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded"
    REQUEST_TIMEOUT = "Request timeout"


# =============================================================================
# SUCCESS MESSAGES
# =============================================================================

class SuccessMessages:
    """Standard success messages."""

    MODEL_TRAINED_SUCCESSFULLY = "Model trained successfully with AUC: {auc:.4f}"
    PREDICTION_COMPLETED = "Prediction completed for {count} samples"
    DATA_PROCESSED_SUCCESSFULLY = "Data processed successfully. Shape: {shape}"
    FILE_SAVED_SUCCESSFULLY = "File saved successfully: {filepath}"
    MODEL_SAVED_SUCCESSFULLY = "Model saved successfully: {filepath}"


# =============================================================================
# UTILITY CONSTANTS
# =============================================================================

class MiscConstants:
    """Miscellaneous constants."""

    # Date formats
    DATE_FORMAT = '%Y-%m-%d'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

    # Currency
    CURRENCY_SYMBOL = 'NT$'
    CURRENCY_NAME = 'New Taiwan Dollar'

    # Precision for floating point comparisons
    FLOAT_PRECISION = 1e-6

    # Default values
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_N_JOBS = -1

    # Regular expressions
    EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    PHONE_REGEX = r'^\+?1?\d{9,15}$'
