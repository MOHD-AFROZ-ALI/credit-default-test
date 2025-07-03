"""
Helper Functions Module

This module provides common utility functions for data processing, validation, formatting,
and other utility operations used throughout the Credit Default Prediction application.
"""

import os
import re
import json
import pickle
import joblib
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import wraps
import time
import warnings
import requests
import tempfile
import shutil
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)


# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

def safe_divide(numerator: Union[int, float], denominator: Union[int, float], 
                default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely divide two numbers, returning default value if division by zero.
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def normalize_column_names(df: pd.DataFrame, method: str = 'snake_case') -> pd.DataFrame:
    """
    Normalize DataFrame column names to a consistent format.
    """
    df_copy = df.copy()

    if method == 'snake_case':
        df_copy.columns = [
            re.sub(r'[^a-zA-Z0-9]', '_', col).lower().strip('_')
            for col in df_copy.columns
        ]
    elif method == 'lower':
        df_copy.columns = [col.lower() for col in df_copy.columns]
    elif method == 'upper':
        df_copy.columns = [col.upper() for col in df_copy.columns]
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df_copy


def detect_outliers(data: Union[pd.Series, np.ndarray], method: str = 'iqr', 
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data using various methods.
    """
    data = np.array(data)

    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame using various strategies.
    """
    df_copy = df.copy()
    columns = columns or df_copy.columns.tolist()

    for col in columns:
        if col not in df_copy.columns:
            continue

        if strategy == 'mean' and df_copy[col].dtype in ['int64', 'float64']:
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median' and df_copy[col].dtype in ['int64', 'float64']:
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            mode_value = df_copy[col].mode()
            if not mode_value.empty:
                df_copy[col].fillna(mode_value[0], inplace=True)
        elif strategy == 'drop':
            df_copy.dropna(subset=[col], inplace=True)

    return df_copy


def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson',
                               threshold: float = 0.5) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate correlation matrix and identify highly correlated features.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append(
                    f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}"
                )

    return corr_matrix, high_corr_pairs


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None,
                      min_rows: int = 1, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    # Basic structure validation
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is None or empty")
        return validation_results

    # Row count validation
    if len(df) < min_rows:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

    if max_rows and len(df) > max_rows:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"DataFrame has {len(df)} rows, maximum allowed: {max_rows}")

    # Required columns validation
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {list(missing_columns)}")

    # Data quality checks
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        validation_results['warnings'].append(f"DataFrame contains {null_counts.sum()} null values")

    # Duplicate rows check
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['warnings'].append(f"DataFrame contains {duplicate_count} duplicate rows")

    # Info about the DataFrame
    validation_results['info'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': null_counts.to_dict(),
        'duplicate_rows': duplicate_count
    }

    return validation_results


def validate_model_input(data: Dict[str, Any], required_features: List[str],
                        feature_types: Optional[Dict[str, type]] = None) -> Dict[str, Any]:
    """
    Validate input data for model prediction.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'processed_data': {}
    }

    # Check for required features
    missing_features = set(required_features) - set(data.keys())
    if missing_features:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required features: {list(missing_features)}")
        return validation_results

    # Process and validate each feature
    for feature in required_features:
        value = data.get(feature)

        # Check for None values
        if value is None:
            validation_results['errors'].append(f"Feature '{feature}' cannot be None")
            validation_results['is_valid'] = False
            continue

        # Type validation
        if feature_types and feature in feature_types:
            expected_type = feature_types[feature]
            try:
                if expected_type == float:
                    processed_value = float(value)
                elif expected_type == int:
                    processed_value = int(value)
                elif expected_type == str:
                    processed_value = str(value)
                else:
                    processed_value = value

                validation_results['processed_data'][feature] = processed_value

            except (ValueError, TypeError) as e:
                validation_results['errors'].append(f"Feature '{feature}' type error: {str(e)}")
                validation_results['is_valid'] = False
        else:
            validation_results['processed_data'][feature] = value

    return validation_results


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_number(value: Union[int, float], format_type: str = 'default',
                 decimal_places: int = 2) -> str:
    """
    Format numbers for display.
    """
    if pd.isna(value) or value is None:
        return "N/A"

    try:
        if format_type == 'currency':
            return f"${value:,.{decimal_places}f}"
        elif format_type == 'percentage':
            return f"{value * 100:.{decimal_places}f}%"
        elif format_type == 'scientific':
            return f"{value:.{decimal_places}e}"
        else:  # default
            return f"{value:,.{decimal_places}f}"
    except (ValueError, TypeError):
        return str(value)


def format_datetime(dt: Union[datetime, str], format_string: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format datetime for display.
    """
    if pd.isna(dt) or dt is None:
        return "N/A"

    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return dt.strftime(format_string)
    except (ValueError, TypeError):
        return str(dt)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    """
    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to JSON file.
    """
    try:
        file_path = Path(file_path)
        ensure_directory_exists(file_path.parent)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Any]:
    """
    Load data from JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return None


def save_pickle(obj: Any, file_path: Union[str, Path]) -> bool:
    """
    Save object to pickle file.
    """
    try:
        file_path = Path(file_path)
        ensure_directory_exists(file_path.parent)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {str(e)}")
        return False


def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
    """
    Load object from pickle file.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {str(e)}")
        return None


def save_model(model: Any, file_path: Union[str, Path], use_joblib: bool = True) -> bool:
    """
    Save machine learning model to file.
    """
    try:
        file_path = Path(file_path)
        ensure_directory_exists(file_path.parent)

        if use_joblib:
            joblib.dump(model, file_path)
        else:
            save_pickle(model, file_path)
        return True
    except Exception as e:
        logger.error(f"Error saving model {file_path}: {str(e)}")
        return False


def load_model(file_path: Union[str, Path], use_joblib: bool = True) -> Optional[Any]:
    """
    Load machine learning model from file.
    """
    try:
        if use_joblib:
            return joblib.load(file_path)
        else:
            return load_pickle(file_path)
    except Exception as e:
        logger.error(f"Error loading model {file_path}: {str(e)}")
        return None


def create_zip_archive(source_dir: Union[str, Path], output_path: Union[str, Path],
                      exclude_patterns: Optional[List[str]] = None) -> bool:
    """
    Create ZIP archive from directory.
    """
    try:
        source_dir = Path(source_dir)
        output_path = Path(output_path)
        exclude_patterns = exclude_patterns or []

        ensure_directory_exists(output_path.parent)

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    # Check if file should be excluded
                    should_exclude = False
                    for pattern in exclude_patterns:
                        if pattern in str(file_path):
                            should_exclude = True
                            break

                    if not should_exclude:
                        arcname = file_path.relative_to(source_dir)
                        zipf.write(file_path, arcname)

        return True
    except Exception as e:
        logger.error(f"Error creating ZIP archive: {str(e)}")
        return False


# ============================================================================
# DATA DOWNLOAD AND PROCESSING
# ============================================================================

def download_file(url: str, output_path: Union[str, Path], 
                 chunk_size: int = 8192, timeout: int = 30) -> bool:
    """
    Download file from URL.
    """
    try:
        output_path = Path(output_path)
        ensure_directory_exists(output_path.parent)

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        logger.info(f"Successfully downloaded file from {url} to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        return False


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
    """
    Calculate file hash.
    """
    try:
        hash_func = getattr(hashlib, algorithm)()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {str(e)}")
        return None


# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")

        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function execution on failure.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise

                    logger.warning(f"Function {func.__name__} attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay:.1f}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None
        return wrapper
    return decorator


# ============================================================================
# MISCELLANEOUS UTILITIES
# ============================================================================

def generate_unique_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate unique identifier.
    """
    import uuid
    unique_part = str(uuid.uuid4()).replace('-', '')[:length]
    return f"{prefix}{unique_part}" if prefix else unique_part


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    """
    import platform
    import sys

    info = {
        'platform': platform.platform(),
        'system': platform.system(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'current_time': datetime.now().isoformat()
    }

    return info


def suppress_warnings(func: Callable) -> Callable:
    """
    Decorator to suppress warnings from function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    """
    Safely cast value to target type.
    """
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default


# ============================================================================
# CREDIT DEFAULT SPECIFIC UTILITIES
# ============================================================================

def calculate_credit_utilization(limit_bal: float, bill_amt: float) -> float:
    """
    Calculate credit utilization ratio.
    """
    return safe_divide(bill_amt, limit_bal, 0.0)


def calculate_payment_ratio(pay_amt: float, bill_amt: float) -> float:
    """
    Calculate payment ratio.
    """
    return safe_divide(pay_amt, bill_amt, 0.0)


def categorize_risk_level(probability: float) -> str:
    """
    Categorize risk level based on default probability.
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"


def get_risk_color(risk_level: str) -> str:
    """
    Get color code for risk level.
    """
    color_map = {
        "Low Risk": "#2ca02c",
        "Medium Risk": "#ff7f0e", 
        "High Risk": "#d62728"
    }
    return color_map.get(risk_level, "#1f77b4")


if __name__ == "__main__":
    # Test some functions
    print("Testing helper functions...")

    # Test safe_divide
    print(f"Safe divide 10/2: {safe_divide(10, 2)}")
    print(f"Safe divide 10/0: {safe_divide(10, 0, default='undefined')}")

    # Test format_number
    print(f"Format currency: {format_number(1234.56, 'currency')}")
    print(f"Format percentage: {format_number(0.1234, 'percentage')}")

    # Test generate_unique_id
    print(f"Unique ID: {generate_unique_id('test_')}")

    print("Helper functions test completed!")
