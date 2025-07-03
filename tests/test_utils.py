"""
Test suite for utility functions and helper modules
Comprehensive tests for configuration, logging, helpers, and application utilities
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
import json
import yaml
import logging
from datetime import datetime
import io
import sys
import warnings
warnings.filterwarnings('ignore')

# Test fixtures
@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'app': {
            'name': 'Credit Default Prediction System',
            'version': '2.0.0',
            'debug': False,
            'log_level': 'INFO'
        },
        'database': {
            'type': 'sqlite',
            'path': 'data/credit_data.db'
        },
        'model': {
            'default_algorithm': 'random_forest',
            'algorithms': ['logistic_regression', 'random_forest', 'gradient_boosting']
        },
        'features': {
            'numerical_features': ['age', 'income', 'credit_score'],
            'categorical_features': ['employment_status', 'home_ownership']
        }
    }

@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary config file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing utility functions"""
    np.random.seed(42)
    return pd.DataFrame({
        'customer_id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'income': np.random.lognormal(10.5, 0.5, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'employment_status': np.random.choice(['employed', 'self_employed', 'unemployed'], 100),
        'default': np.random.choice([0, 1], 100, p=[0.8, 0.2])
    })

class TestConfigurationManager:
    """Test configuration management utilities"""

    def test_load_config_from_yaml(self, temp_config_file, sample_config):
        """Test loading configuration from YAML file"""
        class ConfigManager:
            def load_config(self, config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)

        config_manager = ConfigManager()
        loaded_config = config_manager.load_config(temp_config_file)

        assert loaded_config == sample_config
        assert loaded_config['app']['name'] == 'Credit Default Prediction System'
        assert loaded_config['model']['default_algorithm'] == 'random_forest'

    def test_get_config_value(self, sample_config):
        """Test getting nested configuration values"""
        class ConfigManager:
            def __init__(self, config):
                self.config = config

            def get_value(self, key_path, default=None):
                """Get nested config value using dot notation"""
                keys = key_path.split('.')
                value = self.config

                try:
                    for key in keys:
                        value = value[key]
                    return value
                except (KeyError, TypeError):
                    return default

        config_manager = ConfigManager(sample_config)

        # Test existing nested keys
        assert config_manager.get_value('app.name') == 'Credit Default Prediction System'
        assert config_manager.get_value('model.default_algorithm') == 'random_forest'

        # Test non-existing keys with default
        assert config_manager.get_value('nonexistent.key', 'default_value') == 'default_value'
        assert config_manager.get_value('app.nonexistent', None) is None

    def test_validate_config_structure(self, sample_config):
        """Test configuration structure validation"""
        class ConfigValidator:
            def __init__(self, required_sections):
                self.required_sections = required_sections

            def validate_config(self, config):
                errors = []

                for section in self.required_sections:
                    if section not in config:
                        errors.append(f"Missing required section: {section}")

                if 'app' in config and 'name' not in config['app']:
                    errors.append("Missing app.name")

                if errors:
                    raise ValueError(f"Configuration validation errors: {errors}")

                return True

        validator = ConfigValidator(['app', 'database', 'model', 'features'])

        # Should pass with valid config
        assert validator.validate_config(sample_config) is True

        # Should fail with incomplete config
        incomplete_config = {'app': {'name': 'Test'}}
        with pytest.raises(ValueError, match="Configuration validation errors"):
            validator.validate_config(incomplete_config)

class TestLoggingUtilities:
    """Test logging utility functions"""

    def test_setup_logger_basic(self):
        """Test basic logger setup"""
        class LoggerSetup:
            def setup_logger(self, name, level=logging.INFO):
                logger = logging.getLogger(name)
                logger.setLevel(level)

                # Remove existing handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)

                # Console handler
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)

                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)

                logger.addHandler(console_handler)
                return logger

        logger_setup = LoggerSetup()
        logger = logger_setup.setup_logger('test_logger')

        # Test that logger is properly configured
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_log_function_execution(self):
        """Test decorator for logging function execution"""
        class LoggingDecorator:
            def __init__(self, logger):
                self.logger = logger

            def log_execution(self, func):
                """Decorator to log function execution"""
                def wrapper(*args, **kwargs):
                    self.logger.info(f"Starting execution of {func.__name__}")
                    try:
                        result = func(*args, **kwargs)
                        self.logger.info(f"Successfully completed {func.__name__}")
                        return result
                    except Exception as e:
                        self.logger.error(f"Error in {func.__name__}: {str(e)}")
                        raise
                return wrapper

        # Setup logger with string capture
        logger = logging.getLogger('decorator_test')
        logger.setLevel(logging.INFO)

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)

        decorator = LoggingDecorator(logger)

        @decorator.log_execution
        def test_function(x, y):
            return x + y

        # Test successful execution
        result = test_function(2, 3)
        assert result == 5

        log_output = log_capture.getvalue()
        assert "Starting execution of test_function" in log_output
        assert "Successfully completed test_function" in log_output

class TestDataUtilities:
    """Test data manipulation utility functions"""

    def test_safe_divide(self):
        """Test safe division utility"""
        class MathUtils:
            @staticmethod
            def safe_divide(numerator, denominator, default=0):
                """Safely divide two numbers, returning default if denominator is zero"""
                try:
                    if denominator == 0:
                        return default
                    return numerator / denominator
                except (TypeError, ValueError):
                    return default

        # Test normal division
        assert MathUtils.safe_divide(10, 2) == 5.0
        assert MathUtils.safe_divide(7, 3) == pytest.approx(2.333, rel=1e-3)

        # Test division by zero
        assert MathUtils.safe_divide(10, 0) == 0
        assert MathUtils.safe_divide(10, 0, default=-1) == -1

        # Test invalid inputs
        assert MathUtils.safe_divide("10", 2) == 0
        assert MathUtils.safe_divide(10, "2") == 0

    def test_calculate_percentiles(self, sample_dataframe):
        """Test percentile calculation utility"""
        class StatUtils:
            @staticmethod
            def calculate_percentiles(data, column, percentiles=[25, 50, 75, 90, 95]):
                """Calculate percentiles for a column"""
                if column not in data.columns:
                    raise ValueError(f"Column {column} not found in data")

                result = {}
                for p in percentiles:
                    result[f'p{p}'] = data[column].quantile(p / 100)

                return result

        # Test percentile calculation
        percentiles = StatUtils.calculate_percentiles(sample_dataframe, 'age')

        assert 'p25' in percentiles
        assert 'p50' in percentiles
        assert 'p75' in percentiles

        # Check that percentiles are in ascending order
        assert percentiles['p25'] <= percentiles['p50']
        assert percentiles['p50'] <= percentiles['p75']

        # Test with invalid column
        with pytest.raises(ValueError, match="Column nonexistent not found"):
            StatUtils.calculate_percentiles(sample_dataframe, 'nonexistent')

    def test_detect_outliers_iqr(self, sample_dataframe):
        """Test IQR-based outlier detection"""
        class OutlierDetector:
            @staticmethod
            def detect_outliers_iqr(data, column, multiplier=1.5):
                """Detect outliers using IQR method"""
                if column not in data.columns:
                    raise ValueError(f"Column {column} not found")

                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

                return {
                    'outliers': outliers,
                    'outlier_indices': outliers.index.tolist(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': len(outliers)
                }

        # Add some outliers to test data
        test_data = sample_dataframe.copy()
        test_data.loc[0, 'age'] = 150  # Extreme outlier
        test_data.loc[1, 'age'] = 5    # Extreme outlier

        outlier_results = OutlierDetector.detect_outliers_iqr(test_data, 'age')

        assert 'outliers' in outlier_results
        assert 'outlier_count' in outlier_results
        assert 'lower_bound' in outlier_results
        assert 'upper_bound' in outlier_results

        # Should detect the extreme outliers we added
        assert outlier_results['outlier_count'] >= 2

    def test_normalize_column_names(self):
        """Test column name normalization utility"""
        class DataUtils:
            @staticmethod
            def normalize_column_names(data):
                """Normalize column names to lowercase with underscores"""
                normalized_data = data.copy()

                column_mapping = {}
                for col in data.columns:
                    # Convert to lowercase, replace spaces and special chars with underscores
                    new_col = col.lower().replace(' ', '_').replace('-', '_')
                    # Remove multiple consecutive underscores
                    while '__' in new_col:
                        new_col = new_col.replace('__', '_')
                    # Remove leading/trailing underscores
                    new_col = new_col.strip('_')
                    column_mapping[col] = new_col

                normalized_data.rename(columns=column_mapping, inplace=True)
                return normalized_data, column_mapping

        # Create test data with messy column names
        messy_data = pd.DataFrame({
            'Customer ID': [1, 2, 3],
            'First Name': ['John', 'Jane', 'Bob'],
            'Credit-Score': [700, 750, 680],
            ' Employment Status ': ['employed', 'self_employed', 'unemployed']
        })

        normalized_data, mapping = DataUtils.normalize_column_names(messy_data)

        expected_columns = ['customer_id', 'first_name', 'credit_score', 'employment_status']

        assert list(normalized_data.columns) == expected_columns
        assert len(mapping) == 4
        assert mapping['Customer ID'] == 'customer_id'
        assert mapping['Credit-Score'] == 'credit_score'

class TestFileUtilities:
    """Test file handling utility functions"""

    def test_safe_file_read(self):
        """Test safe file reading utility"""
        class FileUtils:
            @staticmethod
            def safe_file_read(file_path, encoding='utf-8', default_content=''):
                """Safely read file content with error handling"""
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except FileNotFoundError:
                    return default_content
                except Exception:
                    return default_content

        # Test with existing file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            content = FileUtils.safe_file_read(temp_file)
            assert content == "Test content"

            # Test with non-existent file
            content = FileUtils.safe_file_read("nonexistent.txt", default_content="Default")
            assert content == "Default"

        finally:
            os.unlink(temp_file)

    def test_ensure_directory_exists(self):
        """Test directory creation utility"""
        class FileUtils:
            @staticmethod
            def ensure_directory_exists(directory_path):
                """Ensure directory exists, create if it doesn't"""
                try:
                    os.makedirs(directory_path, exist_ok=True)
                    return True
                except Exception:
                    return False

        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, 'test_subdir', 'nested')

            # Directory shouldn't exist initially
            assert not os.path.exists(test_dir)

            # Create directory
            result = FileUtils.ensure_directory_exists(test_dir)
            assert result is True
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)

    def test_get_file_info(self):
        """Test file information utility"""
        class FileUtils:
            @staticmethod
            def get_file_info(file_path):
                """Get comprehensive file information"""
                if not os.path.exists(file_path):
                    return None

                stat = os.stat(file_path)

                return {
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'extension': os.path.splitext(file_path)[1],
                    'size_bytes': stat.st_size,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(stat.st_ctime),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'is_file': os.path.isfile(file_path),
                    'is_directory': os.path.isdir(file_path)
                }

        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for file info")
            temp_file = f.name

        try:
            file_info = FileUtils.get_file_info(temp_file)

            assert file_info is not None
            assert file_info['name'] == os.path.basename(temp_file)
            assert file_info['extension'] == '.txt'
            assert file_info['size_bytes'] > 0
            assert isinstance(file_info['created'], datetime)
            assert file_info['is_file'] is True
            assert file_info['is_directory'] is False

            # Test with non-existent file
            assert FileUtils.get_file_info("nonexistent.txt") is None

        finally:
            os.unlink(temp_file)

class TestValidationUtilities:
    """Test data validation utility functions"""

    def test_validate_email(self):
        """Test email validation utility"""
        import re

        class ValidationUtils:
            @staticmethod
            def validate_email(email):
                """Validate email address format"""
                if not email or not isinstance(email, str):
                    return False

                pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                return bool(re.match(pattern, email))

        # Test valid emails
        assert ValidationUtils.validate_email("test@example.com") is True
        assert ValidationUtils.validate_email("user.name+tag@domain.co.uk") is True

        # Test invalid emails
        assert ValidationUtils.validate_email("invalid-email") is False
        assert ValidationUtils.validate_email("@domain.com") is False
        assert ValidationUtils.validate_email("") is False
        assert ValidationUtils.validate_email(None) is False

    def test_validate_credit_score(self):
        """Test credit score validation utility"""
        class ValidationUtils:
            @staticmethod
            def validate_credit_score(score):
                """Validate credit score range"""
                try:
                    score = float(score)
                    return 300 <= score <= 850
                except (ValueError, TypeError):
                    return False

        # Test valid credit scores
        assert ValidationUtils.validate_credit_score(300) is True
        assert ValidationUtils.validate_credit_score(850) is True
        assert ValidationUtils.validate_credit_score(720) is True
        assert ValidationUtils.validate_credit_score("750") is True

        # Test invalid credit scores
        assert ValidationUtils.validate_credit_score(299) is False
        assert ValidationUtils.validate_credit_score(851) is False
        assert ValidationUtils.validate_credit_score("invalid") is False
        assert ValidationUtils.validate_credit_score(None) is False

    def test_validate_age(self):
        """Test age validation utility"""
        class ValidationUtils:
            @staticmethod
            def validate_age(age):
                """Validate age range"""
                try:
                    age = int(age)
                    return 18 <= age <= 100
                except (ValueError, TypeError):
                    return False

        # Test valid ages
        assert ValidationUtils.validate_age(18) is True
        assert ValidationUtils.validate_age(65) is True
        assert ValidationUtils.validate_age(100) is True
        assert ValidationUtils.validate_age("25") is True

        # Test invalid ages
        assert ValidationUtils.validate_age(17) is False
        assert ValidationUtils.validate_age(101) is False
        assert ValidationUtils.validate_age(-5) is False
        assert ValidationUtils.validate_age("invalid") is False

class TestPerformanceUtilities:
    """Test performance monitoring utility functions"""

    def test_execution_timer(self):
        """Test execution time measurement utility"""
        import time

        class PerformanceUtils:
            @staticmethod
            def time_execution(func):
                """Decorator to measure function execution time"""
                def wrapper(*args, **kwargs):
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    execution_time = end_time - start_time

                    # Store execution time as function attribute
                    wrapper.execution_time = execution_time
                    return result
                return wrapper

        @PerformanceUtils.time_execution
        def test_function():
            time.sleep(0.1)  # Sleep for 100ms
            return "completed"

        result = test_function()

        assert result == "completed"
        assert hasattr(test_function, 'execution_time')
        assert test_function.execution_time >= 0.1  # Should be at least 100ms

    def test_memory_profiler(self):
        """Test memory usage profiling utility"""
        import psutil
        import os

        class PerformanceUtils:
            @staticmethod
            def get_memory_usage():
                """Get current memory usage"""
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()

                return {
                    'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
                    'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                    'percent': process.memory_percent()
                }

        try:
            memory_usage = PerformanceUtils.get_memory_usage()

            assert 'rss_mb' in memory_usage
            assert 'vms_mb' in memory_usage
            assert 'percent' in memory_usage

            assert memory_usage['rss_mb'] > 0
            assert memory_usage['vms_mb'] > 0
            assert 0 <= memory_usage['percent'] <= 100

        except ImportError:
            # psutil might not be available in test environment
            pytest.skip("psutil not available for memory profiling test")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
