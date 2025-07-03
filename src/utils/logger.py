"""
Logging Setup Module

This module provides centralized logging configuration for the Credit Default Prediction application.
It sets up structured logging with different levels, formatters, and handlers for console and file output.
Includes performance monitoring, function call tracking, and specialized logging for ML operations.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import json
from functools import wraps
import traceback
import time
import threading
from contextlib import contextmanager


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""

    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to the log level
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }

        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log records"""

    def filter(self, record):
        return hasattr(record, 'performance') and record.performance


class MLOperationFilter(logging.Filter):
    """Filter for ML operation log records"""

    def filter(self, record):
        return hasattr(record, 'ml_operation') and record.ml_operation


class CreditDefaultLogger:
    """
    Centralized logger for the Credit Default Prediction application.

    This class provides structured logging with multiple handlers, formatters,
    performance monitoring, and specialized ML operation logging.
    """

    def __init__(self, 
                 name: str = "credit_default",
                 level: str = "INFO",
                 log_dir: str = "logs",
                 log_file: str = "app.log",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True,
                 file_output: bool = True,
                 json_format: bool = False,
                 performance_logging: bool = True,
                 ml_logging: bool = True):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            log_file: Log file name
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
            console_output: Enable console output
            file_output: Enable file output
            json_format: Use JSON format for structured logging
            performance_logging: Enable performance logging
            ml_logging: Enable ML operation logging
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.console_output = console_output
        self.file_output = file_output
        self.json_format = json_format
        self.performance_logging = performance_logging
        self.ml_logging = ml_logging

        # Performance tracking
        self.performance_metrics = {}
        self.function_call_counts = {}
        self.error_counts = {}

        # Thread-local storage for context
        self.local = threading.local()

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup logging handlers"""

        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)

            if self.json_format:
                console_formatter = JSONFormatter()
            else:
                console_formatter = ColoredFormatter(
                    '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )

            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if self.file_output:
            # Create log directory if it doesn't exist
            self.log_dir.mkdir(parents=True, exist_ok=True)

            log_file_path = self.log_dir / self.log_file

            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.level)

            if self.json_format:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )

            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Performance handler (separate file)
        if self.performance_logging:
            perf_file_path = self.log_dir / "performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.addFilter(PerformanceFilter())

            perf_formatter = JSONFormatter() if self.json_format else logging.Formatter(
                '%(asctime)s | PERF | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            perf_handler.setFormatter(perf_formatter)
            self.logger.addHandler(perf_handler)

        # ML operations handler (separate file)
        if self.ml_logging:
            ml_file_path = self.log_dir / "ml_operations.log"
            ml_handler = logging.handlers.RotatingFileHandler(
                ml_file_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            ml_handler.setLevel(logging.INFO)
            ml_handler.addFilter(MLOperationFilter())

            ml_formatter = JSONFormatter() if self.json_format else logging.Formatter(
                '%(asctime)s | ML | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            ml_handler.setFormatter(ml_formatter)
            self.logger.addHandler(ml_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
        self._track_error(message)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
        self._track_error(message, critical=True)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)
        self._track_error(message, exception=True)

    def _track_error(self, message: str, critical: bool = False, exception: bool = False):
        """Track error statistics"""
        error_type = 'critical' if critical else 'exception' if exception else 'error'
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None, result: Any = None):
        """Log function call details"""
        kwargs = kwargs or {}

        # Track function call counts
        if func_name not in self.function_call_counts:
            self.function_call_counts[func_name] = 0
        self.function_call_counts[func_name] += 1

        self.logger.debug(
            f"Function call: {func_name}",
            extra={
                'function': func_name,
                'args_count': len(args),
                'kwargs_count': len(kwargs),
                'result_type': type(result).__name__ if result is not None else None,
                'call_count': self.function_call_counts[func_name]
            }
        )

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        # Track performance metrics
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                'total_time': 0,
                'call_count': 0,
                'min_time': float('inf'),
                'max_time': 0
            }

        perf_data = self.performance_metrics[operation]
        perf_data['total_time'] += duration
        perf_data['call_count'] += 1
        perf_data['min_time'] = min(perf_data['min_time'], duration)
        perf_data['max_time'] = max(perf_data['max_time'], duration)
        perf_data['avg_time'] = perf_data['total_time'] / perf_data['call_count']

        self.logger.info(
            f"Performance: {operation} completed in {duration:.4f}s",
            extra={
                'performance': True,
                'operation': operation,
                'duration': duration,
                'avg_duration': perf_data['avg_time'],
                'min_duration': perf_data['min_time'],
                'max_duration': perf_data['max_time'],
                'call_count': perf_data['call_count'],
                **metrics
            }
        )

    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], dataset_info: Dict[str, Any] = None):
        """Log model performance metrics"""
        dataset_info = dataset_info or {}

        self.logger.info(
            f"Model metrics: {model_name}",
            extra={
                'ml_operation': True,
                'operation_type': 'model_evaluation',
                'model': model_name,
                'metrics': metrics,
                'dataset_info': dataset_info,
                'timestamp': datetime.now().isoformat()
            }
        )

    def log_data_info(self, dataset_name: str, shape: tuple, columns: list = None, memory_usage: float = None):
        """Log dataset information"""
        self.logger.info(
            f"Dataset loaded: {dataset_name}",
            extra={
                'ml_operation': True,
                'operation_type': 'data_loading',
                'dataset': dataset_name,
                'shape': shape,
                'columns_count': len(columns) if columns else None,
                'memory_usage_mb': memory_usage,
                'sample_columns': columns[:10] if columns else None
            }
        )

    def log_prediction(self, prediction_type: str, input_data: dict, prediction: Any, 
                      confidence: float = None, model_name: str = None):
        """Log prediction details"""
        self.logger.info(
            f"Prediction made: {prediction_type}",
            extra={
                'ml_operation': True,
                'operation_type': 'prediction',
                'prediction_type': prediction_type,
                'model_name': model_name,
                'input_features_count': len(input_data) if isinstance(input_data, dict) else 'unknown',
                'prediction': str(prediction),
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        )

    def log_model_training(self, model_name: str, training_time: float, 
                          train_score: float = None, val_score: float = None, 
                          hyperparameters: Dict[str, Any] = None):
        """Log model training information"""
        self.logger.info(
            f"Model training completed: {model_name}",
            extra={
                'ml_operation': True,
                'operation_type': 'model_training',
                'model': model_name,
                'training_time': training_time,
                'train_score': train_score,
                'validation_score': val_score,
                'hyperparameters': hyperparameters,
                'timestamp': datetime.now().isoformat()
            }
        )

    def log_data_preprocessing(self, operation: str, input_shape: tuple, output_shape: tuple, 
                              processing_time: float = None):
        """Log data preprocessing operations"""
        self.logger.info(
            f"Data preprocessing: {operation}",
            extra={
                'ml_operation': True,
                'operation_type': 'data_preprocessing',
                'preprocessing_operation': operation,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'processing_time': processing_time,
                'shape_change': (output_shape[0] - input_shape[0], output_shape[1] - input_shape[1]) if len(input_shape) == 2 and len(output_shape) == 2 else None
            }
        )

    def log_feature_engineering(self, operation: str, features_before: int, features_after: int, 
                               feature_names: list = None):
        """Log feature engineering operations"""
        self.logger.info(
            f"Feature engineering: {operation}",
            extra={
                'ml_operation': True,
                'operation_type': 'feature_engineering',
                'engineering_operation': operation,
                'features_before': features_before,
                'features_after': features_after,
                'features_added': features_after - features_before,
                'new_features': feature_names[:10] if feature_names else None
            }
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'function_call_counts': self.function_call_counts.copy(),
            'error_counts': self.error_counts.copy(),
            'total_function_calls': sum(self.function_call_counts.values()),
            'total_errors': sum(self.error_counts.values())
        }

    def reset_performance_metrics(self):
        """Reset all performance tracking metrics"""
        self.performance_metrics.clear()
        self.function_call_counts.clear()
        self.error_counts.clear()
        self.info("Performance metrics reset")

    @contextmanager
    def log_context(self, context_name: str, **context_data):
        """Context manager for logging with additional context"""
        # Store context in thread-local storage
        if not hasattr(self.local, 'context_stack'):
            self.local.context_stack = []

        self.local.context_stack.append({
            'name': context_name,
            'data': context_data,
            'start_time': time.time()
        })

        self.info(f"Entering context: {context_name}", extra=context_data)

        try:
            yield
        except Exception as e:
            self.error(f"Error in context {context_name}: {str(e)}", extra=context_data)
            raise
        finally:
            context = self.local.context_stack.pop()
            duration = time.time() - context['start_time']
            self.info(f"Exiting context: {context_name} (duration: {duration:.4f}s)", 
                     extra={**context_data, 'context_duration': duration})


# Global logger instance
_default_logger = None
_logger_lock = threading.Lock()


def get_logger(name: str = "credit_default") -> CreditDefaultLogger:
    """
    Get a logger instance (thread-safe singleton pattern).

    Args:
        name: Logger name

    Returns:
        CreditDefaultLogger instance
    """
    global _default_logger

    with _logger_lock:
        if _default_logger is None or _default_logger.name != name:
            _default_logger = CreditDefaultLogger(name=name)

    return _default_logger


def setup_logging(config: dict = None) -> CreditDefaultLogger:
    """
    Setup logging with configuration.

    Args:
        config: Logging configuration dictionary

    Returns:
        Configured CreditDefaultLogger instance
    """
    config = config or {}

    logger_config = {
        'name': config.get('name', 'credit_default'),
        'level': config.get('level', 'INFO'),
        'log_dir': config.get('log_dir', 'logs'),
        'log_file': config.get('log_file', 'app.log'),
        'max_file_size': config.get('max_file_size', 10 * 1024 * 1024),
        'backup_count': config.get('backup_count', 5),
        'console_output': config.get('console_output', True),
        'file_output': config.get('file_output', True),
        'json_format': config.get('json_format', False),
        'performance_logging': config.get('performance_logging', True),
        'ml_logging': config.get('ml_logging', True)
    }

    return CreditDefaultLogger(**logger_config)


def log_execution_time(logger: Optional[CreditDefaultLogger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use

    Returns:
        Decorated function
    """
    if logger is None:
        logger = get_logger()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.log_performance(
                    operation=func_name,
                    duration=duration,
                    success=True,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                logger.log_performance(
                    operation=func_name,
                    duration=duration,
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__
                )

                logger.exception(f"Error in {func_name}: {str(e)}")
                raise

        return wrapper
    return decorator


def log_function_calls(logger: Optional[CreditDefaultLogger] = None):
    """
    Decorator to log function calls with arguments and results.

    Args:
        logger: Logger instance to use

    Returns:
        Decorated function
    """
    if logger is None:
        logger = get_logger()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            logger.log_function_call(
                func_name=func_name,
                args=args,
                kwargs=kwargs
            )

            try:
                result = func(*args, **kwargs)

                logger.log_function_call(
                    func_name=func_name,
                    result=result
                )

                return result

            except Exception as e:
                logger.exception(f"Error in {func_name}: {str(e)}")
                raise

        return wrapper
    return decorator


def log_ml_operation(operation_type: str, logger: Optional[CreditDefaultLogger] = None):
    """
    Decorator to log ML operations with timing and metadata.

    Args:
        operation_type: Type of ML operation (training, prediction, etc.)
        logger: Logger instance to use

    Returns:
        Decorated function
    """
    if logger is None:
        logger = get_logger()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"

            logger.info(
                f"Starting ML operation: {operation_type}",
                extra={
                    'ml_operation': True,
                    'operation_type': operation_type,
                    'function': func_name,
                    'start_time': datetime.now().isoformat()
                }
            )

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    f"Completed ML operation: {operation_type}",
                    extra={
                        'ml_operation': True,
                        'operation_type': operation_type,
                        'function': func_name,
                        'duration': duration,
                        'success': True,
                        'end_time': datetime.now().isoformat()
                    }
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                logger.error(
                    f"Failed ML operation: {operation_type}",
                    extra={
                        'ml_operation': True,
                        'operation_type': operation_type,
                        'function': func_name,
                        'duration': duration,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'end_time': datetime.now().isoformat()
                    }
                )

                raise

        return wrapper
    return decorator


class LoggerContextManager:
    """Context manager for temporary logger configuration"""

    def __init__(self, logger: CreditDefaultLogger, level: str = None, **context):
        self.logger = logger
        self.original_level = logger.logger.level
        self.new_level = getattr(logging, level.upper()) if level else None
        self.context = context

    def __enter__(self):
        if self.new_level:
            self.logger.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.logger.setLevel(self.original_level)

        if exc_type:
            self.logger.exception(
                f"Exception in logger context: {exc_type.__name__}",
                extra={**self.context, 'exception_value': str(exc_val)}
            )


def with_logging_context(level: str = None, **context):
    """
    Context manager factory for temporary logging configuration.

    Args:
        level: Temporary logging level
        **context: Additional context to include in logs

    Returns:
        Context manager
    """
    logger = get_logger()
    return LoggerContextManager(logger, level, **context)


if __name__ == "__main__":
    # Example usage and testing
    print("Credit Default Logger Test")
    print("=" * 50)

    # Initialize logger
    logger = CreditDefaultLogger(
        name="test_logger",
        level="DEBUG",
        console_output=True,
        file_output=True,
        performance_logging=True,
        ml_logging=True
    )

    # Test basic logging
    logger.info("Logger initialized successfully")
    logger.debug("Debug message")
    logger.warning("Warning message")

    # Test performance logging
    import time
    start = time.time()
    time.sleep(0.1)  # Simulate work
    logger.log_performance("test_operation", time.time() - start, test_metric=42)

    # Test ML operation logging
    logger.log_model_metrics("TestModel", {"accuracy": 0.95, "precision": 0.92})
    logger.log_data_info("test_dataset", (1000, 20), ["feature1", "feature2"])
    logger.log_prediction("individual", {"feature1": 1}, "positive", 0.85, "TestModel")

    # Test context manager
    with logger.log_context("test_context", user_id=123):
        logger.info("Inside context")

    # Test decorators
    @log_execution_time(logger)
    @log_ml_operation("test_training", logger)
    def test_function(x, y=10):
        time.sleep(0.05)  # Simulate work
        return x + y

    result = test_function(5, y=15)
    logger.info(f"Test function result: {result}")

    # Get performance summary
    summary = logger.get_performance_summary()
    logger.info(f"Performance summary: {summary}")

    print("\n✅ Logger testing completed successfully!")
    print("📁 Check the 'logs' directory for log files:")
    print("   • app.log - Main application logs")
    print("   • performance.log - Performance metrics")
    print("   • ml_operations.log - ML operation logs")
