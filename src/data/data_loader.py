"""
Data Loader Module

This module handles loading and initial processing of the UCI Credit Default Dataset.
It provides functions to download, load, and perform initial validation of the dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import requests
from io import BytesIO
import logging
from datetime import datetime
import warnings

# Import from our utils modules
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.helpers import (
    ensure_directory_exists, 
    download_file, 
    validate_dataframe,
    get_file_hash,
    format_file_size
)

# Configure logging
logger = get_logger(__name__)
config = get_config()


class UCICreditDefaultLoader:
    """
    Loader class for the UCI Credit Default Dataset.

    This class handles downloading, loading, and initial validation of the 
    UCI Credit Default Dataset from the UCI Machine Learning Repository.
    """

    def __init__(self, data_dir: Optional[str] = None, force_download: bool = False):
        """
        Initialize the UCI Credit Default Dataset loader.

        Args:
            data_dir: Directory to store the dataset (uses config if None)
            force_download: Whether to force re-download even if file exists
        """
        self.data_dir = Path(data_dir) if data_dir else Path(config.data.raw_data_path)
        self.force_download = force_download

        # Dataset information
        self.dataset_url = config.data.dataset_url
        self.dataset_filename = config.data.dataset_name
        self.dataset_path = self.data_dir / self.dataset_filename

        # Ensure data directory exists
        ensure_directory_exists(self.data_dir)

        # Dataset metadata
        self.expected_columns = [
            'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
            'default.payment.next.month'
        ]

        self.expected_shape = (30000, 25)  # Expected dataset dimensions

        logger.info(f"UCI Credit Default Loader initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Dataset URL: {self.dataset_url}")

    def download_dataset(self) -> bool:
        """
        Download the UCI Credit Default Dataset.

        Returns:
            True if download successful, False otherwise
        """
        try:
            # Check if file already exists and force_download is False
            if self.dataset_path.exists() and not self.force_download:
                file_size = self.dataset_path.stat().st_size
                logger.info(f"Dataset already exists: {self.dataset_path}")
                logger.info(f"File size: {format_file_size(file_size)}")
                return True

            logger.info(f"Downloading UCI Credit Default Dataset...")
            logger.info(f"URL: {self.dataset_url}")
            logger.info(f"Destination: {self.dataset_path}")

            # Download the file
            success = download_file(
                url=self.dataset_url,
                output_path=self.dataset_path,
                timeout=60  # Longer timeout for large file
            )

            if success:
                file_size = self.dataset_path.stat().st_size
                logger.info(f"Download completed successfully")
                logger.info(f"File size: {format_file_size(file_size)}")

                # Calculate and log file hash for integrity
                file_hash = get_file_hash(self.dataset_path)
                if file_hash:
                    logger.info(f"File hash (MD5): {file_hash}")

                return True
            else:
                logger.error("Failed to download dataset")
                return False

        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return False

    def load_dataset(self, validate: bool = True) -> Optional[pd.DataFrame]:
        """
        Load the UCI Credit Default Dataset from file.

        Args:
            validate: Whether to perform validation after loading

        Returns:
            DataFrame containing the dataset or None if loading fails
        """
        try:
            # Check if file exists, download if not
            if not self.dataset_path.exists():
                logger.info("Dataset file not found, attempting to download...")
                if not self.download_dataset():
                    logger.error("Failed to download dataset")
                    return None

            logger.info(f"Loading dataset from: {self.dataset_path}")

            # Load the Excel file
            # The UCI dataset is in Excel format with specific structure
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Try different methods to load the Excel file
                try:
                    # Method 1: Direct pandas read_excel
                    df = pd.read_excel(self.dataset_path, header=1, engine='xlrd')
                    logger.info("Dataset loaded using xlrd engine")
                except Exception as e1:
                    logger.warning(f"xlrd engine failed: {str(e1)}")
                    try:
                        # Method 2: Try openpyxl engine
                        df = pd.read_excel(self.dataset_path, header=1, engine='openpyxl')
                        logger.info("Dataset loaded using openpyxl engine")
                    except Exception as e2:
                        logger.warning(f"openpyxl engine failed: {str(e2)}")
                        try:
                            # Method 3: Try without specifying engine
                            df = pd.read_excel(self.dataset_path, header=1)
                            logger.info("Dataset loaded using default engine")
                        except Exception as e3:
                            logger.error(f"All Excel loading methods failed: {str(e3)}")
                            return None

            # Log basic information about loaded dataset
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            # Perform validation if requested
            if validate:
                validation_result = self.validate_dataset(df)
                if not validation_result['is_valid']:
                    logger.warning("Dataset validation failed")
                    for error in validation_result['errors']:
                        logger.error(f"Validation error: {error}")
                    for warning in validation_result['warnings']:
                        logger.warning(f"Validation warning: {warning}")
                else:
                    logger.info("Dataset validation passed")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the loaded dataset structure and content.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating dataset structure and content...")

        # Use the general validation function from helpers
        validation_result = validate_dataframe(
            df=df,
            required_columns=self.expected_columns,
            min_rows=25000,  # Allow some tolerance
            max_rows=35000
        )

        # Additional UCI-specific validations
        try:
            # Check target variable
            if 'default.payment.next.month' in df.columns:
                target_values = df['default.payment.next.month'].unique()
                if not set(target_values).issubset({0, 1}):
                    validation_result['errors'].append(
                        f"Target variable contains invalid values: {target_values}"
                    )

                # Check target distribution
                target_dist = df['default.payment.next.month'].value_counts()
                default_rate = target_dist.get(1, 0) / len(df)
                logger.info(f"Default rate: {default_rate:.3f}")

                if default_rate < 0.1 or default_rate > 0.5:
                    validation_result['warnings'].append(
                        f"Unusual default rate: {default_rate:.3f}"
                    )

            # Check demographic variables
            if 'SEX' in df.columns:
                sex_values = df['SEX'].unique()
                if not set(sex_values).issubset({1, 2}):
                    validation_result['warnings'].append(
                        f"SEX variable contains unexpected values: {sex_values}"
                    )

            if 'EDUCATION' in df.columns:
                edu_values = df['EDUCATION'].unique()
                expected_edu = {1, 2, 3, 4, 5, 6, 0}  # Including 0 for others
                if not set(edu_values).issubset(expected_edu):
                    validation_result['warnings'].append(
                        f"EDUCATION variable contains unexpected values: {edu_values}"
                    )

            if 'MARRIAGE' in df.columns:
                marriage_values = df['MARRIAGE'].unique()
                expected_marriage = {1, 2, 3, 0}  # Including 0 for others
                if not set(marriage_values).issubset(expected_marriage):
                    validation_result['warnings'].append(
                        f"MARRIAGE variable contains unexpected values: {marriage_values}"
                    )

            # Check for reasonable age range
            if 'AGE' in df.columns:
                age_min, age_max = df['AGE'].min(), df['AGE'].max()
                if age_min < 18 or age_max > 100:
                    validation_result['warnings'].append(
                        f"AGE variable has unusual range: {age_min} to {age_max}"
                    )

            # Check payment status variables
            pay_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            for col in pay_columns:
                if col in df.columns:
                    pay_values = df[col].unique()
                    # Payment status can range from -2 to 8
                    if df[col].min() < -2 or df[col].max() > 8:
                        validation_result['warnings'].append(
                            f"{col} variable has values outside expected range: {pay_values}"
                        )

            # Check for negative amounts in bill and payment amounts
            amount_columns = [col for col in df.columns if 'AMT' in col]
            for col in amount_columns:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validation_result['warnings'].append(
                            f"{col} has {negative_count} negative values"
                        )

        except Exception as e:
            validation_result['errors'].append(f"Error during UCI-specific validation: {str(e)}")

        return validation_result

    def get_dataset_info(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.

        Args:
            df: DataFrame to analyze (loads if None)

        Returns:
            Dictionary with dataset information
        """
        if df is None:
            df = self.load_dataset()
            if df is None:
                return {"error": "Failed to load dataset"}

        try:
            info = {
                'basic_info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'file_size': format_file_size(self.dataset_path.stat().st_size) if self.dataset_path.exists() else 'N/A'
                },
                'data_quality': {
                    'missing_values': df.isnull().sum().to_dict(),
                    'duplicate_rows': df.duplicated().sum(),
                    'total_missing': df.isnull().sum().sum()
                },
                'target_variable': {},
                'feature_summary': {}
            }

            # Target variable analysis
            if 'default.payment.next.month' in df.columns:
                target = df['default.payment.next.month']
                info['target_variable'] = {
                    'name': 'default.payment.next.month',
                    'type': 'binary',
                    'distribution': target.value_counts().to_dict(),
                    'default_rate': target.mean(),
                    'missing_count': target.isnull().sum()
                }

            # Feature summary
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

            info['feature_summary'] = {
                'total_features': len(df.columns),
                'numeric_features': len(numeric_features),
                'categorical_features': len(categorical_features),
                'numeric_feature_names': numeric_features,
                'categorical_feature_names': categorical_features
            }

            # Statistical summary for numeric features
            if numeric_features:
                numeric_summary = df[numeric_features].describe()
                info['numeric_statistics'] = numeric_summary.to_dict()

            return info

        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return {"error": str(e)}

    def save_processed_dataset(self, df: pd.DataFrame, filename: str = "processed_credit_default.csv") -> bool:
        """
        Save processed dataset to the processed data directory.

        Args:
            df: DataFrame to save
            filename: Name of the output file

        Returns:
            True if successful, False otherwise
        """
        try:
            processed_dir = Path(config.data.processed_data_path)
            ensure_directory_exists(processed_dir)

            output_path = processed_dir / filename

            # Save as CSV for better compatibility
            df.to_csv(output_path, index=False)

            file_size = output_path.stat().st_size
            logger.info(f"Processed dataset saved to: {output_path}")
            logger.info(f"File size: {format_file_size(file_size)}")

            return True

        except Exception as e:
            logger.error(f"Error saving processed dataset: {str(e)}")
            return False

    def load_processed_dataset(self, filename: str = "processed_credit_default.csv") -> Optional[pd.DataFrame]:
        """
        Load processed dataset from the processed data directory.

        Args:
            filename: Name of the file to load

        Returns:
            DataFrame or None if loading fails
        """
        try:
            processed_dir = Path(config.data.processed_data_path)
            file_path = processed_dir / filename

            if not file_path.exists():
                logger.warning(f"Processed dataset not found: {file_path}")
                return None

            df = pd.read_csv(file_path)
            logger.info(f"Processed dataset loaded from: {file_path}")
            logger.info(f"Shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Error loading processed dataset: {str(e)}")
            return None


def load_credit_default_data(data_dir: Optional[str] = None, 
                           force_download: bool = False,
                           validate: bool = True) -> Optional[pd.DataFrame]:
    """
    Convenience function to load the UCI Credit Default Dataset.

    Args:
        data_dir: Directory containing the dataset
        force_download: Whether to force re-download
        validate: Whether to validate the dataset

    Returns:
        DataFrame containing the dataset or None if loading fails
    """
    loader = UCICreditDefaultLoader(data_dir=data_dir, force_download=force_download)
    return loader.load_dataset(validate=validate)


def get_dataset_info(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get dataset information.

    Args:
        data_dir: Directory containing the dataset

    Returns:
        Dictionary with dataset information
    """
    loader = UCICreditDefaultLoader(data_dir=data_dir)
    return loader.get_dataset_info()


def download_dataset(data_dir: Optional[str] = None, force_download: bool = False) -> bool:
    """
    Convenience function to download the dataset.

    Args:
        data_dir: Directory to save the dataset
        force_download: Whether to force re-download

    Returns:
        True if successful, False otherwise
    """
    loader = UCICreditDefaultLoader(data_dir=data_dir, force_download=force_download)
    return loader.download_dataset()


if __name__ == "__main__":
    # Example usage and testing
    print("UCI Credit Default Dataset Loader Test")
    print("=" * 50)

    # Initialize loader
    loader = UCICreditDefaultLoader()

    # Download dataset
    print("\nDownloading dataset...")
    download_success = loader.download_dataset()
    print(f"Download successful: {download_success}")

    if download_success:
        # Load dataset
        print("\nLoading dataset...")
        df = loader.load_dataset()

        if df is not None:
            print(f"Dataset loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Get dataset info
            print("\nGetting dataset information...")
            info = loader.get_dataset_info(df)

            print("\nDataset Information:")
            print(f"Shape: {info['basic_info']['shape']}")
            print(f"Memory usage: {info['basic_info']['memory_usage_mb']:.2f} MB")
            print(f"Missing values: {info['data_quality']['total_missing']}")
            print(f"Default rate: {info['target_variable'].get('default_rate', 'N/A'):.3f}")

            # Save processed dataset
            print("\nSaving processed dataset...")
            save_success = loader.save_processed_dataset(df)
            print(f"Save successful: {save_success}")

        else:
            print("Failed to load dataset")
    else:
        print("Failed to download dataset")

    print("\n✅ Data loader testing completed!")
