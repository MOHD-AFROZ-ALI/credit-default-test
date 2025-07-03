"""
Data Preprocessor Module

This module handles comprehensive data cleaning, preprocessing, and preparation for machine learning.
It includes functions for handling missing values, outliers, data types, feature scaling, and data splitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path
import warnings

# Import from our utils modules
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.helpers import (
    handle_missing_values,
    detect_outliers,
    normalize_column_names,
    validate_dataframe,
    safe_cast,
    save_json,
    load_json,
    safe_divide
)

# Configure logging
logger = get_logger(__name__)
config = get_config()


class CreditDefaultPreprocessor:
    """
    Comprehensive data preprocessor for the Credit Default Dataset.
    
    This class handles all aspects of data preprocessing including:
    - Missing value handling
    - Outlier detection and treatment
    - Data type conversions
    - Feature scaling and normalization
    - Feature engineering
    - Data validation and quality checks
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state or config.data.random_state
        self.preprocessing_steps = []
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.preprocessing_config = {}
        self.feature_names = []
        self.target_column = "default.payment.next.month"
        
        # Define expected feature types for UCI Credit Default Dataset
        self.feature_types = {
            "ID": "identifier",
            "LIMIT_BAL": "numeric",
            "SEX": "categorical",
            "EDUCATION": "categorical",
            "MARRIAGE": "categorical",
            "AGE": "numeric",
            "PAY_0": "categorical",
            "PAY_2": "categorical",
            "PAY_3": "categorical",
            "PAY_4": "categorical",
            "PAY_5": "categorical",
            "PAY_6": "categorical",
            "BILL_AMT1": "numeric",
            "BILL_AMT2": "numeric",
            "BILL_AMT3": "numeric",
            "BILL_AMT4": "numeric",
            "BILL_AMT5": "numeric",
            "BILL_AMT6": "numeric",
            "PAY_AMT1": "numeric",
            "PAY_AMT2": "numeric",
            "PAY_AMT3": "numeric",
            "PAY_AMT4": "numeric",
            "PAY_AMT5": "numeric",
            "PAY_AMT6": "numeric",
            "default.payment.next.month": "target"
        }
        
        logger.info("Credit Default Preprocessor initialized")
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        logger.info("Cleaning column names...")
        
        # Store original column names for reference
        original_columns = df.columns.tolist()
        
        # Create a mapping for specific UCI dataset columns
        column_mapping = {
            "default.payment.next.month": "default_payment_next_month",
            "PAY_0": "PAY_1",  # Rename PAY_0 to PAY_1 for consistency
        }
        
        # Apply the mapping first
        df_cleaned = df.rename(columns=column_mapping)
        
        # Then normalize all column names
        df_cleaned = normalize_column_names(df_cleaned, method="snake_case")
        
        # Update target column name
        if "default_payment_next_month" in df_cleaned.columns:
            self.target_column = "default_payment_next_month"
        
        # Log the changes
        new_columns = df_cleaned.columns.tolist()
        if original_columns != new_columns:
            logger.info("Column names changed:")
            for old, new in zip(original_columns, new_columns):
                if old != new:
                    logger.info(f"  {old} -> {new}")
        
        self.preprocessing_steps.append("clean_column_names")
        return df_cleaned
    
    def handle_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        logger.info("Converting data types...")
        df_typed = df.copy()
        
        try:
            # Convert ID to string (not used for modeling)
            if "id" in df_typed.columns:
                df_typed["id"] = df_typed["id"].astype(str)
            
            # Convert categorical variables
            categorical_columns = ["sex", "education", "marriage"] + [
                f"pay_{i}" for i in range(1, 7)
            ]
            
            for col in categorical_columns:
                if col in df_typed.columns:
                    # Convert to category but keep as numeric for now
                    df_typed[col] = pd.to_numeric(df_typed[col], errors="coerce")
                    logger.debug(f"Converted {col} to numeric (categorical)")
            
            # Convert numeric variables
            numeric_columns = ["limit_bal", "age"] + [
                f"bill_amt{i}" for i in range(1, 7)
            ] + [f"pay_amt{i}" for i in range(1, 7)]
            
            for col in numeric_columns:
                if col in df_typed.columns:
                    df_typed[col] = pd.to_numeric(df_typed[col], errors="coerce")
                    logger.debug(f"Converted {col} to numeric")
            
            # Convert target variable
            if self.target_column in df_typed.columns:
                df_typed[self.target_column] = pd.to_numeric(df_typed[self.target_column], errors="coerce").astype(int)
            
            logger.info("Data type conversion completed")
            self.preprocessing_steps.append("handle_data_types")
            
        except Exception as e:
            logger.error(f"Error in data type conversion: {str(e)}")
            
        return df_typed
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Found {total_missing} missing values across {(missing_counts > 0).sum()} columns")
        
        df_imputed = df.copy()
        
        if strategy == "auto":
            # Automatic strategy based on column type and missing percentage
            for col in df_imputed.columns:
                missing_pct = df_imputed[col].isnull().sum() / len(df_imputed)
                
                if missing_pct == 0:
                    continue
                
                logger.info(f"Column {col}: {missing_pct:.2%} missing")
                
                if missing_pct > 0.5:
                    # Drop columns with >50% missing values
                    logger.warning(f"Dropping column {col} due to high missing percentage: {missing_pct:.2%}")
                    df_imputed = df_imputed.drop(columns=[col])
                elif df_imputed[col].dtype in ["int64", "float64"]:
                    # Use median for numeric columns
                    median_val = df_imputed[col].median()
                    df_imputed[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled {col} missing values with median: {median_val}")
                else:
                    # Use mode for categorical columns
                    mode_value = df_imputed[col].mode()
                    if not mode_value.empty:
                        df_imputed[col].fillna(mode_value[0], inplace=True)
                        logger.info(f"Filled {col} missing values with mode: {mode_value[0]}")
        
        elif strategy == "knn":
            # Use KNN imputation for numeric columns
            numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                imputer = KNNImputer(n_neighbors=5, weights="uniform")
                df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])
                self.imputers["knn"] = imputer
                logger.info("Applied KNN imputation to numeric columns")
        
        else:
            # Use the helper function for other strategies
            df_imputed = handle_missing_values(df_imputed, strategy=strategy)
        
        # Log results
        remaining_missing = df_imputed.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {remaining_missing}")
        
        self.preprocessing_steps.append(f"handle_missing_values_{strategy}")
        return df_imputed
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering new features...")
        
        df_features = df.copy()
        new_features = []
        
        try:
            # Credit utilization features
            if "limit_bal" in df_features.columns:
                for i in range(1, 7):
                    bill_col = f"bill_amt{i}"
                    if bill_col in df_features.columns:
                        util_col = f"credit_util_{i}"
                        df_features[util_col] = df_features[bill_col] / (df_features["limit_bal"] + 1)
                        new_features.append(util_col)
            
            # Payment ratio features
            for i in range(1, 7):
                bill_col = f"bill_amt{i}"
                pay_col = f"pay_amt{i}"
                if bill_col in df_features.columns and pay_col in df_features.columns:
                    ratio_col = f"payment_ratio_{i}"
                    df_features[ratio_col] = df_features[pay_col] / (df_features[bill_col] + 1)
                    new_features.append(ratio_col)
            
            # Average bill amount
            bill_columns = [f"bill_amt{i}" for i in range(1, 7) if f"bill_amt{i}" in df_features.columns]
            if bill_columns:
                df_features["avg_bill_amt"] = df_features[bill_columns].mean(axis=1)
                new_features.append("avg_bill_amt")
            
            # Average payment amount
            pay_columns = [f"pay_amt{i}" for i in range(1, 7) if f"pay_amt{i}" in df_features.columns]
            if pay_columns:
                df_features["avg_pay_amt"] = df_features[pay_columns].mean(axis=1)
                new_features.append("avg_pay_amt")
            
            # Payment delay count (count of positive payment status)
            pay_status_cols = [f"pay_{i}" for i in range(1, 7) if f"pay_{i}" in df_features.columns]
            if pay_status_cols:
                df_features["delay_count"] = (df_features[pay_status_cols] > 0).sum(axis=1)
                new_features.append("delay_count")
            
            # Maximum payment delay
            if pay_status_cols:
                df_features["max_delay"] = df_features[pay_status_cols].max(axis=1)
                new_features.append("max_delay")
            
            logger.info(f"Created {len(new_features)} new features: {new_features}")
            self.preprocessing_config["engineered_features"] = new_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
        
        self.preprocessing_steps.append("engineer_features")
        return df_features
    
    def scale_features(self, df: pd.DataFrame, method: str = "standard", 
                      exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            method: Scaling method ("standard", "minmax", "robust")
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using method: {method}")
        
        df_scaled = df.copy()
        exclude_columns = exclude_columns or ["id", self.target_column]
        
        # Get numeric columns to scale
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
        
        if not columns_to_scale:
            logger.warning("No numeric columns found for scaling")
            return df_scaled
        
        # Choose scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        self.preprocessing_config["scaled_columns"] = columns_to_scale
        
        logger.info(f"Scaled {len(columns_to_scale)} columns using {method} scaling")
        self.preprocessing_steps.append(f"scale_features_{method}")
        return df_scaled
    
    def split_data(self, df: pd.DataFrame, test_size: float = None, 
                  validation_size: float = None, stratify: bool = True) -> Tuple[pd.DataFrame, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Size of test set (uses config if None)
            validation_size: Size of validation set (uses config if None)
            stratify: Whether to stratify split based on target variable
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = test_size or config.data.test_size
        validation_size = validation_size or config.data.validation_size
        
        logger.info(f"Splitting data: test_size={test_size}, validation_size={validation_size}, stratify={stratify}")
        
        # Separate features and target
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Remove ID column if present
        if "id" in X.columns:
            X = X.drop(columns=["id"])
        
        stratify_param = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_temp
        )
        
        # Log split information
        logger.info(f"Data split completed:")
        logger.info(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.2%})")
        logger.info(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df):.2%})")
        logger.info(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.2%})")
        
        if stratify:
            logger.info("Target distribution:")
            logger.info(f"  Train: {y_train.value_counts(normalize=True).to_dict()}")
            logger.info(f"  Validation: {y_val.value_counts(normalize=True).to_dict()}")
            logger.info(f"  Test: {y_test.value_counts(normalize=True).to_dict()}")
        
        self.feature_names = X_train.columns.tolist()
        self.preprocessing_steps.append("split_data")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate processed data quality.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating processed data...")
        
        validation_result = validate_dataframe(df, min_rows=1000)
        
        # Additional checks for processed data
        try:
            # Check for infinite values
            inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_counts > 0:
                validation_result["warnings"].append(f"Found {inf_counts} infinite values")
            
            # Check for very high/low values that might indicate scaling issues
            numeric_df = df.select_dtypes(include=[np.number])
            for col in numeric_df.columns:
                if col != self.target_column:
                    col_min, col_max = numeric_df[col].min(), numeric_df[col].max()
                    if abs(col_min) > 1000 or abs(col_max) > 1000:
                        validation_result["warnings"].append(
                            f"Column {col} has extreme values: [{col_min:.2f}, {col_max:.2f}]"
                        )
            
            # Check target variable distribution
            if self.target_column in df.columns:
                target_dist = df[self.target_column].value_counts(normalize=True)
                minority_class_pct = target_dist.min()
                if minority_class_pct < 0.05:
                    validation_result["warnings"].append(
                        f"Severe class imbalance: minority class = {minority_class_pct:.2%}"
                    )
            
        except Exception as e:
            validation_result["errors"].append(f"Error in processed data validation: {str(e)}")
        
        return validation_result
    
    def save_preprocessing_config(self, filepath: str) -> bool:
        """
        Save preprocessing configuration to file.
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        config_data = {
            "preprocessing_steps": self.preprocessing_steps,
            "preprocessing_config": self.preprocessing_config,
            "feature_names": self.feature_names,
            "target_column": self.target_column,
            "random_state": self.random_state
        }
        
        return save_json(config_data, filepath)
    
    def load_preprocessing_config(self, filepath: str) -> bool:
        """
        Load preprocessing configuration from file.
        
        Args:
            filepath: Path to load configuration from
            
        Returns:
            True if successful, False otherwise
        """
        config_data = load_json(filepath)
        if config_data:
            self.preprocessing_steps = config_data.get("preprocessing_steps", [])
            self.preprocessing_config = config_data.get("preprocessing_config", {})
            self.feature_names = config_data.get("feature_names", [])
            self.target_column = config_data.get("target_column", "default.payment.next.month")
            self.random_state = config_data.get("random_state", 42)
            return True
        return False
    
    def preprocess_full_pipeline(self, df: pd.DataFrame, 
                                include_feature_engineering: bool = True,
                                scaling_method: str = "standard",
                                outlier_method: str = "iqr",
                                outlier_action: str = "cap") -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            include_feature_engineering: Whether to include feature engineering
            scaling_method: Method for feature scaling
            outlier_method: Method for outlier detection
            outlier_action: Action for outlier handling
            
        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("Starting full preprocessing pipeline...")
        
        # Step 1: Clean column names
        df_processed = self.clean_column_names(df)
        
        # Step 2: Handle data types
        df_processed = self.handle_data_types(df_processed)
        
        # Step 3: Handle missing values
        df_processed = self.handle_missing_values(df_processed, strategy="auto")
        
        # Step 4: Handle outliers
        df_processed = self.handle_outliers(
            df_processed, 
            method=outlier_method, 
            action=outlier_action
        )
        
        # Step 5: Feature engineering (optional)
        if include_feature_engineering:
            df_processed = self.engineer_features(df_processed)
        
        # Step 6: Scale features
        df_processed = self.scale_features(df_processed, method=scaling_method)
        
        # Step 7: Validate processed data
        validation_result = self.validate_processed_data(df_processed)
        if not validation_result["is_valid"]:
            logger.warning("Processed data validation failed")
            for error in validation_result["errors"]:
                logger.error(f"Validation error: {error}")
        
        logger.info("Full preprocessing pipeline completed")
        logger.info(f"Final shape: {df_processed.shape}")
        logger.info(f"Preprocessing steps: {self.preprocessing_steps}")
        
        return df_processed
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps and configuration.
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            "preprocessing_steps": self.preprocessing_steps,
            "preprocessing_config": self.preprocessing_config,
            "feature_names": self.feature_names,
            "target_column": self.target_column,
            "random_state": self.random_state,
            "scalers": list(self.scalers.keys()),
            "encoders": list(self.encoders.keys()),
            "imputers": list(self.imputers.keys())
        }


# Utility functions for preprocessing

def preprocess_credit_default_data(df: pd.DataFrame, 
                                  include_feature_engineering: bool = True,
                                  scaling_method: str = "standard",
                                  test_size: float = 0.2,
                                  validation_size: float = 0.2,
                                  random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
    """
    Convenience function to preprocess credit default data with train/val/test split.
    
    Args:
        df: Input DataFrame
        include_feature_engineering: Whether to include feature engineering
        scaling_method: Method for feature scaling
        test_size: Size of test set
        validation_size: Size of validation set
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    preprocessor = CreditDefaultPreprocessor(random_state=random_state)
    
    # Run full preprocessing pipeline
    df_processed = preprocessor.preprocess_full_pipeline(
        df,
        include_feature_engineering=include_feature_engineering,
        scaling_method=scaling_method
    )
    
    # Split the data
    return preprocessor.split_data(
        df_processed,
        test_size=test_size,
        validation_size=validation_size
    )


def create_preprocessing_pipeline(numeric_features: List[str], 
                                categorical_features: List[str],
                                scaling_method: str = "standard") -> Pipeline:
    """
    Create a scikit-learn preprocessing pipeline.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        scaling_method: Scaling method for numeric features
        
    Returns:
        Scikit-learn Pipeline object
    """
    # Choose scaler
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif scaling_method == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    return Pipeline(steps=[("preprocessor", preprocessor)])


def get_feature_types(df: pd.DataFrame, target_column: str = "default_payment_next_month") -> Dict[str, List[str]]:
    """
    Automatically detect feature types in the dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column to exclude
        
    Returns:
        Dictionary with lists of numeric and categorical features
    """
    # Exclude target and ID columns
    exclude_columns = [target_column, "id", "ID"]
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    numeric_features = []
    categorical_features = []
    
    for col in feature_columns:
        if df[col].dtype in ["int64", "float64"]:
            # Check if it should be treated as categorical
            if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    return {
        "numeric": numeric_features,
        "categorical": categorical_features
    }


def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality score and metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data quality metrics
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Calculate completeness score (0-100)
    completeness_score = ((total_cells - missing_cells) / total_cells) * 100
    
    # Calculate uniqueness score (0-100)
    uniqueness_score = ((df.shape[0] - duplicate_rows) / df.shape[0]) * 100
    
    # Calculate consistency score (based on data types)
    consistency_issues = 0
    for col in df.columns:
        if df[col].dtype == "object":
            # Check for mixed types in object columns
            try:
                pd.to_numeric(df[col], errors="raise")
            except:
                consistency_issues += 1
    
    consistency_score = max(0, (len(df.columns) - consistency_issues) / len(df.columns) * 100)
    
    # Overall data quality score
    overall_score = (completeness_score + uniqueness_score + consistency_score) / 3
    
    return {
        "overall_score": round(overall_score, 2),
        "completeness_score": round(completeness_score, 2),
        "uniqueness_score": round(uniqueness_score, 2),
        "consistency_score": round(consistency_score, 2),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "total_cells": total_cells,
        "missing_percentage": round((missing_cells / total_cells) * 100, 2),
        "duplicate_percentage": round((duplicate_rows / df.shape[0]) * 100, 2)
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Credit Default Preprocessor Test")
    print("=" * 50)
    
    # Create sample data for testing
    import numpy as np
    
    # Generate sample data similar to UCI Credit Default Dataset
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        "ID": range(1, n_samples + 1),
        "LIMIT_BAL": np.random.normal(50000, 20000, n_samples),
        "SEX": np.random.choice([1, 2], n_samples),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n_samples),
        "MARRIAGE": np.random.choice([1, 2, 3], n_samples),
        "AGE": np.random.randint(20, 80, n_samples),
        "PAY_0": np.random.choice([-1, 0, 1, 2, 3], n_samples),
        "BILL_AMT1": np.random.normal(10000, 5000, n_samples),
        "PAY_AMT1": np.random.normal(2000, 1000, n_samples),
        "default.payment.next.month": np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    # Add some missing values
    sample_data["EDUCATION"][np.random.choice(n_samples, 50, replace=False)] = np.nan
    sample_data["BILL_AMT1"][np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    df_sample = pd.DataFrame(sample_data)
    
    print(f"Sample data shape: {df_sample.shape}")
    print(f"Missing values: {df_sample.isnull().sum().sum()}")
    
    # Initialize preprocessor
    preprocessor = CreditDefaultPreprocessor(random_state=42)
    
    # Test full preprocessing pipeline
    print("\nRunning full preprocessing pipeline...")
    df_processed = preprocessor.preprocess_full_pipeline(
        df_sample,
        include_feature_engineering=True,
        scaling_method="standard"
    )
    
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Missing values after processing: {df_processed.isnull().sum().sum()}")
    
    # Test data splitting
    print("\nTesting data splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Test data quality calculation
    print("\nCalculating data quality score...")
    quality_score = calculate_data_quality_score(df_sample)
    print(f"Data quality score: {quality_score}")
    
    # Get preprocessing summary
    print("\nPreprocessing summary:")
    summary = preprocessor.get_preprocessing_summary()
    print(f"Steps completed: {summary['preprocessing_steps']}")
    print(f"Features created: {len(summary['feature_names'])}")
    
    print("\n✅ Preprocessor testing completed successfully!")
