"""
Test suite for data processing and validation functionality
Comprehensive tests for data loading, cleaning, validation, and preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from datetime import datetime

# Test fixtures
@pytest.fixture
def sample_credit_data():
    """Generate sample credit data for testing"""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'employment_length': np.random.exponential(5, n_samples),
        'loan_amount': np.random.lognormal(9, 0.8, n_samples).astype(int),
        'loan_purpose': np.random.choice(['home', 'auto', 'personal'], n_samples),
        'employment_status': np.random.choice(['employed', 'self_employed'], n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })

@pytest.fixture
def sample_data_with_missing():
    """Generate sample data with missing values"""
    np.random.seed(42)
    data = pd.DataFrame({
        'customer_id': range(1, 51),
        'age': np.random.randint(18, 80, 50),
        'income': np.random.lognormal(10.5, 0.5, 50).astype(int),
        'credit_score': np.random.randint(300, 850, 50),
        'default': np.random.choice([0, 1], 50, p=[0.8, 0.2])
    })

    # Introduce missing values
    data.loc[0:4, 'age'] = np.nan
    data.loc[10:14, 'income'] = np.nan
    data.loc[20:24, 'credit_score'] = np.nan

    return data

@pytest.fixture
def temp_csv_file(sample_credit_data):
    """Create temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_credit_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

class TestDataLoader:
    """Test data loading functionality"""

    def test_load_csv_file(self, temp_csv_file, sample_credit_data):
        """Test loading CSV files"""
        class DataLoader:
            def load_csv(self, file_path):
                return pd.read_csv(file_path)

        loader = DataLoader()
        loaded_data = loader.load_csv(temp_csv_file)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_credit_data)
        assert list(loaded_data.columns) == list(sample_credit_data.columns)

    def test_load_nonexistent_file(self):
        """Test handling of nonexistent files"""
        class DataLoader:
            def load_csv(self, file_path):
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                return pd.read_csv(file_path)

        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent_file.csv")

class TestDataValidator:
    """Test data validation functionality"""

    def test_validate_required_columns(self, sample_credit_data):
        """Test validation of required columns"""
        class DataValidator:
            def __init__(self, required_columns):
                self.required_columns = required_columns

            def validate_columns(self, data):
                missing_columns = set(self.required_columns) - set(data.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                return True

        required_columns = ['customer_id', 'age', 'income', 'credit_score', 'default']
        validator = DataValidator(required_columns)

        # Should pass with all required columns
        assert validator.validate_columns(sample_credit_data) is True

        # Should fail with missing columns
        incomplete_data = sample_credit_data.drop(columns=['age'])
        with pytest.raises(ValueError, match="Missing required columns"):
            validator.validate_columns(incomplete_data)

    def test_validate_data_types(self, sample_credit_data):
        """Test validation of data types"""
        class DataValidator:
            def validate_data_types(self, data):
                errors = []

                # Check numeric columns
                numeric_columns = ['age', 'income', 'credit_score', 'debt_to_income_ratio']
                for col in numeric_columns:
                    if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                        errors.append(f"Column {col} should be numeric")

                if errors:
                    raise ValueError(f"Data type validation errors: {errors}")
                return True

        validator = DataValidator()
        assert validator.validate_data_types(sample_credit_data) is True

    def test_validate_value_ranges(self, sample_credit_data):
        """Test validation of value ranges"""
        class DataValidator:
            def validate_ranges(self, data):
                errors = []

                # Age should be between 18 and 100
                if 'age' in data.columns:
                    invalid_ages = data[(data['age'] < 18) | (data['age'] > 100)]
                    if not invalid_ages.empty:
                        errors.append(f"Invalid age values: {len(invalid_ages)} records")

                # Credit score should be between 300 and 850
                if 'credit_score' in data.columns:
                    invalid_scores = data[(data['credit_score'] < 300) | (data['credit_score'] > 850)]
                    if not invalid_scores.empty:
                        errors.append(f"Invalid credit scores: {len(invalid_scores)} records")

                if errors:
                    raise ValueError(f"Range validation errors: {errors}")
                return True

        validator = DataValidator()
        assert validator.validate_ranges(sample_credit_data) is True

class TestDataCleaner:
    """Test data cleaning functionality"""

    def test_handle_missing_values_drop(self, sample_data_with_missing):
        """Test dropping rows with missing values"""
        class DataCleaner:
            def handle_missing_values(self, data, strategy='drop'):
                if strategy == 'drop':
                    return data.dropna()
                elif strategy == 'fill_median':
                    data_copy = data.copy()
                    numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        data_copy[col].fillna(data_copy[col].median(), inplace=True)
                    return data_copy
                return data

        cleaner = DataCleaner()
        original_length = len(sample_data_with_missing)
        cleaned_data = cleaner.handle_missing_values(sample_data_with_missing, strategy='drop')

        assert len(cleaned_data) < original_length
        assert cleaned_data.isnull().sum().sum() == 0

    def test_handle_missing_values_fill(self, sample_data_with_missing):
        """Test filling missing values with median"""
        class DataCleaner:
            def handle_missing_values(self, data, strategy='fill_median'):
                data_copy = data.copy()
                if strategy == 'fill_median':
                    numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        data_copy[col].fillna(data_copy[col].median(), inplace=True)
                return data_copy

        cleaner = DataCleaner()
        original_missing = sample_data_with_missing.isnull().sum().sum()
        cleaned_data = cleaner.handle_missing_values(sample_data_with_missing, strategy='fill_median')

        assert original_missing > 0
        assert cleaned_data.isnull().sum().sum() < original_missing

    def test_remove_outliers_iqr(self, sample_credit_data):
        """Test outlier removal using IQR method"""
        class DataCleaner:
            def remove_outliers_iqr(self, data, columns=None, multiplier=1.5):
                if columns is None:
                    columns = data.select_dtypes(include=[np.number]).columns

                data_clean = data.copy()

                for col in columns:
                    if col in data_clean.columns:
                        Q1 = data_clean[col].quantile(0.25)
                        Q3 = data_clean[col].quantile(0.75)
                        IQR = Q3 - Q1

                        lower_bound = Q1 - multiplier * IQR
                        upper_bound = Q3 + multiplier * IQR

                        data_clean = data_clean[
                            (data_clean[col] >= lower_bound) & 
                            (data_clean[col] <= upper_bound)
                        ]

                return data_clean

        # Add some outliers
        data_with_outliers = sample_credit_data.copy()
        data_with_outliers.loc[0, 'age'] = 150  # Impossible age
        data_with_outliers.loc[1, 'income'] = 10000000  # Extremely high income

        cleaner = DataCleaner()
        original_length = len(data_with_outliers)
        cleaned_data = cleaner.remove_outliers_iqr(
            data_with_outliers, 
            columns=['age', 'income']
        )

        # Should remove some outliers
        assert len(cleaned_data) < original_length

    def test_remove_duplicates(self, sample_credit_data):
        """Test duplicate removal"""
        class DataCleaner:
            def remove_duplicates(self, data, subset=None):
                return data.drop_duplicates(subset=subset)

        # Add some duplicates
        data_with_duplicates = pd.concat([sample_credit_data, sample_credit_data.iloc[:5]], ignore_index=True)

        cleaner = DataCleaner()
        cleaned_data = cleaner.remove_duplicates(data_with_duplicates)

        assert len(cleaned_data) == len(sample_credit_data)

class TestDataPreprocessor:
    """Test data preprocessing functionality"""

    def test_encode_categorical_variables(self, sample_credit_data):
        """Test categorical variable encoding"""
        from sklearn.preprocessing import LabelEncoder

        class DataPreprocessor:
            def encode_categorical(self, data, method='label', columns=None):
                if columns is None:
                    columns = data.select_dtypes(include=['object']).columns

                data_encoded = data.copy()

                if method == 'label':
                    for col in columns:
                        if col in data_encoded.columns:
                            le = LabelEncoder()
                            data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))

                elif method == 'onehot':
                    data_encoded = pd.get_dummies(data_encoded, columns=columns, prefix=columns)

                return data_encoded

        preprocessor = DataPreprocessor()
        categorical_columns = ['loan_purpose', 'employment_status']

        # Test label encoding
        encoded_data = preprocessor.encode_categorical(
            sample_credit_data, 
            method='label', 
            columns=categorical_columns
        )

        for col in categorical_columns:
            assert pd.api.types.is_numeric_dtype(encoded_data[col])

        # Test one-hot encoding
        onehot_data = preprocessor.encode_categorical(
            sample_credit_data, 
            method='onehot', 
            columns=categorical_columns
        )

        # Should have more columns after one-hot encoding
        assert len(onehot_data.columns) > len(sample_credit_data.columns)

    def test_scale_numerical_features(self, sample_credit_data):
        """Test numerical feature scaling"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        class DataPreprocessor:
            def scale_features(self, data, method='standard', columns=None):
                if columns is None:
                    columns = data.select_dtypes(include=[np.number]).columns

                data_scaled = data.copy()

                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {method}")

                data_scaled[columns] = scaler.fit_transform(data_scaled[columns])

                return data_scaled, scaler

        preprocessor = DataPreprocessor()
        numerical_columns = ['age', 'income', 'credit_score', 'debt_to_income_ratio']

        # Test standard scaling
        scaled_data, scaler = preprocessor.scale_features(
            sample_credit_data, 
            method='standard', 
            columns=numerical_columns
        )

        # Check that scaled features have approximately mean=0, std=1
        for col in numerical_columns:
            assert abs(scaled_data[col].mean()) < 0.1
            assert abs(scaled_data[col].std() - 1.0) < 0.1

        # Test min-max scaling
        minmax_data, minmax_scaler = preprocessor.scale_features(
            sample_credit_data, 
            method='minmax', 
            columns=numerical_columns
        )

        # Check that scaled features are in range [0, 1]
        for col in numerical_columns:
            assert minmax_data[col].min() >= 0
            assert minmax_data[col].max() <= 1

class TestDataSplitter:
    """Test data splitting functionality"""

    def test_train_test_split(self, sample_credit_data):
        """Test train-test split functionality"""
        from sklearn.model_selection import train_test_split

        class DataSplitter:
            def split_data(self, data, target_column, test_size=0.2, random_state=42):
                X = data.drop(columns=[target_column])
                y = data[target_column]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

                return X_train, X_test, y_train, y_test

        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split_data(
            sample_credit_data, 
            target_column='default', 
            test_size=0.2
        )

        # Check split proportions
        total_samples = len(sample_credit_data)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == total_samples
        assert abs(len(X_test) / total_samples - 0.2) < 0.05

class TestDataQualityMetrics:
    """Test data quality assessment functionality"""

    def test_calculate_missing_percentage(self, sample_data_with_missing):
        """Test missing value percentage calculation"""
        class DataQualityAssessor:
            def calculate_missing_percentage(self, data):
                missing_stats = {}
                for col in data.columns:
                    missing_count = data[col].isnull().sum()
                    missing_percentage = (missing_count / len(data)) * 100
                    missing_stats[col] = {
                        'missing_count': missing_count,
                        'missing_percentage': missing_percentage
                    }
                return missing_stats

        assessor = DataQualityAssessor()
        missing_stats = assessor.calculate_missing_percentage(sample_data_with_missing)

        # Check that missing percentages are calculated correctly
        assert 'age' in missing_stats
        assert missing_stats['age']['missing_count'] == 5
        assert missing_stats['age']['missing_percentage'] == 10.0  # 5/50 * 100

    def test_detect_duplicates(self, sample_credit_data):
        """Test duplicate detection"""
        class DataQualityAssessor:
            def detect_duplicates(self, data, subset=None):
                duplicates = data.duplicated(subset=subset)
                duplicate_count = duplicates.sum()
                duplicate_percentage = (duplicate_count / len(data)) * 100

                return {
                    'duplicate_count': duplicate_count,
                    'duplicate_percentage': duplicate_percentage,
                    'duplicate_rows': data[duplicates]
                }

        # Add some duplicates
        data_with_duplicates = pd.concat([sample_credit_data, sample_credit_data.iloc[:5]], ignore_index=True)

        assessor = DataQualityAssessor()
        duplicate_stats = assessor.detect_duplicates(data_with_duplicates)

        assert duplicate_stats['duplicate_count'] == 5
        assert len(duplicate_stats['duplicate_rows']) == 5

class TestDataPipeline:
    """Test complete data processing pipeline"""

    def test_complete_data_pipeline(self, sample_data_with_missing):
        """Test complete data processing pipeline"""
        from sklearn.preprocessing import StandardScaler

        class DataPipeline:
            def __init__(self):
                self.scaler = None

            def process_data(self, data, target_column):
                # Step 1: Handle missing values
                data_clean = data.copy()
                numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if col != target_column:
                        data_clean[col].fillna(data_clean[col].median(), inplace=True)

                # Step 2: Scale numerical features
                feature_columns = [col for col in data_clean.columns if col != target_column]

                self.scaler = StandardScaler()
                data_clean[feature_columns] = self.scaler.fit_transform(data_clean[feature_columns])

                return data_clean

        pipeline = DataPipeline()
        processed_data = pipeline.process_data(sample_data_with_missing, target_column='default')

        # Check that pipeline completed successfully
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert processed_data.isnull().sum().sum() == 0  # No missing values
        assert pipeline.scaler is not None

        # Check that numerical features are scaled
        feature_columns = [col for col in processed_data.columns if col != 'default']
        for col in feature_columns:
            if processed_data[col].dtype in ['int64', 'float64']:
                assert abs(processed_data[col].mean()) < 0.1  # Approximately zero mean

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
