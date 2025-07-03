"""
Test suite for machine learning models and evaluation functionality
Comprehensive tests for model training, evaluation, prediction, and performance metrics
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import pickle
import json

# Import ML libraries for testing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Test fixtures
@pytest.fixture
def sample_training_data():
    """Generate sample training data for ML models"""
    np.random.seed(42)
    n_samples = 1000

    # Generate features
    X = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'employment_length': np.random.exponential(5, n_samples),
        'loan_amount': np.random.lognormal(9, 0.8, n_samples),
        'interest_rate': np.random.uniform(3, 18, n_samples),
        'num_credit_lines': np.random.poisson(3, n_samples),
        'credit_utilization': np.random.uniform(0.1, 0.9, n_samples)
    })

    # Generate target with realistic correlation
    risk_score = (
        (850 - X['credit_score']) / 550 * 0.3 +
        X['debt_to_income_ratio'] * 0.3 +
        (X['credit_utilization'] - 0.5) * 0.2 +
        np.random.random(n_samples) * 0.2
    )

    y = (risk_score > 0.5).astype(int)

    return X, y

@pytest.fixture
def sample_test_data():
    """Generate sample test data for model evaluation"""
    np.random.seed(123)
    n_samples = 200

    X = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'employment_length': np.random.exponential(5, n_samples),
        'loan_amount': np.random.lognormal(9, 0.8, n_samples),
        'interest_rate': np.random.uniform(3, 18, n_samples),
        'num_credit_lines': np.random.poisson(3, n_samples),
        'credit_utilization': np.random.uniform(0.1, 0.9, n_samples)
    })

    # Generate target with same logic as training data
    risk_score = (
        (850 - X['credit_score']) / 550 * 0.3 +
        X['debt_to_income_ratio'] * 0.3 +
        (X['credit_utilization'] - 0.5) * 0.2 +
        np.random.random(n_samples) * 0.2
    )

    y = (risk_score > 0.5).astype(int)

    return X, y

@pytest.fixture
def trained_models(sample_training_data):
    """Fixture providing pre-trained models for testing"""
    X_train, y_train = sample_training_data

    models = {
        'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=10, random_state=42)
    }

    # Train all models
    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

@pytest.fixture
def temp_model_file():
    """Create temporary file for model serialization tests"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)

class TestModelTraining:
    """Test model training functionality"""

    def test_random_forest_training(self, sample_training_data):
        """Test Random Forest model training"""
        X_train, y_train = sample_training_data

        class ModelTrainer:
            def train_random_forest(self, X, y, **kwargs):
                model = RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', None),
                    random_state=kwargs.get('random_state', 42)
                )
                model.fit(X, y)
                return model

        trainer = ModelTrainer()
        model = trainer.train_random_forest(X_train, y_train, n_estimators=50)

        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]
        assert model.n_estimators == 50

    def test_logistic_regression_training(self, sample_training_data):
        """Test Logistic Regression model training"""
        X_train, y_train = sample_training_data

        class ModelTrainer:
            def train_logistic_regression(self, X, y, **kwargs):
                model = LogisticRegression(
                    C=kwargs.get('C', 1.0),
                    max_iter=kwargs.get('max_iter', 1000),
                    random_state=kwargs.get('random_state', 42)
                )
                model.fit(X, y)
                return model

        trainer = ModelTrainer()
        model = trainer.train_logistic_regression(X_train, y_train, C=0.5)

        assert isinstance(model, LogisticRegression)
        assert hasattr(model, 'coef_')
        assert model.coef_.shape[1] == X_train.shape[1]
        assert model.C == 0.5

    def test_training_with_preprocessing_pipeline(self, sample_training_data):
        """Test model training with preprocessing pipeline"""
        X_train, y_train = sample_training_data

        class ModelTrainer:
            def train_with_pipeline(self, X, y, model_type='random_forest'):
                if model_type == 'random_forest':
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                elif model_type == 'logistic_regression':
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])

                pipeline.fit(X, y)
                return pipeline

        trainer = ModelTrainer()

        # Test with Random Forest
        rf_pipeline = trainer.train_with_pipeline(X_train, y_train, 'random_forest')
        assert isinstance(rf_pipeline, Pipeline)
        assert 'scaler' in rf_pipeline.named_steps
        assert 'model' in rf_pipeline.named_steps

        # Test with Logistic Regression
        lr_pipeline = trainer.train_with_pipeline(X_train, y_train, 'logistic_regression')
        assert isinstance(lr_pipeline.named_steps['model'], LogisticRegression)

class TestModelEvaluation:
    """Test model evaluation functionality"""

    def test_basic_metrics_calculation(self, trained_models, sample_test_data):
        """Test calculation of basic performance metrics"""
        X_test, y_test = sample_test_data

        class ModelEvaluator:
            def calculate_metrics(self, model, X_test, y_test):
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary'),
                    'f1_score': f1_score(y_test, y_pred, average='binary')
                }

                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

                return metrics

        evaluator = ModelEvaluator()

        for model_name, model in trained_models.items():
            metrics = evaluator.calculate_metrics(model, X_test, y_test)

            # Check that all metrics are calculated
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'roc_auc' in metrics

            # Check that metrics are in valid ranges
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1_score'] <= 1
            assert 0 <= metrics['roc_auc'] <= 1

    def test_confusion_matrix_calculation(self, trained_models, sample_test_data):
        """Test confusion matrix calculation"""
        X_test, y_test = sample_test_data

        class ModelEvaluator:
            def calculate_confusion_matrix(self, model, X_test, y_test):
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)

                return {
                    'confusion_matrix': cm,
                    'true_negatives': cm[0, 0],
                    'false_positives': cm[0, 1],
                    'false_negatives': cm[1, 0],
                    'true_positives': cm[1, 1]
                }

        evaluator = ModelEvaluator()

        for model_name, model in trained_models.items():
            cm_results = evaluator.calculate_confusion_matrix(model, X_test, y_test)

            assert 'confusion_matrix' in cm_results
            assert cm_results['confusion_matrix'].shape == (2, 2)

            # Check that all values sum to total test samples
            total = (cm_results['true_negatives'] + cm_results['false_positives'] + 
                    cm_results['false_negatives'] + cm_results['true_positives'])
            assert total == len(y_test)

    def test_classification_report_generation(self, trained_models, sample_test_data):
        """Test classification report generation"""
        X_test, y_test = sample_test_data

        class ModelEvaluator:
            def generate_classification_report(self, model, X_test, y_test):
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                return report

        evaluator = ModelEvaluator()

        for model_name, model in trained_models.items():
            report = evaluator.generate_classification_report(model, X_test, y_test)

            assert isinstance(report, dict)
            assert '0' in report  # Class 0 metrics
            assert '1' in report  # Class 1 metrics
            assert 'accuracy' in report
            assert 'macro avg' in report
            assert 'weighted avg' in report

class TestModelPrediction:
    """Test model prediction functionality"""

    def test_single_prediction(self, trained_models):
        """Test single sample prediction"""
        # Create a single sample for prediction
        single_sample = pd.DataFrame({
            'age': [35],
            'income': [75000],
            'credit_score': [720],
            'debt_to_income_ratio': [0.3],
            'employment_length': [5],
            'loan_amount': [200000],
            'interest_rate': [4.5],
            'num_credit_lines': [3],
            'credit_utilization': [0.4]
        })

        class ModelPredictor:
            def predict_single(self, model, sample):
                prediction = model.predict(sample)[0]
                probability = model.predict_proba(sample)[0] if hasattr(model, 'predict_proba') else None

                return {
                    'prediction': prediction,
                    'probability': probability
                }

        predictor = ModelPredictor()

        for model_name, model in trained_models.items():
            result = predictor.predict_single(model, single_sample)

            assert 'prediction' in result
            assert result['prediction'] in [0, 1]

            if result['probability'] is not None:
                assert len(result['probability']) == 2
                assert abs(sum(result['probability']) - 1.0) < 1e-6  # Probabilities sum to 1

    def test_batch_prediction(self, trained_models, sample_test_data):
        """Test batch prediction"""
        X_test, y_test = sample_test_data

        class ModelPredictor:
            def predict_batch(self, model, X):
                predictions = model.predict(X)
                probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

                return {
                    'predictions': predictions,
                    'probabilities': probabilities
                }

        predictor = ModelPredictor()

        for model_name, model in trained_models.items():
            result = predictor.predict_batch(model, X_test)

            assert 'predictions' in result
            assert len(result['predictions']) == len(X_test)
            assert all(pred in [0, 1] for pred in result['predictions'])

            if result['probabilities'] is not None:
                assert result['probabilities'].shape == (len(X_test), 2)
                # Check that probabilities sum to 1 for each sample
                prob_sums = result['probabilities'].sum(axis=1)
                assert all(abs(s - 1.0) < 1e-6 for s in prob_sums)

    def test_prediction_with_feature_importance(self, trained_models, sample_test_data):
        """Test prediction with feature importance"""
        X_test, y_test = sample_test_data

        class ModelPredictor:
            def predict_with_importance(self, model, X):
                predictions = model.predict(X)

                # Get feature importance if available
                importance = None
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = abs(model.coef_[0])  # Absolute values for logistic regression

                return {
                    'predictions': predictions,
                    'feature_importance': importance,
                    'feature_names': X.columns.tolist()
                }

        predictor = ModelPredictor()

        for model_name, model in trained_models.items():
            result = predictor.predict_with_importance(model, X_test)

            assert 'predictions' in result
            assert 'feature_importance' in result
            assert 'feature_names' in result

            if result['feature_importance'] is not None:
                assert len(result['feature_importance']) == len(result['feature_names'])
                assert all(imp >= 0 for imp in result['feature_importance'])

class TestModelSerialization:
    """Test model serialization and deserialization"""

    def test_model_save_load(self, trained_models, temp_model_file):
        """Test saving and loading models"""
        class ModelSerializer:
            def save_model(self, model, filepath):
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)

            def load_model(self, filepath):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)

        serializer = ModelSerializer()

        for model_name, model in trained_models.items():
            # Save model
            serializer.save_model(model, temp_model_file)
            assert os.path.exists(temp_model_file)

            # Load model
            loaded_model = serializer.load_model(temp_model_file)

            # Test that loaded model works
            assert type(loaded_model) == type(model)

            # Test prediction consistency
            test_sample = np.random.random((1, 9))  # 9 features
            original_pred = model.predict(test_sample)
            loaded_pred = loaded_model.predict(test_sample)

            assert original_pred[0] == loaded_pred[0]

class TestCrossValidation:
    """Test cross-validation functionality"""

    def test_k_fold_cross_validation(self, sample_training_data):
        """Test k-fold cross-validation"""
        X_train, y_train = sample_training_data

        class ModelValidator:
            def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

                return {
                    'scores': scores,
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'cv_folds': cv
                }

        validator = ModelValidator()

        models_to_test = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        for model_name, model in models_to_test.items():
            cv_results = validator.cross_validate_model(model, X_train, y_train, cv=3)

            assert 'scores' in cv_results
            assert 'mean_score' in cv_results
            assert 'std_score' in cv_results
            assert 'cv_folds' in cv_results

            assert len(cv_results['scores']) == 3
            assert 0 <= cv_results['mean_score'] <= 1
            assert cv_results['std_score'] >= 0

class TestHyperparameterTuning:
    """Test hyperparameter tuning functionality"""

    def test_grid_search_cv(self, sample_training_data):
        """Test grid search cross-validation"""
        X_train, y_train = sample_training_data

        class HyperparameterTuner:
            def grid_search_tune(self, model, param_grid, X, y, cv=3, scoring='accuracy'):
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
                )
                grid_search.fit(X, y)

                return {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_estimator': grid_search.best_estimator_
                }

        tuner = HyperparameterTuner()

        # Test with Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }

        rf_results = tuner.grid_search_tune(rf_model, rf_param_grid, X_train, y_train, cv=2)

        assert 'best_params' in rf_results
        assert 'best_score' in rf_results
        assert 'best_estimator' in rf_results

        # Check that best parameters are from the grid
        assert rf_results['best_params']['n_estimators'] in [10, 20]
        assert rf_results['best_params']['max_depth'] in [3, 5]
        assert 0 <= rf_results['best_score'] <= 1

class TestModelComparison:
    """Test model comparison functionality"""

    def test_compare_multiple_models(self, sample_training_data, sample_test_data):
        """Test comparison of multiple models"""
        X_train, y_train = sample_training_data
        X_test, y_test = sample_test_data

        class ModelComparator:
            def compare_models(self, models, X_train, y_train, X_test, y_test):
                results = {}

                for name, model in models.items():
                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred)
                    }

                    if y_pred_proba is not None:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

                    results[name] = metrics

                return results

        comparator = ModelComparator()

        models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=10, random_state=42)
        }

        results = comparator.compare_models(models, X_train, y_train, X_test, y_test)

        assert len(results) == 3
        for model_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 0 <= metrics['accuracy'] <= 1

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
