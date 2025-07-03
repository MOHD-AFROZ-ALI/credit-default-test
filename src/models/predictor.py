
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.utils.helpers import (
    ConfigManager, FileManager, DataValidator, 
    format_percentage, get_risk_category, get_risk_color
)


class CreditDefaultPredictor:
    """
    Credit default predictor for making predictions using trained models.
    
    This class handles individual and batch predictions, risk assessment,
    and provides explanations for predictions.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the predictor.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = get_logger()
        self.config = ConfigManager(config_path)
        
        # Prediction parameters
        self.default_threshold = 0.5
        self.high_risk_threshold = self.config.get('business_rules.high_risk_threshold', 0.7)
        self.medium_risk_threshold = self.config.get('business_rules.medium_risk_threshold', 0.3)
        
        # Model storage
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        
        # Business rules
        self.max_credit_limit = self.config.get('business_rules.max_credit_limit', 1000000)
        self.min_age = self.config.get('business_rules.min_age', 18)
        self.max_age = self.config.get('business_rules.max_age', 100)
        
        self.logger.info("CreditDefaultPredictor initialized")
    
    def load_models(self, models_dir: str = "data/models") -> None:
        """
        Load trained models from directory.
        
        Args:
            models_dir (str): Directory containing saved models
        """
        self.logger.info(f"Loading models from {models_dir}")
        
        # List of expected model files
        model_files = [
            'xgboost_model.pkl',
            'lightgbm_model.pkl', 
            'catboost_model.pkl',
            'random_forest_model.pkl',
            'logistic_regression_model.pkl'
        ]
        
        loaded_count = 0
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            
            if os.path.exists(model_path):
                try:
                    model_name = model_file.replace('_model.pkl', '')
                    model = FileManager.load_model(model_path)
                    self.models[model_name] = model
                    loaded_count += 1
                    self.logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    self.logger.error(f"Failed to load {model_file}: {str(e)}")
        
        if loaded_count == 0:
            self.logger.warning("No models were loaded successfully")
        else:
            self.logger.info(f"Successfully loaded {loaded_count} models")
    
    def load_preprocessor(self, preprocessor_path: str = "data/models/preprocessor.pkl") -> None:
        """
        Load the data preprocessor.
        
        Args:
            preprocessor_path (str): Path to saved preprocessor
        """
        try:
            preprocessor_data = FileManager.load_model(preprocessor_path)
            self.preprocessor = preprocessor_data
            self.feature_names = preprocessor_data.get('feature_names', [])
            self.logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            self.logger.error(f"Failed to load preprocessor: {str(e)}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data for prediction.
        
        Args:
            input_data (Dict[str, Any]): Input data dictionary
            
        Returns:
            Dict[str, Any]: Validation results
        """
        return DataValidator.validate_prediction_input(input_data)
    
    def preprocess_input(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Input data (dict or DataFrame)
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Apply preprocessing if available
        if self.preprocessor:
            # Apply the same preprocessing steps used during training
            # This is a simplified version - in practice, you'd use the actual preprocessor
            
            # Handle missing values
            df = df.fillna(0)
            
            # Ensure all required features are present
            if self.feature_names:
                for feature in self.feature_names:
                    if feature not in df.columns:
                        df[feature] = 0
                
                # Select only the features used during training
                df = df[self.feature_names]
        
        return df
    
    def predict_single(self, input_data: Dict[str, Any], model_name: str = None,
                      return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Make prediction for a single instance.
        
        Args:
            input_data (Dict[str, Any]): Input data dictionary
            model_name (str): Specific model to use (if None, uses ensemble)
            return_probabilities (bool): Whether to return probabilities
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Validate input
        validation_result = self.validate_input(input_data)
        if not validation_result['is_valid']:
            return {
                'success': False,
                'error': validation_result['errors'],
                'warnings': validation_result.get('warnings', [])
            }
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            if model_name and model_name in self.models:
                # Use specific model
                model = self.models[model_name]
                prediction = model.predict(processed_data)[0]
                
                if return_probabilities and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(processed_data)[0]
                    probability = probabilities[1]  # Probability of default
                else:
                    probability = float(prediction)
            
            elif len(self.models) > 0:
                # Use ensemble of all available models
                predictions = []
                probabilities = []
                
                for name, model in self.models.items():
                    pred = model.predict(processed_data)[0]
                    predictions.append(pred)
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(processed_data)[0][1]
                        probabilities.append(prob)
                    else:
                        probabilities.append(float(pred))
                
                # Ensemble prediction (average)
                prediction = int(np.mean(predictions) >= self.default_threshold)
                probability = np.mean(probabilities)
            
            else:
                return {
                    'success': False,
                    'error': 'No models available for prediction'
                }
            
            # Risk assessment
            risk_category = get_risk_category(probability)
            risk_color = get_risk_color(probability)
            
            # Prepare result
            result = {
                'success': True,
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_category': risk_category,
                'risk_color': risk_color,
                'confidence': self._calculate_confidence(probability),
                'recommendation': self._get_recommendation(probability, input_data),
                'model_used': model_name if model_name else 'ensemble',
                'input_data': input_data
            }
            
            # Add warnings if any
            if validation_result.get('warnings'):
                result['warnings'] = validation_result['warnings']
            
            self.logger.info(f"Single prediction completed: {risk_category} ({probability:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }
    
    def predict_batch(self, input_data: pd.DataFrame, model_name: str = None,
                     return_probabilities: bool = True, 
                     batch_size: int = 1000) -> Dict[str, Any]:
        """
        Make predictions for a batch of instances.
        
        Args:
            input_data (pd.DataFrame): Input data DataFrame
            model_name (str): Specific model to use (if None, uses ensemble)
            return_probabilities (bool): Whether to return probabilities
            batch_size (int): Size of processing batches
            
        Returns:
            Dict[str, Any]: Batch prediction results
        """
        self.logger.info(f"Starting batch prediction for {len(input_data)} instances")
        
        try:
            # Preprocess input data
            processed_data = self.preprocess_input(input_data)
            
            # Initialize results
            predictions = []
            probabilities = []
            risk_categories = []
            
            # Process in batches
            for i in range(0, len(processed_data), batch_size):
                batch_end = min(i + batch_size, len(processed_data))
                batch_data = processed_data.iloc[i:batch_end]
                
                if model_name and model_name in self.models:
                    # Use specific model
                    model = self.models[model_name]
                    batch_predictions = model.predict(batch_data)
                    predictions.extend(batch_predictions)
                    
                    if return_probabilities and hasattr(model, 'predict_proba'):
                        batch_probabilities = model.predict_proba(batch_data)[:, 1]
                        probabilities.extend(batch_probabilities)
                    else:
                        probabilities.extend(batch_predictions.astype(float))
                
                elif len(self.models) > 0:
                    # Use ensemble
                    batch_predictions_all = []
                    batch_probabilities_all = []
                    
                    for name, model in self.models.items():
                        batch_preds = model.predict(batch_data)
                        batch_predictions_all.append(batch_preds)
                        
                        if hasattr(model, 'predict_proba'):
                            batch_probs = model.predict_proba(batch_data)[:, 1]
                            batch_probabilities_all.append(batch_probs)
                        else:
                            batch_probabilities_all.append(batch_preds.astype(float))
                    
                    # Ensemble predictions
                    ensemble_predictions = np.mean(batch_predictions_all, axis=0)
                    ensemble_probabilities = np.mean(batch_probabilities_all, axis=0)
                    
                    predictions.extend((ensemble_predictions >= self.default_threshold).astype(int))
                    probabilities.extend(ensemble_probabilities)
                
                else:
                    return {
                        'success': False,
                        'error': 'No models available for prediction'
                    }
            
            # Calculate risk categories
            risk_categories = [get_risk_category(prob) for prob in probabilities]
            
            # Create results DataFrame
            results_df = input_data.copy()
            results_df['prediction'] = predictions
            results_df['probability'] = probabilities
            results_df['risk_category'] = risk_categories
            results_df['risk_color'] = [get_risk_color(prob) for prob in probabilities]
            
            # Summary statistics
            summary = {
                'total_predictions': len(predictions),
                'default_predictions': sum(predictions),
                'default_rate': np.mean(predictions),
                'average_probability': np.mean(probabilities),
                'risk_distribution': pd.Series(risk_categories).value_counts().to_dict(),
                'high_risk_count': sum(1 for cat in risk_categories if cat == 'High Risk'),
                'medium_risk_count': sum(1 for cat in risk_categories if cat == 'Medium Risk'),
                'low_risk_count': sum(1 for cat in risk_categories if cat == 'Low Risk')
            }
            
            result = {
                'success': True,
                'results': results_df,
                'summary': summary,
                'model_used': model_name if model_name else 'ensemble'
            }
            
            self.logger.info(f"Batch prediction completed: {len(predictions)} predictions")
            return result
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {str(e)}")
            return {
                'success': False,
                'error': f'Batch prediction failed: {str(e)}'
            }
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability."""
        if probability < 0.1 or probability > 0.9:
            return "High"
        elif probability < 0.3 or probability > 0.7:
            return "Medium"
        else:
            return "Low"
    
    def _get_recommendation(self, probability: float, input_data: Dict[str, Any]) -> str:
        """Get recommendation based on prediction."""
        if probability >= self.high_risk_threshold:
            return "REJECT - High risk of default. Consider declining the application."
        elif probability >= self.medium_risk_threshold:
            return "REVIEW - Medium risk. Consider additional verification or reduced credit limit."
        else:
            return "APPROVE - Low risk of default. Application can be approved."
    
    def get_feature_importance_explanation(self, input_data: Dict[str, Any], 
                                         model_name: str = None) -> Dict[str, Any]:
        """
        Get feature importance explanation for a prediction.
        
        Args:
            input_data (Dict[str, Any]): Input data dictionary
            model_name (str): Specific model to use
            
        Returns:
            Dict[str, Any]: Feature importance explanation
        """
        try:
            if model_name and model_name in self.models:
                model = self.models[model_name]
            elif len(self.models) > 0:
                # Use first available model
                model = list(self.models.values())[0]
                model_name = list(self.models.keys())[0]
            else:
                return {'error': 'No models available'}
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                # Get top features
                top_features = importance_df.head(10)
                
                return {
                    'model_name': model_name,
                    'top_features': top_features.to_dict('records'),
                    'feature_values': {k: v for k, v in input_data.items() if k in feature_names}
                }
            
            else:
                return {'error': f'Model {model_name} does not support feature importance'}
                
        except Exception as e:
            return {'error': f'Failed to get feature importance: {str(e)}'}
    
    def explain_prediction(self, input_data: Dict[str, Any], 
                          model_name: str = None) -> Dict[str, Any]:
        """
        Provide detailed explanation for a prediction.
        
        Args:
            input_data (Dict[str, Any]): Input data dictionary
            model_name (str): Specific model to use
            
        Returns:
            Dict[str, Any]: Detailed prediction explanation
        """
        # Get prediction
        prediction_result = self.predict_single(input_data, model_name)
        
        if not prediction_result['success']:
            return prediction_result
        
        # Get feature importance
        importance_result = self.get_feature_importance_explanation(input_data, model_name)
        
        # Create explanation
        explanation = {
            'prediction_summary': {
                'probability': prediction_result['probability'],
                'risk_category': prediction_result['risk_category'],
                'recommendation': prediction_result['recommendation'],
                'confidence': prediction_result['confidence']
            },
            'key_factors': [],
            'risk_factors': [],
            'protective_factors': []
        }
        
        # Analyze key input factors
        if 'LIMIT_BAL' in input_data:
            limit_bal = input_data['LIMIT_BAL']
            if limit_bal > 500000:
                explanation['protective_factors'].append(f"High credit limit (${limit_bal:,}) indicates financial stability")
            elif limit_bal < 50000:
                explanation['risk_factors'].append(f"Low credit limit (${limit_bal:,}) may indicate limited creditworthiness")
        
        if 'AGE' in input_data:
            age = input_data['AGE']
            if age < 25:
                explanation['risk_factors'].append(f"Young age ({age}) associated with higher default risk")
            elif age > 50:
                explanation['protective_factors'].append(f"Mature age ({age}) associated with lower default risk")
        
        # Add payment history analysis
        pay_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        late_payments = sum(1 for feat in pay_features if feat in input_data and input_data[feat] > 0)
        
        if late_payments > 3:
            explanation['risk_factors'].append(f"Multiple late payments ({late_payments}/6 months) indicate payment difficulties")
        elif late_payments == 0:
            explanation['protective_factors'].append("No late payments in recent history")
        
        # Add feature importance if available
        if 'top_features' in importance_result:
            explanation['feature_importance'] = importance_result['top_features'][:5]
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dict[str, Any]: Model information
        """
        model_info = {
            'loaded_models': list(self.models.keys()),
            'model_count': len(self.models),
            'default_threshold': self.default_threshold,
            'risk_thresholds': {
                'high_risk': self.high_risk_threshold,
                'medium_risk': self.medium_risk_threshold
            },
            'feature_count': len(self.feature_names),
            'preprocessor_loaded': self.preprocessor is not None
        }
        
        return model_info


# Convenience functions
def predict_credit_default(input_data: Dict[str, Any], models_dir: str = "data/models",
                          config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Convenience function to make a credit default prediction.
    
    Args:
        input_data (Dict[str, Any]): Input data dictionary
        models_dir (str): Directory containing saved models
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Prediction results
    """
    predictor = CreditDefaultPredictor(config_path)
    predictor.load_models(models_dir)
    return predictor.predict_single(input_data)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Credit Default Predictor...")
    
    # Create sample input data
    sample_input = {
        'LIMIT_BAL': 50000,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 35,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 10000,
        'BILL_AMT2': 8000,
        'BILL_AMT3': 6000,
        'BILL_AMT4': 4000,
        'BILL_AMT5': 2000,
        'BILL_AMT6': 1000,
        'PAY_AMT1': 2000,
        'PAY_AMT2': 1500,
        'PAY_AMT3': 1000,
        'PAY_AMT4': 800,
        'PAY_AMT5': 500,
        'PAY_AMT6': 300
    }
    
    # Test predictor
    predictor = CreditDefaultPredictor()
    
    # Test input validation
    validation_result = predictor.validate_input(sample_input)
    print(f"Input validation: {'Passed' if validation_result['is_valid'] else 'Failed'}")
    
    # Test preprocessing
    processed_data = predictor.preprocess_input(sample_input)
    print(f"Preprocessing completed. Shape: {processed_data.shape}")
    
    # Test model info
    model_info = predictor.get_model_info()
    print(f"Model info: {model_info['model_count']} models loaded")
    
    print("Credit Default Predictor testing completed successfully!")