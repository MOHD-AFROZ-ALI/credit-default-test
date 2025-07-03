import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditDefaultFeatureEngineer:
    """
    Comprehensive feature engineering for UCI Credit Default Dataset.
    Creates advanced features for credit default prediction.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature engineer with configuration."""
        self.config = config or {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            df: Input DataFrame with raw features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting comprehensive feature engineering...")

        # Make a copy to avoid modifying original data
        data = df.copy()

        # Basic feature validation
        self._validate_input_features(data)

        # Create engineered features
        data = self._create_credit_utilization_features(data)
        data = self._create_payment_behavior_features(data)
        data = self._create_financial_health_features(data)
        data = self._create_risk_scoring_features(data)
        data = self._create_temporal_features(data)
        data = self._create_interaction_features(data)
        data = self._create_demographic_features(data)

        # Feature quality checks
        data = self._perform_feature_quality_checks(data)

        logger.info(f"Feature engineering complete. Created {len(data.columns)} total features.")
        return data

    def _validate_input_features(self, df: pd.DataFrame) -> None:
        """Validate that required features are present."""
        required_features = [
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]

        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        logger.info("Input feature validation passed.")

    def _create_credit_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create credit utilization and balance-related features."""
        # Average bill amount
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
        df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
        df['max_bill_amt'] = df[bill_cols].max(axis=1)
        df['min_bill_amt'] = df[bill_cols].min(axis=1)
        df['std_bill_amt'] = df[bill_cols].std(axis=1)

        # Credit utilization ratios
        df['avg_utilization_ratio'] = df['avg_bill_amt'] / (df['LIMIT_BAL'] + 1)
        df['max_utilization_ratio'] = df['max_bill_amt'] / (df['LIMIT_BAL'] + 1)

        # Bill amount trends
        df['bill_amt_trend'] = (df['BILL_AMT1'] - df['BILL_AMT6']) / (df['BILL_AMT6'] + 1)
        df['bill_amt_volatility'] = df['std_bill_amt'] / (df['avg_bill_amt'] + 1)

        return df

    def _create_payment_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create payment behavior and history features."""
        pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]

        # Payment status statistics
        df['avg_pay_status'] = df[pay_cols].mean(axis=1)
        df['max_pay_delay'] = df[pay_cols].max(axis=1)
        df['pay_delay_frequency'] = (df[pay_cols] > 0).sum(axis=1)
        df['severe_delay_count'] = (df[pay_cols] >= 2).sum(axis=1)

        # Payment amount statistics
        df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
        df['total_pay_amt'] = df[pay_amt_cols].sum(axis=1)
        df['pay_amt_consistency'] = 1 - (df[pay_amt_cols].std(axis=1) / (df['avg_pay_amt'] + 1))

        # Payment ratio features
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
        for i in range(6):
            df[f'pay_ratio_{i+1}'] = df[pay_amt_cols[i]] / (df[bill_cols[i]] + 1)

        df['avg_pay_ratio'] = df[[f'pay_ratio_{i+1}' for i in range(6)]].mean(axis=1)

        return df

    def _create_financial_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial health and stability indicators."""
        # Credit limit relative to demographics
        df['limit_per_age'] = df['LIMIT_BAL'] / (df['AGE'] + 1)

        # Financial stress indicators
        df['debt_burden'] = df['avg_bill_amt'] / (df['LIMIT_BAL'] + 1)
        df['payment_burden'] = df['avg_pay_amt'] / (df['avg_bill_amt'] + 1)

        # Financial improvement/deterioration
        df['financial_trend'] = (df['PAY_AMT1'] - df['PAY_AMT6']) / (df['PAY_AMT6'] + 1)
        df['balance_trend'] = (df['BILL_AMT6'] - df['BILL_AMT1']) / (df['BILL_AMT1'] + 1)

        return df

    def _create_risk_scoring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk scoring and composite features."""
        # Payment risk score (higher = more risky)
        df['payment_risk_score'] = (
            df['max_pay_delay'] * 0.3 +
            df['pay_delay_frequency'] * 0.2 +
            df['severe_delay_count'] * 0.5
        )

        # Credit risk score
        df['credit_risk_score'] = (
            df['max_utilization_ratio'] * 0.4 +
            df['debt_burden'] * 0.3 +
            (1 - df['payment_burden']) * 0.3
        )

        # Overall risk score
        df['overall_risk_score'] = (
            df['payment_risk_score'] * 0.6 +
            df['credit_risk_score'] * 0.4
        )

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal pattern features."""
        pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

        # Payment improvement/deterioration
        df['payment_improving'] = (df['PAY_6'] > df['PAY_0']).astype(int)
        df['payment_deteriorating'] = (df['PAY_0'] > df['PAY_6']).astype(int)

        # Recent vs historical behavior
        df['recent_pay_avg'] = df[['PAY_0', 'PAY_2', 'PAY_3']].mean(axis=1)
        df['historical_pay_avg'] = df[['PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
        df['behavior_change'] = df['recent_pay_avg'] - df['historical_pay_avg']

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        # Age-based interactions
        df['age_limit_interaction'] = df['AGE'] * df['LIMIT_BAL'] / 1000000
        df['age_risk_interaction'] = df['AGE'] * df['overall_risk_score']

        # Gender-based interactions
        df['gender_limit_interaction'] = df['SEX'] * df['LIMIT_BAL'] / 1000000
        df['gender_risk_interaction'] = df['SEX'] * df['overall_risk_score']

        # Education-based interactions
        df['education_limit_interaction'] = df['EDUCATION'] * df['LIMIT_BAL'] / 1000000

        return df

    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic-based features."""
        # Age groups
        df['age_group'] = pd.cut(df['AGE'], bins=[0, 30, 40, 50, 100], labels=[1, 2, 3, 4])
        df['age_group'] = df['age_group'].astype(float)

        # Credit limit groups
        df['limit_group'] = pd.qcut(df['LIMIT_BAL'], q=5, labels=[1, 2, 3, 4, 5])
        df['limit_group'] = df['limit_group'].astype(float)

        # Demographic risk profiles
        df['high_risk_demo'] = (
            ((df['AGE'] < 25) | (df['AGE'] > 65)) |
            (df['EDUCATION'] > 3) |
            (df['MARRIAGE'] == 3)
        ).astype(int)

        return df

    def _perform_feature_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform feature quality checks and cleanup."""
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Remove features with zero variance
        zero_var_cols = []
        for col in numeric_cols:
            if df[col].var() == 0:
                zero_var_cols.append(col)

        if zero_var_cols:
            df = df.drop(columns=zero_var_cols)
            logger.info(f"Removed {len(zero_var_cols)} zero-variance features")

        return df

    def get_feature_importance_ranking(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> List[str]:
        """
        Get top k most important features using statistical tests.

        Args:
            X: Feature matrix
            y: Target variable
            k: Number of top features to return

        Returns:
            List of top k feature names
        """
        # Remove target column if present
        feature_cols = [col for col in X.columns if col not in ['default.payment.next.month', 'default']]
        X_features = X[feature_cols]

        # Select top k features
        selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
        selector.fit(X_features, y)

        # Get feature names and scores
        feature_scores = list(zip(feature_cols, selector.scores_))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        top_features = [name for name, score in feature_scores[:k]]

        logger.info(f"Top {k} features identified: {top_features[:5]}...")
        return top_features


class AdvancedFeatureUtils:
    """
    Advanced feature engineering utilities and helper functions.
    Complements the CreditDefaultFeatureEngineer class.
    """

    def __init__(self):
        """Initialize advanced feature utilities."""
        self.correlation_threshold = 0.95
        self.stability_threshold = 0.1

    def generate_polynomial_features(self, df: pd.DataFrame, features: List[str], 
                                   degree: int = 2, interaction_only: bool = False) -> pd.DataFrame:
        """
        Generate polynomial and interaction features for specified columns.

        Args:
            df: Input DataFrame
            features: List of feature names to create polynomials for
            degree: Polynomial degree
            interaction_only: If True, only create interaction terms

        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures

        logger.info(f"Generating polynomial features (degree={degree}) for {len(features)} features...")

        # Select only specified features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            logger.warning("No specified features found in dataframe")
            return df

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                                include_bias=False)

        # Fit and transform the selected features
        X_poly = poly.fit_transform(df[available_features])

        # Create feature names
        poly_feature_names = poly.get_feature_names_out(available_features)

        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=df.index)

        # Remove original features to avoid duplication
        poly_df = poly_df.drop(columns=available_features, errors='ignore')

        # Combine with original dataframe
        result_df = pd.concat([df, poly_df], axis=1)

        logger.info(f"Created {len(poly_df.columns)} polynomial features")
        return result_df

    def advanced_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 method: str = 'mutual_info', k: int = 50) -> List[str]:
        """
        Advanced feature selection using multiple methods.

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('mutual_info', 'chi2', 'f_classif', 'rfe')
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import (mutual_info_classif, chi2, f_classif, 
                                             SelectKBest, RFE)
        from sklearn.ensemble import RandomForestClassifier

        logger.info(f"Performing advanced feature selection using {method}...")

        # Remove target column if present
        feature_cols = [col for col in X.columns if col not in ['default.payment.next.month', 'default']]
        X_features = X[feature_cols]

        # Ensure no negative values for chi2
        if method == 'chi2':
            X_features = X_features - X_features.min() + 1

        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_cols)))
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=min(k, len(feature_cols)))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=min(k, len(feature_cols)))
        else:
            raise ValueError(f"Unknown selection method: {method}")

        # Fit selector
        selector.fit(X_features, y)

        # Get selected features
        if hasattr(selector, 'get_support'):
            selected_features = [feature_cols[i] for i, selected in enumerate(selector.get_support()) if selected]
        else:
            selected_features = feature_cols[:k]  # Fallback

        logger.info(f"Selected {len(selected_features)} features using {method}")
        return selected_features

    def remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce multicollinearity.

        Args:
            df: Input DataFrame
            threshold: Correlation threshold for removal

        Returns:
            DataFrame with correlated features removed
        """
        logger.info(f"Removing features with correlation > {threshold}...")

        # Calculate correlation matrix for numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()

        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to remove
        to_remove = [column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > threshold)]

        # Remove correlated features
        result_df = df.drop(columns=to_remove)

        logger.info(f"Removed {len(to_remove)} highly correlated features")
        return result_df

    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Any]:
        """
        Scale features using various methods.

        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile')
            exclude_cols: Columns to exclude from scaling

        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

        logger.info(f"Scaling features using {method} method...")

        exclude_cols = exclude_cols or []

        # Select numeric columns for scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Create a copy of the dataframe
        scaled_df = df.copy()

        # Scale the selected columns
        if cols_to_scale:
            scaled_values = scaler.fit_transform(df[cols_to_scale])
            scaled_df[cols_to_scale] = scaled_values

        logger.info(f"Scaled {len(cols_to_scale)} features")
        return scaled_df, scaler

    def test_feature_stability(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             features: List[str] = None) -> Dict[str, float]:
        """
        Test feature stability between two datasets (e.g., train vs validation).

        Args:
            df1: First dataset
            df2: Second dataset
            features: List of features to test (if None, test all common features)

        Returns:
            Dictionary with feature stability scores
        """
        logger.info("Testing feature stability between datasets...")

        # Get common features
        if features is None:
            features = list(set(df1.columns) & set(df2.columns))

        stability_scores = {}

        for feature in features:
            if feature in df1.columns and feature in df2.columns:
                # Calculate Population Stability Index (PSI)
                psi_score = self._calculate_psi(df1[feature], df2[feature])
                stability_scores[feature] = psi_score

        # Identify unstable features
        unstable_features = [f for f, score in stability_scores.items() 
                           if score > self.stability_threshold]

        logger.info(f"Found {len(unstable_features)} potentially unstable features")
        return stability_scores

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI) between two distributions."""
        try:
            # Create bins based on expected distribution
            _, bin_edges = np.histogram(expected.dropna(), bins=buckets)

            # Calculate expected and actual distributions
            expected_counts, _ = np.histogram(expected.dropna(), bins=bin_edges)
            actual_counts, _ = np.histogram(actual.dropna(), bins=bin_edges)

            # Convert to percentages
            expected_pct = expected_counts / len(expected.dropna())
            actual_pct = actual_counts / len(actual.dropna())

            # Avoid division by zero
            expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
            actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

            # Calculate PSI
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return psi
        except:
            return 0.0  # Return 0 if calculation fails

    def export_feature_metadata(self, df: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
        """
        Export comprehensive feature metadata for documentation and monitoring.

        Args:
            df: DataFrame to analyze
            output_path: Path to save metadata JSON file

        Returns:
            Dictionary containing feature metadata
        """
        logger.info("Generating feature metadata...")

        metadata = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns),
            'feature_details': {}
        }

        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_values': int(df[col].nunique())
            }

            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'skewness': float(df[col].skew()) if not df[col].isnull().all() else None
                })

            metadata['feature_details'][col] = col_info

        # Save to file if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Feature metadata saved to {output_path}")

        return metadata

    def create_feature_engineering_report(self, original_df: pd.DataFrame, 
                                        engineered_df: pd.DataFrame, 
                                        output_path: str = None) -> Dict[str, Any]:
        """
        Create a comprehensive report comparing original and engineered features.

        Args:
            original_df: Original DataFrame before feature engineering
            engineered_df: DataFrame after feature engineering
            output_path: Path to save the report

        Returns:
            Dictionary containing the feature engineering report
        """
        logger.info("Creating feature engineering report...")

        report = {
            'original_features': len(original_df.columns),
            'engineered_features': len(engineered_df.columns),
            'new_features_created': len(engineered_df.columns) - len(original_df.columns),
            'feature_creation_ratio': len(engineered_df.columns) / len(original_df.columns),
            'original_feature_list': list(original_df.columns),
            'new_feature_list': [col for col in engineered_df.columns if col not in original_df.columns],
            'memory_usage': {
                'original_mb': original_df.memory_usage(deep=True).sum() / 1024**2,
                'engineered_mb': engineered_df.memory_usage(deep=True).sum() / 1024**2
            }
        }

        # Calculate memory increase
        report['memory_increase_ratio'] = (
            report['memory_usage']['engineered_mb'] / report['memory_usage']['original_mb']
        )

        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Feature engineering report saved to {output_path}")

        return report


def create_feature_pipeline(config: Dict[str, Any] = None) -> Tuple[CreditDefaultFeatureEngineer, AdvancedFeatureUtils]:
    """
    Factory function to create a complete feature engineering pipeline.

    Args:
        config: Configuration dictionary for feature engineering

    Returns:
        Tuple of (main feature engineer, advanced utilities)
    """
    config = config or {}

    # Initialize main feature engineer
    feature_engineer = CreditDefaultFeatureEngineer(config)

    # Initialize advanced utilities
    advanced_utils = AdvancedFeatureUtils()

    logger.info("Feature engineering pipeline created successfully")
    return feature_engineer, advanced_utils


# Example usage function
def example_feature_engineering_workflow(df: pd.DataFrame, target_col: str = 'default.payment.next.month'):
    """
    Example workflow demonstrating the complete feature engineering process.

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        Processed DataFrame with engineered features
    """
    logger.info("Starting example feature engineering workflow...")

    # Initialize components
    feature_engineer, advanced_utils = create_feature_pipeline()

    # Step 1: Basic feature engineering
    engineered_df = feature_engineer.engineer_features(df)

    # Step 2: Generate polynomial features for key numeric features
    key_features = ['LIMIT_BAL', 'AGE', 'avg_bill_amt', 'avg_pay_amt']
    engineered_df = advanced_utils.generate_polynomial_features(
        engineered_df, key_features, degree=2, interaction_only=True
    )

    # Step 3: Remove highly correlated features
    engineered_df = advanced_utils.remove_correlated_features(engineered_df, threshold=0.95)

    # Step 4: Feature selection
    if target_col in engineered_df.columns:
        y = engineered_df[target_col]
        X = engineered_df.drop(columns=[target_col])

        selected_features = advanced_utils.advanced_feature_selection(
            X, y, method='mutual_info', k=50
        )

        # Keep only selected features plus target
        engineered_df = engineered_df[selected_features + [target_col]]

    # Step 5: Scale features
    exclude_cols = [target_col] if target_col in engineered_df.columns else []
    scaled_df, scaler = advanced_utils.scale_features(
        engineered_df, method='standard', exclude_cols=exclude_cols
    )

    logger.info("Feature engineering workflow completed successfully")
    return scaled_df
