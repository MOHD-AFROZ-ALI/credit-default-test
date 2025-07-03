"""
Form Components for Credit Default Prediction System
Provides reusable form components for user input and data collection
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date, timedelta
import json
import io

def customer_information_form(
    key_prefix: str = "customer",
    default_values: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create customer information input form

    Args:
        key_prefix: Unique prefix for form keys
        default_values: Default values for form fields

    Returns:
        Dict containing customer information
    """
    st.markdown("### 👤 Customer Information")

    defaults = default_values or {}

    with st.form(f"{key_prefix}_info_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Personal Information
            st.markdown("**Personal Details:**")
            first_name = st.text_input(
                "First Name",
                value=defaults.get('first_name', ''),
                key=f"{key_prefix}_first_name"
            )

            last_name = st.text_input(
                "Last Name", 
                value=defaults.get('last_name', ''),
                key=f"{key_prefix}_last_name"
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=defaults.get('age', 30),
                key=f"{key_prefix}_age"
            )

            gender = st.selectbox(
                "Gender",
                options=["Male", "Female", "Other", "Prefer not to say"],
                index=0 if not defaults.get('gender') else 
                      ["Male", "Female", "Other", "Prefer not to say"].index(defaults.get('gender')),
                key=f"{key_prefix}_gender"
            )

            marital_status = st.selectbox(
                "Marital Status",
                options=["Single", "Married", "Divorced", "Widowed"],
                index=0 if not defaults.get('marital_status') else
                      ["Single", "Married", "Divorced", "Widowed"].index(defaults.get('marital_status')),
                key=f"{key_prefix}_marital_status"
            )

        with col2:
            # Contact Information
            st.markdown("**Contact Information:**")
            email = st.text_input(
                "Email Address",
                value=defaults.get('email', ''),
                key=f"{key_prefix}_email"
            )

            phone = st.text_input(
                "Phone Number",
                value=defaults.get('phone', ''),
                key=f"{key_prefix}_phone"
            )

            address = st.text_area(
                "Address",
                value=defaults.get('address', ''),
                height=100,
                key=f"{key_prefix}_address"
            )

            city = st.text_input(
                "City",
                value=defaults.get('city', ''),
                key=f"{key_prefix}_city"
            )

            state = st.text_input(
                "State/Province",
                value=defaults.get('state', ''),
                key=f"{key_prefix}_state"
            )

            zip_code = st.text_input(
                "ZIP/Postal Code",
                value=defaults.get('zip_code', ''),
                key=f"{key_prefix}_zip_code"
            )

        # Financial Information
        st.markdown("**Financial Information:**")

        fin_col1, fin_col2, fin_col3 = st.columns(3)

        with fin_col1:
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                max_value=1000000,
                value=defaults.get('annual_income', 50000),
                step=1000,
                key=f"{key_prefix}_annual_income"
            )

            employment_status = st.selectbox(
                "Employment Status",
                options=["Employed", "Self-Employed", "Unemployed", "Retired", "Student"],
                index=0 if not defaults.get('employment_status') else
                      ["Employed", "Self-Employed", "Unemployed", "Retired", "Student"].index(defaults.get('employment_status')),
                key=f"{key_prefix}_employment_status"
            )

        with fin_col2:
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=defaults.get('credit_score', 650),
                key=f"{key_prefix}_credit_score"
            )

            years_employed = st.number_input(
                "Years at Current Job",
                min_value=0,
                max_value=50,
                value=defaults.get('years_employed', 2),
                key=f"{key_prefix}_years_employed"
            )

        with fin_col3:
            existing_debt = st.number_input(
                "Existing Debt ($)",
                min_value=0,
                max_value=500000,
                value=defaults.get('existing_debt', 10000),
                step=500,
                key=f"{key_prefix}_existing_debt"
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                options=["Own", "Rent", "Mortgage", "Other"],
                index=0 if not defaults.get('home_ownership') else
                      ["Own", "Rent", "Mortgage", "Other"].index(defaults.get('home_ownership')),
                key=f"{key_prefix}_home_ownership"
            )

        submitted = st.form_submit_button("Save Customer Information")

        if submitted:
            customer_data = {
                'first_name': first_name,
                'last_name': last_name,
                'age': age,
                'gender': gender,
                'marital_status': marital_status,
                'email': email,
                'phone': phone,
                'address': address,
                'city': city,
                'state': state,
                'zip_code': zip_code,
                'annual_income': annual_income,
                'employment_status': employment_status,
                'credit_score': credit_score,
                'years_employed': years_employed,
                'existing_debt': existing_debt,
                'home_ownership': home_ownership,
                'created_at': datetime.now().isoformat()
            }

            st.success("✅ Customer information saved successfully!")
            return customer_data

    return {}

def loan_application_form(
    key_prefix: str = "loan",
    default_values: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create loan application form

    Args:
        key_prefix: Unique prefix for form keys
        default_values: Default values for form fields

    Returns:
        Dict containing loan application data
    """
    st.markdown("### 💰 Loan Application")

    defaults = default_values or {}

    with st.form(f"{key_prefix}_application_form"):
        col1, col2 = st.columns(2)

        with col1:
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=100000,
                value=defaults.get('loan_amount', 25000),
                step=1000,
                key=f"{key_prefix}_amount"
            )

            loan_purpose = st.selectbox(
                "Loan Purpose",
                options=[
                    "Personal", "Home Improvement", "Auto", "Business",
                    "Education", "Medical", "Debt Consolidation", "Other"
                ],
                index=0 if not defaults.get('loan_purpose') else
                      ["Personal", "Home Improvement", "Auto", "Business",
                       "Education", "Medical", "Debt Consolidation", "Other"].index(defaults.get('loan_purpose')),
                key=f"{key_prefix}_purpose"
            )

            loan_term = st.selectbox(
                "Loan Term (months)",
                options=[12, 24, 36, 48, 60, 72, 84],
                index=2 if not defaults.get('loan_term') else
                      [12, 24, 36, 48, 60, 72, 84].index(defaults.get('loan_term')),
                key=f"{key_prefix}_term"
            )

        with col2:
            interest_rate = st.number_input(
                "Requested Interest Rate (%)",
                min_value=1.0,
                max_value=30.0,
                value=defaults.get('interest_rate', 8.5),
                step=0.1,
                key=f"{key_prefix}_interest_rate"
            )

            collateral = st.selectbox(
                "Collateral",
                options=["None", "Vehicle", "Property", "Savings", "Other"],
                index=0 if not defaults.get('collateral') else
                      ["None", "Vehicle", "Property", "Savings", "Other"].index(defaults.get('collateral')),
                key=f"{key_prefix}_collateral"
            )

            urgency = st.selectbox(
                "Urgency",
                options=["Low", "Medium", "High", "Urgent"],
                index=1 if not defaults.get('urgency') else
                      ["Low", "Medium", "High", "Urgent"].index(defaults.get('urgency')),
                key=f"{key_prefix}_urgency"
            )

        # Additional Information
        st.markdown("**Additional Information:**")

        additional_info = st.text_area(
            "Additional Comments or Information",
            value=defaults.get('additional_info', ''),
            height=100,
            key=f"{key_prefix}_additional_info"
        )

        # Terms and Conditions
        terms_accepted = st.checkbox(
            "I accept the terms and conditions",
            value=defaults.get('terms_accepted', False),
            key=f"{key_prefix}_terms"
        )

        submitted = st.form_submit_button("Submit Loan Application")

        if submitted:
            if not terms_accepted:
                st.error("❌ Please accept the terms and conditions to proceed.")
                return {}

            loan_data = {
                'loan_amount': loan_amount,
                'loan_purpose': loan_purpose,
                'loan_term': loan_term,
                'interest_rate': interest_rate,
                'collateral': collateral,
                'urgency': urgency,
                'additional_info': additional_info,
                'terms_accepted': terms_accepted,
                'application_date': datetime.now().isoformat()
            }

            st.success("✅ Loan application submitted successfully!")
            return loan_data

    return {}

def model_configuration_form(
    key_prefix: str = "model_config",
    available_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create model configuration form

    Args:
        key_prefix: Unique prefix for form keys
        available_models: List of available model types

    Returns:
        Dict containing model configuration
    """
    st.markdown("### 🤖 Model Configuration")

    if available_models is None:
        available_models = [
            "Random Forest", "XGBoost", "Logistic Regression", 
            "Neural Network", "Support Vector Machine", "Ensemble"
        ]

    with st.form(f"{key_prefix}_form"):
        # Model Selection
        model_type = st.selectbox(
            "Model Type",
            options=available_models,
            key=f"{key_prefix}_type"
        )

        # Training Parameters
        st.markdown("**Training Parameters:**")

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=50,
                value=20,
                key=f"{key_prefix}_test_size"
            )

            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=1000,
                value=42,
                key=f"{key_prefix}_random_state"
            )

            cross_validation_folds = st.number_input(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=5,
                key=f"{key_prefix}_cv_folds"
            )

        with col2:
            feature_selection = st.checkbox(
                "Enable Feature Selection",
                value=True,
                key=f"{key_prefix}_feature_selection"
            )

            hyperparameter_tuning = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=False,
                key=f"{key_prefix}_hyperparameter_tuning"
            )

            class_balancing = st.selectbox(
                "Class Balancing Method",
                options=["None", "SMOTE", "Random Oversampling", "Random Undersampling"],
                key=f"{key_prefix}_class_balancing"
            )

        # Model-Specific Parameters
        st.markdown("**Model-Specific Parameters:**")

        model_params = {}

        if model_type == "Random Forest":
            rf_col1, rf_col2 = st.columns(2)
            with rf_col1:
                model_params['n_estimators'] = st.number_input(
                    "Number of Trees", 10, 500, 100, key=f"{key_prefix}_rf_n_estimators"
                )
                model_params['max_depth'] = st.number_input(
                    "Max Depth", 3, 20, 10, key=f"{key_prefix}_rf_max_depth"
                )
            with rf_col2:
                model_params['min_samples_split'] = st.number_input(
                    "Min Samples Split", 2, 20, 2, key=f"{key_prefix}_rf_min_samples_split"
                )
                model_params['min_samples_leaf'] = st.number_input(
                    "Min Samples Leaf", 1, 10, 1, key=f"{key_prefix}_rf_min_samples_leaf"
                )

        elif model_type == "XGBoost":
            xgb_col1, xgb_col2 = st.columns(2)
            with xgb_col1:
                model_params['n_estimators'] = st.number_input(
                    "Number of Estimators", 10, 500, 100, key=f"{key_prefix}_xgb_n_estimators"
                )
                model_params['learning_rate'] = st.slider(
                    "Learning Rate", 0.01, 0.3, 0.1, key=f"{key_prefix}_xgb_learning_rate"
                )
            with xgb_col2:
                model_params['max_depth'] = st.number_input(
                    "Max Depth", 3, 15, 6, key=f"{key_prefix}_xgb_max_depth"
                )
                model_params['subsample'] = st.slider(
                    "Subsample", 0.5, 1.0, 0.8, key=f"{key_prefix}_xgb_subsample"
                )

        elif model_type == "Logistic Regression":
            lr_col1, lr_col2 = st.columns(2)
            with lr_col1:
                model_params['C'] = st.slider(
                    "Regularization Strength", 0.01, 10.0, 1.0, key=f"{key_prefix}_lr_C"
                )
                model_params['solver'] = st.selectbox(
                    "Solver", ["liblinear", "lbfgs", "saga"], key=f"{key_prefix}_lr_solver"
                )
            with lr_col2:
                model_params['penalty'] = st.selectbox(
                    "Penalty", ["l1", "l2", "elasticnet", "none"], key=f"{key_prefix}_lr_penalty"
                )
                model_params['max_iter'] = st.number_input(
                    "Max Iterations", 100, 2000, 1000, key=f"{key_prefix}_lr_max_iter"
                )

        elif model_type == "Neural Network":
            nn_col1, nn_col2 = st.columns(2)
            with nn_col1:
                model_params['hidden_layer_sizes'] = st.text_input(
                    "Hidden Layer Sizes (comma-separated)", "100,50", key=f"{key_prefix}_nn_hidden_layers"
                )
                model_params['activation'] = st.selectbox(
                    "Activation Function", ["relu", "tanh", "logistic"], key=f"{key_prefix}_nn_activation"
                )
            with nn_col2:
                model_params['learning_rate'] = st.selectbox(
                    "Learning Rate", ["constant", "invscaling", "adaptive"], key=f"{key_prefix}_nn_learning_rate"
                )
                model_params['max_iter'] = st.number_input(
                    "Max Iterations", 100, 1000, 200, key=f"{key_prefix}_nn_max_iter"
                )

        # Evaluation Metrics
        st.markdown("**Evaluation Metrics:**")
        evaluation_metrics = st.multiselect(
            "Select Metrics to Track",
            options=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Confusion Matrix"],
            default=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            key=f"{key_prefix}_evaluation_metrics"
        )

        # Save Configuration
        save_config = st.checkbox(
            "Save Configuration for Future Use",
            value=True,
            key=f"{key_prefix}_save_config"
        )

        config_name = ""
        if save_config:
            config_name = st.text_input(
                "Configuration Name",
                value=f"{model_type}_config_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key=f"{key_prefix}_config_name"
            )

        submitted = st.form_submit_button("Apply Configuration")

        if submitted:
            config_data = {
                'model_type': model_type,
                'test_size': test_size / 100,  # Convert percentage to decimal
                'random_state': random_state,
                'cross_validation_folds': cross_validation_folds,
                'feature_selection': feature_selection,
                'hyperparameter_tuning': hyperparameter_tuning,
                'class_balancing': class_balancing,
                'model_params': model_params,
                'evaluation_metrics': evaluation_metrics,
                'save_config': save_config,
                'config_name': config_name,
                'created_at': datetime.now().isoformat()
            }

            st.success("✅ Model configuration applied successfully!")
            return config_data

    return {}


def data_upload_form(
    key_prefix: str = "data_upload",
    accepted_formats: Optional[List[str]] = None,
    max_file_size: int = 200
) -> Dict[str, Any]:
    """
    Create data upload and validation form

    Args:
        key_prefix: Unique prefix for form keys
        accepted_formats: List of accepted file formats
        max_file_size: Maximum file size in MB

    Returns:
        Dict containing upload results and validation info
    """
    st.markdown("### 📁 Data Upload & Validation")

    if accepted_formats is None:
        accepted_formats = ['csv', 'xlsx', 'json', 'parquet']

    with st.form(f"{key_prefix}_form"):
        # File Upload Section
        st.markdown("**File Upload:**")

        uploaded_file = st.file_uploader(
            f"Choose a file ({', '.join(accepted_formats).upper()})",
            type=accepted_formats,
            key=f"{key_prefix}_file"
        )

        if uploaded_file:
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")

        # Data Processing Options
        st.markdown("**Data Processing Options:**")

        col1, col2 = st.columns(2)

        with col1:
            has_header = st.checkbox(
                "File has header row",
                value=True,
                key=f"{key_prefix}_has_header"
            )

            delimiter = st.selectbox(
                "CSV Delimiter",
                options=[",", ";", "|", "	"],
                format_func=lambda x: {"," : "Comma", ";" : "Semicolon", "|" : "Pipe", "	" : "Tab"}[x],
                key=f"{key_prefix}_delimiter"
            )

            encoding = st.selectbox(
                "File Encoding",
                options=["utf-8", "latin-1", "cp1252", "iso-8859-1"],
                key=f"{key_prefix}_encoding"
            )

        with col2:
            skip_rows = st.number_input(
                "Skip rows from top",
                min_value=0,
                max_value=100,
                value=0,
                key=f"{key_prefix}_skip_rows"
            )

            max_rows = st.number_input(
                "Maximum rows to read (0 = all)",
                min_value=0,
                max_value=1000000,
                value=0,
                key=f"{key_prefix}_max_rows"
            )

            date_format = st.text_input(
                "Date format (if applicable)",
                value="%Y-%m-%d",
                key=f"{key_prefix}_date_format"
            )

        # Data Validation Options
        st.markdown("**Data Validation:**")

        validation_col1, validation_col2 = st.columns(2)

        with validation_col1:
            check_duplicates = st.checkbox(
                "Check for duplicate rows",
                value=True,
                key=f"{key_prefix}_check_duplicates"
            )

            check_missing = st.checkbox(
                "Check for missing values",
                value=True,
                key=f"{key_prefix}_check_missing"
            )

            validate_data_types = st.checkbox(
                "Validate data types",
                value=True,
                key=f"{key_prefix}_validate_types"
            )

        with validation_col2:
            remove_duplicates = st.checkbox(
                "Automatically remove duplicates",
                value=False,
                key=f"{key_prefix}_remove_duplicates"
            )

            handle_missing = st.selectbox(
                "Handle missing values",
                options=["Keep as is", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
                key=f"{key_prefix}_handle_missing"
            )

            outlier_detection = st.checkbox(
                "Detect outliers",
                value=False,
                key=f"{key_prefix}_outlier_detection"
            )

        # Expected Schema (Optional)
        st.markdown("**Expected Schema (Optional):**")
        expected_columns = st.text_area(
            "Expected column names (one per line)",
            height=100,
            key=f"{key_prefix}_expected_columns",
            help="List expected column names to validate against uploaded data"
        )

        submitted = st.form_submit_button("Upload and Validate Data")

        if submitted and uploaded_file:
            try:
                # Read the file based on format
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(
                        uploaded_file,
                        delimiter=delimiter,
                        encoding=encoding,
                        header=0 if has_header else None,
                        skiprows=skip_rows,
                        nrows=max_rows if max_rows > 0 else None
                    )
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(
                        uploaded_file,
                        header=0 if has_header else None,
                        skiprows=skip_rows,
                        nrows=max_rows if max_rows > 0 else None
                    )
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file, encoding=encoding)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)

                # Validation results
                validation_results = {
                    'file_name': uploaded_file.name,
                    'file_size_mb': uploaded_file.size / 1024 / 1024,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'data_types': df.dtypes.to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }

                # Perform validations
                if check_duplicates:
                    duplicates = df.duplicated().sum()
                    validation_results['duplicates'] = duplicates
                    if remove_duplicates and duplicates > 0:
                        df = df.drop_duplicates()
                        validation_results['duplicates_removed'] = duplicates

                if check_missing:
                    missing_values = df.isnull().sum().to_dict()
                    validation_results['missing_values'] = missing_values

                    if handle_missing != "Keep as is":
                        if handle_missing == "Drop rows":
                            df = df.dropna()
                        elif handle_missing == "Fill with mean":
                            df = df.fillna(df.select_dtypes(include=[np.number]).mean())
                        elif handle_missing == "Fill with median":
                            df = df.fillna(df.select_dtypes(include=[np.number]).median())
                        elif handle_missing == "Fill with mode":
                            df = df.fillna(df.mode().iloc[0])

                if expected_columns:
                    expected_cols = [col.strip() for col in expected_columns.split('') if col.strip()]
                    missing_cols = set(expected_cols) - set(df.columns)
                    extra_cols = set(df.columns) - set(expected_cols)
                    validation_results['schema_validation'] = {
                        'missing_columns': list(missing_cols),
                        'extra_columns': list(extra_cols),
                        'schema_match': len(missing_cols) == 0 and len(extra_cols) == 0
                    }

                if outlier_detection:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    outliers = {}
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                        outliers[col] = outlier_count
                    validation_results['outliers'] = outliers

                st.success("✅ Data uploaded and validated successfully!")

                # Store data in session state
                st.session_state[f"{key_prefix}_data"] = df
                st.session_state[f"{key_prefix}_validation"] = validation_results

                return {
                    'data': df,
                    'validation_results': validation_results,
                    'success': True
                }

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                return {'success': False, 'error': str(e)}

        elif submitted and not uploaded_file:
            st.warning("⚠️ Please select a file to upload.")

    return {}

def batch_prediction_form(
    key_prefix: str = "batch_prediction",
    available_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create batch prediction form

    Args:
        key_prefix: Unique prefix for form keys
        available_models: List of available trained models

    Returns:
        Dict containing batch prediction configuration
    """
    st.markdown("### 🔮 Batch Prediction")

    if available_models is None:
        available_models = ["Default Model", "Random Forest v1.0", "XGBoost v2.1", "Ensemble v1.5"]

    with st.form(f"{key_prefix}_form"):
        # Model Selection
        st.markdown("**Model Selection:**")

        selected_model = st.selectbox(
            "Choose Model",
            options=available_models,
            key=f"{key_prefix}_model"
        )

        model_version = st.text_input(
            "Model Version (optional)",
            key=f"{key_prefix}_version"
        )

        # Data Source
        st.markdown("**Data Source:**")

        data_source = st.radio(
            "Select data source",
            options=["Upload new file", "Use previously uploaded data", "Connect to database"],
            key=f"{key_prefix}_data_source"
        )

        if data_source == "Upload new file":
            batch_file = st.file_uploader(
                "Upload batch prediction file",
                type=['csv', 'xlsx', 'json'],
                key=f"{key_prefix}_batch_file"
            )

        elif data_source == "Use previously uploaded data":
            # Show available datasets from session state
            available_datasets = [key for key in st.session_state.keys() if key.endswith('_data')]
            if available_datasets:
                selected_dataset = st.selectbox(
                    "Select dataset",
                    options=available_datasets,
                    key=f"{key_prefix}_dataset"
                )
            else:
                st.info("No previously uploaded datasets found.")

        elif data_source == "Connect to database":
            db_connection = st.text_input(
                "Database connection string",
                key=f"{key_prefix}_db_connection"
            )

            db_query = st.text_area(
                "SQL Query",
                height=100,
                key=f"{key_prefix}_db_query"
            )

        # Prediction Options
        st.markdown("**Prediction Options:**")

        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            include_probabilities = st.checkbox(
                "Include prediction probabilities",
                value=True,
                key=f"{key_prefix}_include_probs"
            )

            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
                key=f"{key_prefix}_confidence_threshold"
            )

            batch_size = st.number_input(
                "Batch processing size",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key=f"{key_prefix}_batch_size"
            )

        with pred_col2:
            include_explanations = st.checkbox(
                "Include prediction explanations",
                value=False,
                key=f"{key_prefix}_include_explanations"
            )

            output_format = st.selectbox(
                "Output format",
                options=["CSV", "Excel", "JSON", "Parquet"],
                key=f"{key_prefix}_output_format"
            )

            save_to_database = st.checkbox(
                "Save results to database",
                value=False,
                key=f"{key_prefix}_save_to_db"
            )

        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            feature_columns = st.text_area(
                "Specify feature columns (optional, one per line)",
                height=80,
                key=f"{key_prefix}_feature_columns",
                help="Leave empty to use all available features"
            )

            preprocessing_steps = st.multiselect(
                "Preprocessing steps",
                options=["Normalize", "Scale", "Handle missing values", "Encode categoricals"],
                default=["Handle missing values", "Encode categoricals"],
                key=f"{key_prefix}_preprocessing"
            )

            parallel_processing = st.checkbox(
                "Enable parallel processing",
                value=True,
                key=f"{key_prefix}_parallel"
            )

            if parallel_processing:
                n_jobs = st.slider(
                    "Number of parallel jobs",
                    min_value=1,
                    max_value=8,
                    value=4,
                    key=f"{key_prefix}_n_jobs"
                )

        # Schedule Options
        st.markdown("**Scheduling (Optional):**")

        schedule_prediction = st.checkbox(
            "Schedule this prediction",
            value=False,
            key=f"{key_prefix}_schedule"
        )

        if schedule_prediction:
            schedule_col1, schedule_col2 = st.columns(2)

            with schedule_col1:
                schedule_type = st.selectbox(
                    "Schedule type",
                    options=["One-time", "Daily", "Weekly", "Monthly"],
                    key=f"{key_prefix}_schedule_type"
                )

                schedule_time = st.time_input(
                    "Execution time",
                    key=f"{key_prefix}_schedule_time"
                )

            with schedule_col2:
                if schedule_type == "Weekly":
                    schedule_day = st.selectbox(
                        "Day of week",
                        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                        key=f"{key_prefix}_schedule_day"
                    )
                elif schedule_type == "Monthly":
                    schedule_day = st.number_input(
                        "Day of month",
                        min_value=1,
                        max_value=31,
                        value=1,
                        key=f"{key_prefix}_schedule_day_month"
                    )

                notification_email = st.text_input(
                    "Notification email",
                    key=f"{key_prefix}_notification_email"
                )

        submitted = st.form_submit_button("Start Batch Prediction")

        if submitted:
            prediction_config = {
                'selected_model': selected_model,
                'model_version': model_version,
                'data_source': data_source,
                'include_probabilities': include_probabilities,
                'confidence_threshold': confidence_threshold,
                'batch_size': batch_size,
                'include_explanations': include_explanations,
                'output_format': output_format,
                'save_to_database': save_to_database,
                'preprocessing_steps': preprocessing_steps,
                'parallel_processing': parallel_processing,
                'schedule_prediction': schedule_prediction,
                'created_at': datetime.now().isoformat()
            }

            if feature_columns:
                prediction_config['feature_columns'] = [col.strip() for col in feature_columns.split('') if col.strip()]

            if parallel_processing:
                prediction_config['n_jobs'] = n_jobs

            if schedule_prediction:
                prediction_config['schedule_config'] = {
                    'type': schedule_type,
                    'time': schedule_time.strftime('%H:%M'),
                    'notification_email': notification_email
                }

                if schedule_type in ["Weekly", "Monthly"]:
                    prediction_config['schedule_config']['day'] = schedule_day

            st.success("✅ Batch prediction job configured successfully!")
            return prediction_config

    return {}

def system_settings_form(
    key_prefix: str = "system_settings",
    current_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create system settings and configuration form

    Args:
        key_prefix: Unique prefix for form keys
        current_settings: Current system settings

    Returns:
        Dict containing updated system settings
    """
    st.markdown("### ⚙️ System Settings & Configuration")

    defaults = current_settings or {}

    with st.form(f"{key_prefix}_form"):
        # General Settings
        st.markdown("**General Settings:**")

        gen_col1, gen_col2 = st.columns(2)

        with gen_col1:
            app_name = st.text_input(
                "Application Name",
                value=defaults.get('app_name', 'Credit Default Prediction System'),
                key=f"{key_prefix}_app_name"
            )

            default_theme = st.selectbox(
                "Default Theme",
                options=["Light", "Dark", "Auto"],
                index=["Light", "Dark", "Auto"].index(defaults.get('default_theme', 'Light')),
                key=f"{key_prefix}_theme"
            )

            language = st.selectbox(
                "Language",
                options=["English", "Spanish", "French", "German", "Chinese"],
                index=["English", "Spanish", "French", "German", "Chinese"].index(defaults.get('language', 'English')),
                key=f"{key_prefix}_language"
            )

        with gen_col2:
            timezone = st.selectbox(
                "Timezone",
                options=["UTC", "EST", "PST", "GMT", "CET"],
                index=["UTC", "EST", "PST", "GMT", "CET"].index(defaults.get('timezone', 'UTC')),
                key=f"{key_prefix}_timezone"
            )

            session_timeout = st.number_input(
                "Session Timeout (minutes)",
                min_value=15,
                max_value=480,
                value=defaults.get('session_timeout', 60),
                key=f"{key_prefix}_session_timeout"
            )

            max_file_size = st.number_input(
                "Max Upload File Size (MB)",
                min_value=1,
                max_value=1000,
                value=defaults.get('max_file_size', 200),
                key=f"{key_prefix}_max_file_size"
            )

        # Performance Settings
        st.markdown("**Performance Settings:**")

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            cache_enabled = st.checkbox(
                "Enable Data Caching",
                value=defaults.get('cache_enabled', True),
                key=f"{key_prefix}_cache_enabled"
            )

            if cache_enabled:
                cache_ttl = st.number_input(
                    "Cache TTL (hours)",
                    min_value=1,
                    max_value=168,
                    value=defaults.get('cache_ttl', 24),
                    key=f"{key_prefix}_cache_ttl"
                )

            parallel_processing = st.checkbox(
                "Enable Parallel Processing",
                value=defaults.get('parallel_processing', True),
                key=f"{key_prefix}_parallel_processing"
            )

        with perf_col2:
            max_workers = st.number_input(
                "Max Worker Threads",
                min_value=1,
                max_value=16,
                value=defaults.get('max_workers', 4),
                key=f"{key_prefix}_max_workers"
            )

            memory_limit = st.number_input(
                "Memory Limit (GB)",
                min_value=1,
                max_value=32,
                value=defaults.get('memory_limit', 8),
                key=f"{key_prefix}_memory_limit"
            )

            auto_cleanup = st.checkbox(
                "Auto Cleanup Temp Files",
                value=defaults.get('auto_cleanup', True),
                key=f"{key_prefix}_auto_cleanup"
            )

        # Security Settings
        st.markdown("**Security Settings:**")

        sec_col1, sec_col2 = st.columns(2)

        with sec_col1:
            enable_logging = st.checkbox(
                "Enable Audit Logging",
                value=defaults.get('enable_logging', True),
                key=f"{key_prefix}_enable_logging"
            )

            log_level = st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(defaults.get('log_level', 'INFO')),
                key=f"{key_prefix}_log_level"
            )

            data_encryption = st.checkbox(
                "Enable Data Encryption",
                value=defaults.get('data_encryption', False),
                key=f"{key_prefix}_data_encryption"
            )

        with sec_col2:
            backup_enabled = st.checkbox(
                "Enable Auto Backup",
                value=defaults.get('backup_enabled', True),
                key=f"{key_prefix}_backup_enabled"
            )

            if backup_enabled:
                backup_frequency = st.selectbox(
                    "Backup Frequency",
                    options=["Daily", "Weekly", "Monthly"],
                    index=["Daily", "Weekly", "Monthly"].index(defaults.get('backup_frequency', 'Daily')),
                    key=f"{key_prefix}_backup_frequency"
                )

            retention_days = st.number_input(
                "Data Retention (days)",
                min_value=30,
                max_value=2555,
                value=defaults.get('retention_days', 365),
                key=f"{key_prefix}_retention_days"
            )

        # Notification Settings
        st.markdown("**Notification Settings:**")

        notif_col1, notif_col2 = st.columns(2)

        with notif_col1:
            email_notifications = st.checkbox(
                "Enable Email Notifications",
                value=defaults.get('email_notifications', True),
                key=f"{key_prefix}_email_notifications"
            )

            if email_notifications:
                smtp_server = st.text_input(
                    "SMTP Server",
                    value=defaults.get('smtp_server', ''),
                    key=f"{key_prefix}_smtp_server"
                )

                smtp_port = st.number_input(
                    "SMTP Port",
                    min_value=1,
                    max_value=65535,
                    value=defaults.get('smtp_port', 587),
                    key=f"{key_prefix}_smtp_port"
                )

        with notif_col2:
            admin_email = st.text_input(
                "Admin Email",
                value=defaults.get('admin_email', ''),
                key=f"{key_prefix}_admin_email"
            )

            notification_types = st.multiselect(
                "Notification Types",
                options=["System Errors", "Model Training Complete", "Batch Jobs", "Security Alerts"],
                default=defaults.get('notification_types', ["System Errors", "Security Alerts"]),
                key=f"{key_prefix}_notification_types"
            )

        submitted = st.form_submit_button("Save Settings")

        if submitted:
            settings_data = {
                'app_name': app_name,
                'default_theme': default_theme,
                'language': language,
                'timezone': timezone,
                'session_timeout': session_timeout,
                'max_file_size': max_file_size,
                'cache_enabled': cache_enabled,
                'parallel_processing': parallel_processing,
                'max_workers': max_workers,
                'memory_limit': memory_limit,
                'auto_cleanup': auto_cleanup,
                'enable_logging': enable_logging,
                'log_level': log_level,
                'data_encryption': data_encryption,
                'backup_enabled': backup_enabled,
                'retention_days': retention_days,
                'email_notifications': email_notifications,
                'admin_email': admin_email,
                'notification_types': notification_types,
                'updated_at': datetime.now().isoformat()
            }

            if cache_enabled:
                settings_data['cache_ttl'] = cache_ttl

            if backup_enabled:
                settings_data['backup_frequency'] = backup_frequency

            if email_notifications:
                settings_data.update({
                    'smtp_server': smtp_server,
                    'smtp_port': smtp_port
                })

            st.success("✅ System settings saved successfully!")
            return settings_data

    return {}
