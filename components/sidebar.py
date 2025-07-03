"""
Sidebar Components for Credit Default Prediction System
Provides navigation, model selection, data filters, and settings controls
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

def render_main_navigation() -> str:
    """
    Render main navigation menu in sidebar

    Returns:
        str: Selected page/section
    """
    st.sidebar.title("🏦 Credit Default Prediction")
    st.sidebar.markdown("---")

    # Main navigation options
    nav_options = [
        "📊 Dashboard",
        "🔍 Data Explorer", 
        "🤖 Model Training",
        "📈 Model Evaluation",
        "🎯 Predictions",
        "⚙️ Settings"
    ]

    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        nav_options,
        key="main_navigation"
    )

    return selected_page

def render_model_selection() -> Dict[str, Any]:
    """
    Render model selection controls

    Returns:
        Dict[str, Any]: Selected model configuration
    """
    st.sidebar.markdown("### 🤖 Model Configuration")

    # Model type selection
    model_types = [
        "Random Forest",
        "XGBoost", 
        "Logistic Regression",
        "Neural Network",
        "Ensemble"
    ]

    selected_model = st.sidebar.selectbox(
        "Model Type:",
        model_types,
        key="model_type_selection"
    )

    # Model parameters based on selection
    model_config = {"model_type": selected_model}

    if selected_model == "Random Forest":
        model_config.update({
            "n_estimators": st.sidebar.slider("Number of Trees", 10, 500, 100),
            "max_depth": st.sidebar.slider("Max Depth", 3, 20, 10),
            "min_samples_split": st.sidebar.slider("Min Samples Split", 2, 20, 2)
        })

    elif selected_model == "XGBoost":
        model_config.update({
            "n_estimators": st.sidebar.slider("Number of Estimators", 10, 500, 100),
            "learning_rate": st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1),
            "max_depth": st.sidebar.slider("Max Depth", 3, 15, 6)
        })

    elif selected_model == "Logistic Regression":
        model_config.update({
            "C": st.sidebar.slider("Regularization Strength", 0.01, 10.0, 1.0),
            "solver": st.sidebar.selectbox("Solver", ["liblinear", "lbfgs", "saga"])
        })

    elif selected_model == "Neural Network":
        model_config.update({
            "hidden_layers": st.sidebar.slider("Hidden Layers", 1, 5, 2),
            "neurons_per_layer": st.sidebar.slider("Neurons per Layer", 10, 200, 50),
            "learning_rate": st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
        })

    return model_config

def render_data_filters() -> Dict[str, Any]:
    """
    Render data filtering controls

    Returns:
        Dict[str, Any]: Applied filters configuration
    """
    st.sidebar.markdown("### 🔍 Data Filters")

    filters = {}

    # Date range filter
    st.sidebar.markdown("**Date Range:**")
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(datetime.now() - timedelta(days=365), datetime.now()),
        key="date_range_filter"
    )
    filters["date_range"] = date_range

    # Credit score range
    st.sidebar.markdown("**Credit Score Range:**")
    credit_score_range = st.sidebar.slider(
        "Credit Score:",
        min_value=300,
        max_value=850,
        value=(500, 800),
        key="credit_score_filter"
    )
    filters["credit_score_range"] = credit_score_range

    # Income range
    st.sidebar.markdown("**Annual Income Range:**")
    income_range = st.sidebar.slider(
        "Income ($):",
        min_value=0,
        max_value=200000,
        value=(20000, 100000),
        step=5000,
        key="income_filter"
    )
    filters["income_range"] = income_range

    # Loan amount range
    st.sidebar.markdown("**Loan Amount Range:**")
    loan_amount_range = st.sidebar.slider(
        "Loan Amount ($):",
        min_value=1000,
        max_value=100000,
        value=(5000, 50000),
        step=1000,
        key="loan_amount_filter"
    )
    filters["loan_amount_range"] = loan_amount_range

    # Categorical filters
    st.sidebar.markdown("**Categorical Filters:**")

    # Employment status
    employment_options = ["All", "Employed", "Self-Employed", "Unemployed", "Retired"]
    selected_employment = st.sidebar.multiselect(
        "Employment Status:",
        employment_options[1:],  # Exclude "All"
        default=employment_options[1:],
        key="employment_filter"
    )
    filters["employment_status"] = selected_employment

    # Loan purpose
    loan_purpose_options = [
        "Personal", "Home", "Auto", "Business", 
        "Education", "Medical", "Debt Consolidation"
    ]
    selected_purposes = st.sidebar.multiselect(
        "Loan Purpose:",
        loan_purpose_options,
        default=loan_purpose_options,
        key="loan_purpose_filter"
    )
    filters["loan_purpose"] = selected_purposes

    # Risk category
    risk_categories = ["Low", "Medium", "High"]
    selected_risk = st.sidebar.multiselect(
        "Risk Category:",
        risk_categories,
        default=risk_categories,
        key="risk_category_filter"
    )
    filters["risk_category"] = selected_risk

    return filters

def render_settings_panel() -> Dict[str, Any]:
    """
    Render application settings panel

    Returns:
        Dict[str, Any]: Application settings
    """
    st.sidebar.markdown("### ⚙️ Settings")

    settings = {}

    # Display settings
    st.sidebar.markdown("**Display Options:**")
    settings["theme"] = st.sidebar.selectbox(
        "Theme:",
        ["Light", "Dark", "Auto"],
        key="theme_setting"
    )

    settings["chart_style"] = st.sidebar.selectbox(
        "Chart Style:",
        ["Default", "Minimal", "Professional"],
        key="chart_style_setting"
    )

    settings["show_tooltips"] = st.sidebar.checkbox(
        "Show Tooltips",
        value=True,
        key="tooltips_setting"
    )

    # Data settings
    st.sidebar.markdown("**Data Options:**")
    settings["auto_refresh"] = st.sidebar.checkbox(
        "Auto Refresh Data",
        value=False,
        key="auto_refresh_setting"
    )

    if settings["auto_refresh"]:
        settings["refresh_interval"] = st.sidebar.slider(
            "Refresh Interval (minutes):",
            min_value=1,
            max_value=60,
            value=5,
            key="refresh_interval_setting"
        )

    settings["cache_data"] = st.sidebar.checkbox(
        "Cache Data",
        value=True,
        key="cache_setting"
    )

    # Model settings
    st.sidebar.markdown("**Model Options:**")
    settings["confidence_threshold"] = st.sidebar.slider(
        "Prediction Confidence Threshold:",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        key="confidence_threshold_setting"
    )

    settings["enable_explanations"] = st.sidebar.checkbox(
        "Enable Model Explanations",
        value=True,
        key="explanations_setting"
    )

    # Export settings
    st.sidebar.markdown("**Export Options:**")
    settings["export_format"] = st.sidebar.selectbox(
        "Default Export Format:",
        ["CSV", "Excel", "JSON", "PDF"],
        key="export_format_setting"
    )

    settings["include_metadata"] = st.sidebar.checkbox(
        "Include Metadata in Exports",
        value=True,
        key="metadata_setting"
    )

    return settings

def render_quick_actions() -> Optional[str]:
    """
    Render quick action buttons

    Returns:
        Optional[str]: Selected quick action
    """
    st.sidebar.markdown("### ⚡ Quick Actions")

    action = None

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🔄 Refresh Data", key="refresh_action"):
            action = "refresh_data"

        if st.button("📊 Generate Report", key="report_action"):
            action = "generate_report"

    with col2:
        if st.button("💾 Save Model", key="save_model_action"):
            action = "save_model"

        if st.button("📤 Export Data", key="export_action"):
            action = "export_data"

    return action

def render_system_status() -> None:
    """
    Render system status information
    """
    st.sidebar.markdown("### 📊 System Status")

    # Mock system metrics (in real app, these would come from actual system monitoring)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.metric("Models", "5", delta="1")
        st.metric("Accuracy", "94.2%", delta="0.3%")

    with col2:
        st.metric("Records", "10.5K", delta="250")
        st.metric("Uptime", "99.9%", delta="0.1%")

    # Status indicators
    st.sidebar.markdown("**Service Status:**")
    st.sidebar.success("🟢 API Service: Online")
    st.sidebar.success("🟢 Database: Connected")
    st.sidebar.success("🟢 Model Server: Active")

def render_help_section() -> None:
    """
    Render help and documentation section
    """
    st.sidebar.markdown("### ❓ Help & Support")

    with st.sidebar.expander("📖 Quick Guide"):
        st.markdown("""
        **Getting Started:**
        1. Select a page from navigation
        2. Configure filters and settings
        3. Upload or select data
        4. Train or load a model
        5. Generate predictions

        **Tips:**
        - Use filters to focus on specific data segments
        - Adjust model parameters for better performance
        - Export results for further analysis
        """)

    with st.sidebar.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        - **Slow loading**: Check data size and filters
        - **Model errors**: Verify data quality and parameters
        - **Export issues**: Check file permissions

        **Contact Support:**
        - Email: support@creditml.com
        - Docs: docs.creditml.com
        """)

def render_complete_sidebar() -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[str]]:
    """
    Render complete sidebar with all components

    Returns:
        Tuple containing:
        - Selected page
        - Model configuration
        - Applied filters
        - Settings
        - Quick action (if any)
    """
    # Main navigation
    selected_page = render_main_navigation()

    # Model selection (show only on relevant pages)
    model_config = {}
    if any(page in selected_page for page in ["Model Training", "Model Evaluation", "Predictions"]):
        model_config = render_model_selection()

    # Data filters (show on data-related pages)
    filters = {}
    if any(page in selected_page for page in ["Dashboard", "Data Explorer", "Predictions"]):
        filters = render_data_filters()

    # Settings panel
    settings = render_settings_panel()

    # Quick actions
    quick_action = render_quick_actions()

    # System status
    render_system_status()

    # Help section
    render_help_section()

    return selected_page, model_config, filters, settings, quick_action
