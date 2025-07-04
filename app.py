"""
Credit Default Prediction System - Main Application
==================================================

This is the main entry point for the Credit Default Prediction Streamlit application.
It handles page routing, sidebar navigation, global configuration, and session state management.

Author: Credit Risk Analytics Team
Version: 1.0.0
"""

import streamlit as st
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Import custom components
try:
    from components.sidebar import render_sidebar
    from components.metrics import display_key_metrics
    from src.utils.helpers import load_config, setup_logging, initialize_session_state
    from src.data.data_loader import DataLoader
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all required modules are installed and paths are correct.")

# Page configuration
st.set_page_config(
    page_title="Credit Default Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/MOHD-AFROZ-ALI/credit-default-test',
        'Report a bug': 'https://github.com/MOHD-AFROZ-ALI/credit-default-test/issues',
        'About': """
        # Credit Default Prediction System

        A comprehensive machine learning application for predicting credit default risk.

        **Features:**
        - Individual customer risk assessment
        - Batch prediction processing
        - Model performance monitoring
        - Data exploration and visualization
        - Business intelligence dashboard
        - Customer segmentation analysis
        - Compliance reporting

        **Version:** 1.0.0
        """
    }
)

# Load custom CSS
def load_css():
    """Load custom CSS styling"""
    css_file = Path("assets/css/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Default styling if CSS file doesn't exist
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(90deg, #f0f2f6, #ffffff);
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }

        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }

        .sidebar-info {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }

        .status-success {
            color: #28a745;
            font-weight: bold;
        }

        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }

        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

def load_configuration():
    """Load application configuration"""
    config_file = Path("config.yaml")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return get_default_config()
    else:
        st.warning("Configuration file not found. Using default settings.")
        return get_default_config()

def get_default_config():
    """Return default configuration"""
    return {
        'app': {
            'name': 'Credit Default Prediction System',
            'version': '1.0.0',
            'debug': False
        },
        'model': {
            'default_model': 'random_forest',
            'model_path': 'models/',
            'threshold': 0.5
        },
        'data': {
            'upload_path': 'data/uploads/',
            'sample_data': 'data/sample_data.csv',
            'max_file_size': 200  # MB
        },
        'ui': {
            'theme': 'light',
            'show_advanced_options': True,
            'items_per_page': 50
        }
    }

def initialize_app_state():
    """Initialize application session state"""
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.current_page = 'Home'
        st.session_state.user_data = {}
        st.session_state.model_loaded = False
        st.session_state.data_loaded = False
        st.session_state.predictions = {}
        st.session_state.app_config = load_configuration()

        # Initialize data loader
        try:
            st.session_state.data_loader = DataLoader()
        except:
            st.session_state.data_loader = None

        # Setup logging
        setup_logging(st.session_state.app_config.get('app', {}).get('debug', False))

def render_main_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        💳 Credit Default Prediction System
    </div>
    """, unsafe_allow_html=True)

    # Display system status
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_status = "✅ Ready" if st.session_state.get('model_loaded', False) else "⚠️ Not Loaded"
        st.metric("Model Status", model_status)

    with col2:
        data_status = "✅ Loaded" if st.session_state.get('data_loaded', False) else "⚠️ No Data"
        st.metric("Data Status", data_status)

    with col3:
        predictions_count = len(st.session_state.get('predictions', {}))
        st.metric("Predictions Made", predictions_count)

    with col4:
        app_version = st.session_state.app_config.get('app', {}).get('version', '1.0.0')
        st.metric("Version", app_version)

def render_navigation_info():
    """Render navigation information"""
    st.markdown("---")

    st.markdown("""
    ## 🧭 Navigation Guide

    Use the sidebar to navigate between different sections of the application:

    **📊 Core Prediction Features:**
    - **Individual Prediction**: Assess single customer default risk
    - **Batch Prediction**: Process multiple customers at once
    - **Model Performance**: Monitor and evaluate model accuracy

    **📈 Analytics & Insights:**
    - **Data Exploration**: Explore and visualize your data
    - **Business Intelligence**: View key business metrics and trends
    - **Customer Segmentation**: Analyze customer groups and patterns
    - **Compliance Report**: Generate regulatory compliance reports

    **💡 Getting Started:**
    1. Start with **Individual Prediction** to test single customers
    2. Use **Data Exploration** to understand your dataset
    3. Check **Model Performance** to ensure accuracy
    4. Generate **Batch Predictions** for multiple customers
    """)

def render_quick_stats():
    """Render quick statistics dashboard"""
    st.markdown("## 📊 Quick Statistics")

    # Sample statistics (replace with real data when available)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>📈 Model Accuracy</h4>
            <p style="font-size: 2rem; color: #28a745; font-weight: bold;">94.2%</p>
            <p style="color: #666;">Last updated: Today</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>👥 Customers Analyzed</h4>
            <p style="font-size: 2rem; color: #1f77b4; font-weight: bold;">1,247</p>
            <p style="color: #666;">This month</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>⚠️ High Risk Detected</h4>
            <p style="font-size: 2rem; color: #dc3545; font-weight: bold;">23</p>
            <p style="color: #666;">Requires attention</p>
        </div>
        """, unsafe_allow_html=True)

def render_recent_activity():
    """Render recent activity section"""
    st.markdown("## 🕒 Recent Activity")

    # Sample recent activity data
    recent_activities = [
        {"time": "2 minutes ago", "action": "Batch prediction completed", "status": "success"},
        {"time": "15 minutes ago", "action": "New data uploaded", "status": "info"},
        {"time": "1 hour ago", "action": "Model performance check", "status": "success"},
        {"time": "3 hours ago", "action": "Individual prediction", "status": "warning"},
        {"time": "1 day ago", "action": "Compliance report generated", "status": "success"}
    ]

    for activity in recent_activities:
        status_class = f"status-{activity['status']}"
        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid #ddd;">
            <span style="color: #666;">{activity['time']}</span> - 
            <span class="{status_class}">{activity['action']}</span>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Load CSS and initialize app
    load_css()
    initialize_app_state()

    # Render sidebar
    try:
        render_sidebar()
    except:
        # Fallback sidebar if component not available
        st.sidebar.title("Navigation")
        st.sidebar.info("Sidebar component not available. Please check component files.")

    # Main content area
    render_main_header()

    # Welcome message
    st.markdown("""
    ## 👋 Welcome to the Credit Default Prediction System

    This comprehensive platform helps financial institutions assess credit risk, 
    make informed lending decisions, and maintain regulatory compliance.

    ### 🎯 Key Features:
    - **AI-Powered Predictions**: Advanced machine learning models for accurate risk assessment
    - **Real-Time Analysis**: Instant individual customer risk evaluation
    - **Batch Processing**: Efficient processing of large customer datasets
    - **Interactive Dashboards**: Comprehensive data visualization and business intelligence
    - **Compliance Ready**: Built-in regulatory reporting and audit trails
    - **Customer Insights**: Advanced segmentation and behavioral analysis
    """)

    # Render main dashboard components
    render_quick_stats()

    st.markdown("---")

    # Two-column layout for navigation and activity
    col1, col2 = st.columns([2, 1])

    with col1:
        render_navigation_info()

    with col2:
        render_recent_activity()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Credit Default Prediction System v1.0.0</p>
        <p>Built with ❤️ using Streamlit | 
        <a href="https://github.com/MOHD-AFROZ-ALI/credit-default-test" target="_blank">GitHub</a> | 
        <a href="#" target="_blank">Documentation</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
