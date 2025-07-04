import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import sys
import os
import io
import zipfile

# Add the parent directory to the path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data.data_loader import UCICreditDefaultLoader
    from src.models.model_evaluator import ModelEvaluator
    from src.models.model_trainer import ModelTrainer
    from src.data.feature_engineering import CreditDefaultFeatureEngineer
    from src.models.risk_calculator import RiskCalculator
    from src.visualization.visualizations import  visualization_component
except ImportError as e:
    st.error(f"Error importing components: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Batch Credit Risk Processing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
css_styles = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 2px dashed #3498db;
    }
    .results-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 0.25rem;
        height: 1rem;
        overflow: hidden;
    }
    .progress-fill {
        background-color: #28a745;
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
"""

st.markdown(css_styles, unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Batch Credit Risk Processing</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if "batch_data" not in st.session_state:
        st.session_state.batch_data = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "ready"
    
    # Sidebar for batch operations
    with st.sidebar:
        st.markdown("### 📋 Batch Operations")
        
        # File upload section
        st.markdown("#### Upload Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file with customer data for batch processing"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Load Data", use_container_width=True):
                load_batch_data(uploaded_file)
        
        # Sample data option
        if st.button("📊 Load Sample Data", use_container_width=True):
            load_sample_batch_data()
        
        # Processing controls
        st.markdown("#### Processing Controls")
        
        if st.session_state.batch_data is not None:
            if st.button("🔍 Process Batch", type="primary", use_container_width=True):
                process_batch_predictions()
        
        if st.session_state.batch_results is not None:
            if st.button("📥 Download Results", use_container_width=True):
                download_results()
        
        # Clear data
        if st.button("🗑️ Clear All Data", use_container_width=True):
            clear_batch_data()
        
        # Batch status
        st.markdown("#### Batch Status")
        status_color = {
            "ready": "🟢",
            "processing": "🟡", 
            "completed": "🟢",
            "error": "🔴"
        }
        st.markdown(f"{status_color.get(st.session_state.processing_status, '⚪')} **Status:** {st.session_state.processing_status.title()}")
        
        if st.session_state.batch_data is not None:
            st.markdown(f"**Records:** {len(st.session_state.batch_data)}")
        
        if st.session_state.batch_results is not None:
            processed_count = len(st.session_state.batch_results)
            st.markdown(f"**Processed:** {processed_count}")
    
    # Main content area
    if st.session_state.batch_data is None:
        render_upload_section()
    else:
        render_data_preview()
        
        if st.session_state.batch_results is not None:
            render_batch_results()

def render_upload_section():
    """Render the file upload section"""
    st.markdown('<div class="section-header">📤 Data Upload</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### Upload Customer Data")
        st.markdown("""
        Upload a CSV file containing customer information for batch credit risk assessment.
        
        **Required Columns:**
        - age, gender, education, marital_status, employment_status
        - annual_income, monthly_expenses, existing_debt
        - credit_history_length, previous_defaults, credit_utilization
        
        **File Requirements:**
        - Format: CSV (.csv)
        - Maximum size: 50MB
        - Maximum records: 10,000
        """)
        
        # File upload widget (main one)
        uploaded_file = st.file_uploader(
            "Choose your CSV file",
            type=['csv'],
            key="main_uploader",
            help="Select a CSV file with customer data"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Load and Validate Data", type="primary"):
                load_batch_data(uploaded_file)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Quick Start")
        st.markdown("""
        **Option 1:** Upload your own CSV file
        
        **Option 2:** Use sample data to test the system
        """)
        
        if st.button("📊 Generate Sample Data", use_container_width=True):
            load_sample_batch_data()
        
        st.markdown("### 📖 Template")
        if st.button("📥 Download Template", use_container_width=True):
            download_template()

def render_data_preview():
    """Render the data preview section"""
    st.markdown('<div class="section-header">👀 Data Preview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(st.session_state.batch_data))
    
    with col2:
        st.metric("Columns", len(st.session_state.batch_data.columns))
    
    with col3:
        processed = len(st.session_state.batch_results) if st.session_state.batch_results is not None else 0
        st.metric("Processed", processed)
    
    with col4:
        remaining = len(st.session_state.batch_data) - processed
        st.metric("Remaining", remaining)
    
    # Data preview table
    st.markdown("### Data Sample")
    st.dataframe(
        st.session_state.batch_data.head(10),
        use_container_width=True,
        height=300
    )
    
    # Data validation
    st.markdown("### Data Validation")
    validation_results = validate_batch_data(st.session_state.batch_data)
    
    if validation_results["is_valid"]:
        st.success("✅ Data validation passed! Ready for processing.")
    else:
        st.error("❌ Data validation failed!")
        for error in validation_results["errors"]:
            st.error(f"• {error}")

def render_batch_results():
    """Render the batch processing results"""
    st.markdown('<div class="section-header">📊 Batch Results</div>', unsafe_allow_html=True)
    
    results_df = st.session_state.batch_results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk = len(results_df[results_df['risk_score'] >= 70])
        st.metric("High Risk", high_risk, delta=f"{high_risk/len(results_df)*100:.1f}%")
    
    with col2:
        medium_risk = len(results_df[(results_df['risk_score'] >= 40) & (results_df['risk_score'] < 70)])
        st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/len(results_df)*100:.1f}%")
    
    with col3:
        low_risk = len(results_df[results_df['risk_score'] < 40])
        st.metric("Low Risk", low_risk, delta=f"{low_risk/len(results_df)*100:.1f}%")
    
    with col4:
        avg_score = results_df['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_score:.1f}")
    
    # Risk distribution chart
    st.markdown("### Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level pie chart
        risk_counts = pd.cut(results_df['risk_score'], 
                           bins=[0, 40, 70, 100], 
                           labels=['Low Risk', 'Medium Risk', 'High Risk']).value_counts()
        
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color_discrete_map={
                'Low Risk': '#27ae60',
                'Medium Risk': '#f39c12', 
                'High Risk': '#e74c3c'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk score histogram
        fig_hist = px.histogram(
            results_df,
            x='risk_score',
            nbins=20,
            title="Risk Score Distribution",
            labels={'risk_score': 'Risk Score', 'count': 'Number of Customers'}
        )
        fig_hist.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Detailed results table
    st.markdown("### Detailed Results")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            options=["All", "Low Risk", "Medium Risk", "High Risk"]
        )
    
    with col2:
        score_range = st.slider(
            "Risk Score Range",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )
    
    with col3:
        show_details = st.checkbox("Show All Columns", value=False)
    
    # Apply filters
    filtered_results = results_df.copy()
    
    if risk_filter != "All":
        if risk_filter == "Low Risk":
            filtered_results = filtered_results[filtered_results['risk_score'] < 40]
        elif risk_filter == "Medium Risk":
            filtered_results = filtered_results[(filtered_results['risk_score'] >= 40) & (filtered_results['risk_score'] < 70)]
        elif risk_filter == "High Risk":
            filtered_results = filtered_results[filtered_results['risk_score'] >= 70]
    
    filtered_results = filtered_results[
        (filtered_results['risk_score'] >= score_range[0]) & 
        (filtered_results['risk_score'] <= score_range[1])
    ]
    
    # Select columns to display
    if show_details:
        display_columns = filtered_results.columns.tolist()
    else:
        display_columns = ['age', 'annual_income', 'risk_score', 'risk_level', 'recommendation']
        display_columns = [col for col in display_columns if col in filtered_results.columns]
    
    st.dataframe(
        filtered_results[display_columns],
        use_container_width=True,
        height=400
    )
    
    st.markdown(f"Showing {len(filtered_results)} of {len(results_df)} records")

def load_batch_data(uploaded_file):
    """Load and validate batch data from uploaded file"""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Basic validation
        if len(df) == 0:
            st.error("The uploaded file is empty.")
            return
        
        if len(df) > 10000:
            st.error("File contains too many records. Maximum allowed: 10,000")
            return
        
        # Store the data
        st.session_state.batch_data = df
        st.session_state.batch_results = None
        st.session_state.processing_status = "ready"
        
        st.success(f"✅ Successfully loaded {len(df)} records from {uploaded_file.name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

def load_sample_batch_data():
    """Generate sample batch data for testing"""
    np.random.seed(42)
    
    n_samples = 100
    
    sample_data = {
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
        'annual_income': np.random.randint(20000, 150000, n_samples),
        'monthly_expenses': np.random.randint(1000, 8000, n_samples),
        'existing_debt': np.random.randint(0, 100000, n_samples),
        'credit_history_length': np.random.randint(0, 30, n_samples),
        'previous_defaults': np.random.randint(0, 3, n_samples),
        'credit_utilization': np.random.randint(0, 100, n_samples)
    }
    
    st.session_state.batch_data = pd.DataFrame(sample_data)
    st.session_state.batch_results = None
    st.session_state.processing_status = "ready"
    
    st.success(f"✅ Generated {n_samples} sample records for testing")
    st.rerun()

def validate_batch_data(df):
    """Validate the batch data"""
    required_columns = [
        'age', 'gender', 'education', 'marital_status', 'employment_status',
        'annual_income', 'monthly_expenses', 'existing_debt',
        'credit_history_length', 'previous_defaults', 'credit_utilization'
    ]
    
    errors = []
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types and ranges
    if 'age' in df.columns:
        if df['age'].dtype not in ['int64', 'float64'] or df['age'].min() < 0 or df['age'].max() > 120:
            errors.append("Age values must be numeric and between 0-120")
    
    if 'annual_income' in df.columns:
        if df['annual_income'].dtype not in ['int64', 'float64'] or df['annual_income'].min() < 0:
            errors.append("Annual income must be non-negative numeric values")
    
    # Check for missing values in critical columns
    critical_columns = ['age', 'annual_income', 'monthly_expenses']
    for col in critical_columns:
        if col in df.columns and df[col].isnull().any():
            errors.append(f"Missing values found in critical column: {col}")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors
    }

def process_batch_predictions():
    """Process batch predictions for all records"""
    if st.session_state.batch_data is None:
        st.error("No data loaded for processing")
        return
    
    st.session_state.processing_status = "processing"
    
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_records = len(st.session_state.batch_data)
        
        for idx, row in st.session_state.batch_data.iterrows():
            # Update progress
            progress = (idx + 1) / total_records
            progress_bar.progress(progress)
            status_text.text(f"Processing record {idx + 1} of {total_records}")
            
            # Calculate risk score for this record
            risk_score = calculate_mock_risk_score(row.to_dict())
            
            # Create result record
            result = row.to_dict()
            result['risk_score'] = risk_score
            result['risk_level'] = get_risk_level(risk_score)
            result['recommendation'] = get_recommendation(risk_score)
            result['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            results.append(result)
        
        # Store results
        st.session_state.batch_results = pd.DataFrame(results)
        st.session_state.processing_status = "completed"
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully processed {total_records} records!")
        st.rerun()
        
    except Exception as e:
        st.session_state.processing_status = "error"
        st.error(f"Error during batch processing: {str(e)}")

def calculate_mock_risk_score(data):
    """Calculate a mock risk score based on customer data"""
    score = 50  # Base score
    
    # Age factor
    if data['age'] < 25:
        score += 10
    elif data['age'] > 60:
        score += 5
    else:
        score -= 5
    
    # Income factor
    if data['annual_income'] < 30000:
        score += 15
    elif data['annual_income'] > 80000:
        score -= 10
    
    # Debt-to-income ratio
    monthly_income = data['annual_income'] / 12
    if monthly_income > 0:
        debt_ratio = data['monthly_expenses'] / monthly_income
        if debt_ratio > 0.5:
            score += 20
        elif debt_ratio < 0.3:
            score -= 10
    
    # Credit history
    if data['credit_history_length'] < 2:
        score += 15
    elif data['credit_history_length'] > 10:
        score -= 10
    
    # Previous defaults
    score += data['previous_defaults'] * 10
    
    # Credit utilization
    if data['credit_utilization'] > 80:
        score += 15
    elif data['credit_utilization'] < 30:
        score -= 5
    
    # Existing debt
    if data['existing_debt'] > data['annual_income'] * 0.5:
        score += 10
    
    return max(0, min(100, score))

def get_risk_level(score):
    """Determine risk level based on score"""
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_recommendation(score):
    """Get recommendation based on risk score"""
    if score >= 70:
        return "Reject"
    elif score >= 40:
        return "Review Required"
    else:
        return "Approve"

def download_results():
    """Download batch processing results"""
    if st.session_state.batch_results is None:
        st.error("No results available for download")
        return
    
    # Convert results to CSV
    csv_buffer = io.StringIO()
    st.session_state.batch_results.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    # Create download button
    st.download_button(
        label="📥 Download Results CSV",
        data=csv_data,
        file_name=f"credit_risk_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def download_template():
    """Download CSV template for batch processing"""
    template_data = {
        'age': [35, 28, 45],
        'gender': ['Female', 'Male', 'Female'],
        'education': ['Bachelor', 'Master', 'High School'],
        'marital_status': ['Married', 'Single', 'Divorced'],
        'dependents': [2, 0, 1],
        'employment_status': ['Employed', 'Employed', 'Self-Employed'],
        'annual_income': [75000, 85000, 45000],
        'monthly_expenses': [3500, 2800, 2200],
        'existing_debt': [15000, 5000, 25000],
        'credit_history_length': [8, 5, 12],
        'previous_defaults': [0, 0, 1],
        'credit_utilization': [25, 15, 65]
    }
    
    template_df = pd.DataFrame(template_data)
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Template",
        data=csv_data,
        file_name="credit_risk_template.csv",
        mime="text/csv",
        use_container_width=True
    )

def clear_batch_data():
    """Clear all batch data and results"""
    st.session_state.batch_data = None
    st.session_state.batch_results = None
    st.session_state.processing_status = "ready"
    st.success("✅ All data cleared successfully!")
    st.rerun()

if __name__ == "__main__":
    main()