import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import sys
import os

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
    page_title="Individual Credit Risk Assessment",
    page_icon="👤",
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
    .risk-high {
        border-left-color: #e74c3c !important;
        background-color: #fdf2f2 !important;
    }
    .risk-medium {
        border-left-color: #f39c12 !important;
        background-color: #fef9e7 !important;
    }
    .risk-low {
        border-left-color: #27ae60 !important;
        background-color: #eafaf1 !important;
    }
    .input-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
"""

st.markdown(css_styles, unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">👤 Individual Credit Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "customer_data" not in st.session_state:
        st.session_state.customer_data = {}
    
    # Sidebar for navigation and quick actions
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.customer_data = {}
            st.session_state.prediction_result = None
            st.rerun()
        
        if st.button("📊 Load Sample Data", use_container_width=True):
            load_sample_data()
        
        st.markdown("### 📋 Assessment Status")
        if st.session_state.prediction_result:
            risk_score = st.session_state.prediction_result.get("risk_score", 0)
            risk_level = get_risk_level(risk_score)
            st.markdown(f"**Risk Level:** {risk_level}")
            st.markdown(f"**Risk Score:** {risk_score:.2f}")
        else:
            st.markdown("*No assessment completed*")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Customer Information Form
        render_customer_form()
        
        # Prediction button and results
        if st.button("🔍 Assess Credit Risk", type="primary", use_container_width=True):
            if validate_form_data():
                with st.spinner("Analyzing credit risk..."):
                    perform_risk_assessment()
            else:
                st.error("Please fill in all required fields.")
    
    with col2:
        # Risk Assessment Results
        if st.session_state.prediction_result:
            render_risk_results()
        else:
            render_assessment_guide()

def render_customer_form():
    """Render the customer information input form"""
    st.markdown('<div class="section-header">📝 Customer Information</div>', unsafe_allow_html=True)
    
    # Personal Information Section
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("**Personal Information**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.customer_data["age"] = st.number_input(
                "Age", 
                min_value=18, 
                max_value=100, 
                value=st.session_state.customer_data.get("age", 30),
                help="Customer's age in years"
            )
            
            st.session_state.customer_data["gender"] = st.selectbox(
                "Gender",
                options=["Male", "Female", "Other"],
                index=["Male", "Female", "Other"].index(st.session_state.customer_data.get("gender", "Male")),
                help="Customer's gender"
            )
        
        with col2:
            st.session_state.customer_data["education"] = st.selectbox(
                "Education Level",
                options=["High School", "Bachelor", "Master", "PhD", "Other"],
                index=["High School", "Bachelor", "Master", "PhD", "Other"].index(
                    st.session_state.customer_data.get("education", "Bachelor")
                ),
                help="Highest education level completed"
            )
            
            st.session_state.customer_data["marital_status"] = st.selectbox(
                "Marital Status",
                options=["Single", "Married", "Divorced", "Widowed"],
                index=["Single", "Married", "Divorced", "Widowed"].index(
                    st.session_state.customer_data.get("marital_status", "Single")
                ),
                help="Current marital status"
            )
        
        with col3:
            st.session_state.customer_data["dependents"] = st.number_input(
                "Number of Dependents",
                min_value=0,
                max_value=10,
                value=st.session_state.customer_data.get("dependents", 0),
                help="Number of financial dependents"
            )
            
            st.session_state.customer_data["employment_status"] = st.selectbox(
                "Employment Status",
                options=["Employed", "Self-Employed", "Unemployed", "Retired", "Student"],
                index=["Employed", "Self-Employed", "Unemployed", "Retired", "Student"].index(
                    st.session_state.customer_data.get("employment_status", "Employed")
                ),
                help="Current employment status"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial Information Section
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("**Financial Information**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.customer_data["annual_income"] = st.number_input(
                "Annual Income ($)",
                min_value=0,
                max_value=1000000,
                value=st.session_state.customer_data.get("annual_income", 50000),
                step=1000,
                help="Total annual income in USD"
            )
            
            st.session_state.customer_data["monthly_expenses"] = st.number_input(
                "Monthly Expenses ($)",
                min_value=0,
                max_value=50000,
                value=st.session_state.customer_data.get("monthly_expenses", 2000),
                step=100,
                help="Total monthly expenses in USD"
            )
            
            st.session_state.customer_data["existing_debt"] = st.number_input(
                "Existing Debt ($)",
                min_value=0,
                max_value=500000,
                value=st.session_state.customer_data.get("existing_debt", 0),
                step=1000,
                help="Total existing debt in USD"
            )
        
        with col2:
            st.session_state.customer_data["credit_history_length"] = st.number_input(
                "Credit History Length (years)",
                min_value=0,
                max_value=50,
                value=st.session_state.customer_data.get("credit_history_length", 5),
                help="Length of credit history in years"
            )
            
            st.session_state.customer_data["previous_defaults"] = st.number_input(
                "Previous Defaults",
                min_value=0,
                max_value=10,
                value=st.session_state.customer_data.get("previous_defaults", 0),
                help="Number of previous loan defaults"
            )
            
            st.session_state.customer_data["credit_utilization"] = st.slider(
                "Credit Utilization (%)",
                min_value=0,
                max_value=100,
                value=st.session_state.customer_data.get("credit_utilization", 30),
                help="Percentage of available credit currently used"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

def load_sample_data():
    """Load sample customer data for testing"""
    sample_data = {
        "age": 35,
        "gender": "Female",
        "education": "Bachelor",
        "marital_status": "Married",
        "dependents": 2,
        "employment_status": "Employed",
        "annual_income": 75000,
        "monthly_expenses": 3500,
        "existing_debt": 15000,
        "credit_history_length": 8,
        "previous_defaults": 0,
        "credit_utilization": 25
    }
    
    st.session_state.customer_data = sample_data
    st.success("Sample data loaded successfully!")
    st.rerun()

def validate_form_data():
    """Validate that all required form fields are filled"""
    required_fields = [
        "age", "gender", "education", "marital_status", "employment_status",
        "annual_income", "monthly_expenses", "credit_history_length"
    ]
    
    for field in required_fields:
        if field not in st.session_state.customer_data or st.session_state.customer_data[field] is None:
            return False
    
    return True

def perform_risk_assessment():
    """Perform the credit risk assessment"""
    try:
        # Calculate risk score (mock calculation for now)
        risk_score = calculate_mock_risk_score(st.session_state.customer_data)
        
        # Store results
        st.session_state.prediction_result = {
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score),
            "recommendation": get_recommendation(risk_score),
            "key_factors": get_key_risk_factors(st.session_state.customer_data),
            "timestamp": datetime.now()
        }
        
        st.success("Risk assessment completed successfully!")
        
    except Exception as e:
        st.error(f"Error during risk assessment: {str(e)}")

def calculate_mock_risk_score(data):
    """Calculate a mock risk score based on customer data"""
    score = 50  # Base score
    
    # Age factor
    if data["age"] < 25:
        score += 10
    elif data["age"] > 60:
        score += 5
    else:
        score -= 5
    
    # Income factor
    if data["annual_income"] < 30000:
        score += 15
    elif data["annual_income"] > 80000:
        score -= 10
    
    # Debt-to-income ratio
    monthly_income = data["annual_income"] / 12
    if monthly_income > 0:
        debt_ratio = data["monthly_expenses"] / monthly_income
        if debt_ratio > 0.5:
            score += 20
        elif debt_ratio < 0.3:
            score -= 10
    
    # Credit history
    if data["credit_history_length"] < 2:
        score += 15
    elif data["credit_history_length"] > 10:
        score -= 10
    
    # Previous defaults
    score += data["previous_defaults"] * 10
    
    # Credit utilization
    if data["credit_utilization"] > 80:
        score += 15
    elif data["credit_utilization"] < 30:
        score -= 5
    
    # Existing debt
    if data["existing_debt"] > data["annual_income"] * 0.5:
        score += 10
    
    return max(0, min(100, score))

def get_risk_level(score):
    """Determine risk level based on score"""
    if score >= 70:
        return "🔴 High Risk"
    elif score >= 40:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

def get_recommendation(score):
    """Get recommendation based on risk score"""
    if score >= 70:
        return "❌ Loan application should be rejected. High probability of default."
    elif score >= 40:
        return "⚠️ Loan application requires additional review and possibly higher interest rates."
    else:
        return "✅ Loan application can be approved with standard terms."

def get_key_risk_factors(data):
    """Identify key risk factors"""
    factors = []
    
    if data["annual_income"] < 30000:
        factors.append("Low annual income")
    
    if data["previous_defaults"] > 0:
        factors.append(f"{data['previous_defaults']} previous defaults")
    
    if data["credit_utilization"] > 80:
        factors.append("High credit utilization")
    
    if data["credit_history_length"] < 2:
        factors.append("Limited credit history")
    
    monthly_income = data["annual_income"] / 12
    if monthly_income > 0 and (data["monthly_expenses"] / monthly_income) > 0.5:
        factors.append("High debt-to-income ratio")
    
    if data["existing_debt"] > data["annual_income"] * 0.5:
        factors.append("High existing debt burden")
    
    return factors if factors else ["No significant risk factors identified"]

def render_risk_results():
    """Render the risk assessment results"""
    st.markdown('<div class="section-header">📊 Risk Assessment Results</div>', unsafe_allow_html=True)
    
    result = st.session_state.prediction_result
    risk_score = result["risk_score"]
    risk_level = result["risk_level"]
    
    # Risk level indicator
    risk_class = "risk-high" if risk_score >= 70 else "risk-medium" if risk_score >= 40 else "risk-low"
    
    risk_card_html = f"""
    <div class="metric-card {risk_class}">
        <h3>{risk_level}</h3>
        <h2>Risk Score: {risk_score:.1f}/100</h2>
        <p><strong>Recommendation:</strong> {result["recommendation"]}</p>
    </div>
    """
    
    st.markdown(risk_card_html, unsafe_allow_html=True)
    
    # Key risk factors
    st.markdown("**Key Risk Factors:**")
    for factor in result["key_factors"]:
        st.markdown(f"• {factor}")
    
    # Risk score gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {"x": [0, 1], "y": [0, 1]},
        title = {"text": "Risk Score"},
        delta = {"reference": 50},
        gauge = {
            "axis": {"range": [None, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 40], "color": "lightgreen"},
                {"range": [40, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_assessment_guide():
    """Render the assessment guide when no results are available"""
    st.markdown('<div class="section-header">📋 Assessment Guide</div>', unsafe_allow_html=True)
    
    guide_text = """
      ### How to Use This Tool
    
    1. **Fill in Customer Information**: Complete all required fields in the form
    2. **Review Data**: Ensure all information is accurate
    3. **Assess Risk**: Click the "Assess Credit Risk" button
    4. **Review Results**: Analyze the risk score and recommendations
    
    ### Risk Levels
    
    - 🟢 **Low Risk (0-39)**: Approve with standard terms
    - 🟡 **Medium Risk (40-69)**: Additional review required
    - 🔴 **High Risk (70-100)**: Recommend rejection
    
    ### Key Factors Considered
    
    - Personal demographics and employment
    - Income and expense ratios
    - Credit history and utilization
    - Existing debt burden
    - Previous default history
    """
    
    st.markdown(guide_text)

if __name__ == "__main__":
    main()