
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import json
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Compliance Report",
    page_icon="📋",
    layout="wide"
)

class ComplianceEngine:
    """Comprehensive Compliance and Fair Lending Analysis Engine"""
    
    def __init__(self):
        self.data = None
        self.predictions = None
        self.protected_attributes = []
        self.compliance_metrics = {}
        self.audit_log = []
        self.regulations = {
            'ECOA': 'Equal Credit Opportunity Act',
            'FCRA': 'Fair Credit Reporting Act', 
            'HMDA': 'Home Mortgage Disclosure Act',
            'CRA': 'Community Reinvestment Act',
            'GDPR': 'General Data Protection Regulation',
            'CCPA': 'California Consumer Privacy Act'
        }
        
    def load_data(self, data, predictions=None):
        """Load customer data and model predictions"""
        self.data = data.copy()
        if predictions is not None:
            self.predictions = predictions.copy()
        self._log_activity("Data loaded for compliance analysis")
        return True
        
    def _log_activity(self, activity, details=None):
        """Log compliance activities for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity': activity,
            'details': details or {},
            'user': 'system'  # In production, use actual user ID
        }
        self.audit_log.append(log_entry)
    
    def identify_protected_attributes(self, attributes):
        """Identify protected class attributes for fair lending analysis"""
        self.protected_attributes = attributes
        self._log_activity("Protected attributes identified", {'attributes': attributes})
        return True
    
    def calculate_demographic_parity(self, outcome_col, protected_attr):
        """Calculate demographic parity metrics"""
        if self.data is None or outcome_col not in self.data.columns:
            return None
            
        results = {}
        
        # Overall approval rate
        overall_rate = self.data[outcome_col].mean()
        results['overall_rate'] = overall_rate
        
        # Group-specific rates
        group_rates = {}
        for group in self.data[protected_attr].unique():
            group_data = self.data[self.data[protected_attr] == group]
            group_rate = group_data[outcome_col].mean()
            group_rates[str(group)] = group_rate
            
        results['group_rates'] = group_rates
        
        # Calculate disparate impact ratio
        if len(group_rates) >= 2:
            rates = list(group_rates.values())
            min_rate = min(rates)
            max_rate = max(rates)
            disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else 0
            results['disparate_impact_ratio'] = disparate_impact_ratio
            results['passes_80_percent_rule'] = disparate_impact_ratio >= 0.8
        
        self._log_activity("Demographic parity calculated", {
            'protected_attribute': protected_attr,
            'outcome': outcome_col
        })
        
        return results
    
    def calculate_equalized_odds(self, outcome_col, predicted_col, protected_attr):
        """Calculate equalized odds fairness metrics"""
        if self.data is None:
            return None
            
        results = {}
        
        for group in self.data[protected_attr].unique():
            group_data = self.data[self.data[protected_attr] == group]
            
            # True Positive Rate (Sensitivity)
            tp = len(group_data[(group_data[outcome_col] == 1) & (group_data[predicted_col] == 1)])
            fn = len(group_data[(group_data[outcome_col] == 1) & (group_data[predicted_col] == 0)])
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Positive Rate
            fp = len(group_data[(group_data[outcome_col] == 0) & (group_data[predicted_col] == 1)])
            tn = len(group_data[(group_data[outcome_col] == 0) & (group_data[predicted_col] == 0)])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results[str(group)] = {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'sample_size': len(group_data)
            }
        
        self._log_activity("Equalized odds calculated", {
            'protected_attribute': protected_attr
        })
        
        return results
    
    def perform_adverse_impact_analysis(self, outcome_col, protected_attrs):
        """Comprehensive adverse impact analysis"""
        adverse_impact_results = {}
        
        for attr in protected_attrs:
            if attr in self.data.columns:
                # Demographic parity analysis
                demo_parity = self.calculate_demographic_parity(outcome_col, attr)
                
                # Statistical significance test
                groups = []
                outcomes = []
                
                for group in self.data[attr].unique():
                    group_data = self.data[self.data[attr] == group]
                    groups.extend([group] * len(group_data))
                    outcomes.extend(group_data[outcome_col].tolist())
                
                # Chi-square test
                contingency_table = pd.crosstab(groups, outcomes)
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                adverse_impact_results[attr] = {
                    'demographic_parity': demo_parity,
                    'chi_square_statistic': chi2,
                    'p_value': p_value,
                    'statistically_significant': p_value < 0.05,
                    'contingency_table': contingency_table.to_dict()
                }
        
        self._log_activity("Adverse impact analysis completed")
        return adverse_impact_results
    
    def generate_hmda_report(self):
        """Generate HMDA (Home Mortgage Disclosure Act) compliance report"""
        if self.data is None:
            return None
            
        hmda_report = {
            'report_date': datetime.now().isoformat(),
            'total_applications': len(self.data),
            'reporting_period': f"{datetime.now().year}",
            'institution_info': {
                'name': 'Credit Default Prediction System',
                'type': 'Financial Institution'
            }
        }
        
        # Demographic breakdown
        demographic_fields = ['race', 'ethnicity', 'sex', 'age_group']
        available_demographics = [field for field in demographic_fields if field in self.data.columns]
        
        for field in available_demographics:
            hmda_report[f'{field}_breakdown'] = self.data[field].value_counts().to_dict()
        
        # Loan characteristics
        if 'loan_amount' in self.data.columns:
            hmda_report['loan_amount_stats'] = {
                'mean': float(self.data['loan_amount'].mean()),
                'median': float(self.data['loan_amount'].median()),
                'min': float(self.data['loan_amount'].min()),
                'max': float(self.data['loan_amount'].max())
            }
        
        # Action taken summary
        if 'loan_approved' in self.data.columns:
            hmda_report['action_taken'] = {
                'approved': int(self.data['loan_approved'].sum()),
                'denied': int(len(self.data) - self.data['loan_approved'].sum()),
                'approval_rate': float(self.data['loan_approved'].mean())
            }
        
        self._log_activity("HMDA report generated")
        return hmda_report
    
    def assess_model_fairness(self, outcome_col, predicted_col, protected_attrs):
        """Comprehensive model fairness assessment"""
        fairness_results = {
            'assessment_date': datetime.now().isoformat(),
            'metrics': {}
        }
        
        for attr in protected_attrs:
            if attr in self.data.columns:
                # Demographic parity
                demo_parity = self.calculate_demographic_parity(predicted_col, attr)
                
                # Equalized odds
                eq_odds = self.calculate_equalized_odds(outcome_col, predicted_col, attr)
                
                # Calibration analysis
                calibration_results = {}
                for group in self.data[attr].unique():
                    group_data = self.data[self.data[attr] == group]
                    if len(group_data) > 0:
                        # Calculate calibration (predicted vs actual rates)
                        predicted_rate = group_data[predicted_col].mean()
                        actual_rate = group_data[outcome_col].mean()
                        calibration_error = abs(predicted_rate - actual_rate)
                        
                        calibration_results[str(group)] = {
                            'predicted_rate': predicted_rate,
                            'actual_rate': actual_rate,
                            'calibration_error': calibration_error
                        }
                
                fairness_results['metrics'][attr] = {
                    'demographic_parity': demo_parity,
                    'equalized_odds': eq_odds,
                    'calibration': calibration_results
                }
        
        self._log_activity("Model fairness assessment completed")
        return fairness_results
    
    def generate_audit_trail(self):
        """Generate comprehensive audit trail"""
        return {
            'audit_period': {
                'start': min([entry['timestamp'] for entry in self.audit_log]) if self.audit_log else None,
                'end': max([entry['timestamp'] for entry in self.audit_log]) if self.audit_log else None
            },
            'total_activities': len(self.audit_log),
            'activities': self.audit_log
        }
    
    def check_regulatory_compliance(self):
        """Check compliance with various regulations"""
        compliance_status = {}
        
        # ECOA Compliance
        ecoa_compliant = True
        ecoa_issues = []
        
        if 'race' in self.data.columns or 'ethnicity' in self.data.columns:
            # Check for potential discrimination
            for attr in ['race', 'ethnicity']:
                if attr in self.data.columns and 'loan_approved' in self.data.columns:
                    demo_parity = self.calculate_demographic_parity('loan_approved', attr)
                    if demo_parity and not demo_parity.get('passes_80_percent_rule', True):
                        ecoa_compliant = False
                        ecoa_issues.append(f"Potential disparate impact detected for {attr}")
        
        compliance_status['ECOA'] = {
            'compliant': ecoa_compliant,
            'issues': ecoa_issues,
            'description': self.regulations['ECOA']
        }
        
        # GDPR Compliance (basic checks)
        gdpr_compliant = True
        gdpr_issues = []
        
        # Check for PII handling
        pii_columns = ['ssn', 'social_security', 'phone', 'email', 'address']
        found_pii = [col for col in pii_columns if col in self.data.columns]
        if found_pii:
            gdpr_issues.append(f"PII detected in dataset: {found_pii}")
        
        compliance_status['GDPR'] = {
            'compliant': gdpr_compliant,
            'issues': gdpr_issues,
            'description': self.regulations['GDPR']
        }
        
        self._log_activity("Regulatory compliance check completed")
        return compliance_status
    
    def generate_compliance_summary(self):
        """Generate executive compliance summary"""
        summary = {
            'report_date': datetime.now().isoformat(),
            'data_summary': {
                'total_records': len(self.data) if self.data is not None else 0,
                'protected_attributes': len(self.protected_attributes),
                'analysis_scope': 'Fair Lending and Regulatory Compliance'
            },
            'key_findings': [],
            'recommendations': [],
            'risk_level': 'Low'  # Will be updated based on findings
        }
        
        return summary

# Initialize the compliance engine
@st.cache_resource
def get_compliance_engine():
    return ComplianceEngine()

# Main page content
st.title("📋 Compliance Report")
st.markdown("### Regulatory Compliance and Fair Lending Analysis")

# Initialize session state
if 'compliance_data' not in st.session_state:
    st.session_state.compliance_data = None
if 'compliance_engine' not in st.session_state:
    st.session_state.compliance_engine = get_compliance_engine()

# Sidebar controls
st.sidebar.header("🔧 Compliance Controls")

# Data upload section
st.sidebar.subheader("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload loan/credit data",
    type=['csv', 'xlsx'],
    help="Upload CSV or Excel file with loan application data"
)

# Generate sample data if no file uploaded
if uploaded_file is None:
    if st.sidebar.button("🎲 Generate Sample Data"):
        np.random.seed(42)
        n_applications = 1000
        
        # Generate realistic demographic data
        races = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
        race_probs = [0.6, 0.15, 0.15, 0.08, 0.02]
        
        sample_data = pd.DataFrame({
            'application_id': range(1, n_applications + 1),
            'age': np.random.normal(40, 12, n_applications).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_applications).astype(int),
            'credit_score': np.random.normal(650, 100, n_applications).astype(int),
            'loan_amount': np.random.lognormal(9, 0.8, n_applications).astype(int),
            'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_applications),
            'employment_length': np.random.exponential(5, n_applications),
            'race': np.random.choice(races, n_applications, p=race_probs),
            'ethnicity': np.random.choice(['Hispanic', 'Non-Hispanic'], n_applications, p=[0.18, 0.82]),
            'sex': np.random.choice(['Male', 'Female'], n_applications, p=[0.52, 0.48]),
            'marital_status': np.random.choice(['Married', 'Single', 'Divorced'], n_applications, p=[0.5, 0.35, 0.15])
        })
        
        # Ensure realistic ranges
        sample_data['age'] = np.clip(sample_data['age'], 18, 80)
        sample_data['credit_score'] = np.clip(sample_data['credit_score'], 300, 850)
        sample_data['employment_length'] = np.clip(sample_data['employment_length'], 0, 40)
        
        # Create age groups
        sample_data['age_group'] = pd.cut(sample_data['age'], 
                                        bins=[0, 25, 35, 45, 55, 100], 
                                        labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Generate loan approval decisions with some bias for demonstration
        approval_prob = 0.7
        
        # Introduce subtle bias (for compliance testing)
        bias_factor = np.where(sample_data['race'] == 'White', 1.1, 
                              np.where(sample_data['race'] == 'Asian', 1.05, 0.95))
        
        credit_factor = (sample_data['credit_score'] - 300) / 550
        income_factor = np.log(sample_data['income']) / 15
        
        approval_scores = (credit_factor * 0.4 + income_factor * 0.3 + 
                          np.random.random(n_applications) * 0.3) * bias_factor
        
        sample_data['loan_approved'] = (approval_scores > 0.5).astype(int)
        sample_data['predicted_approval'] = (approval_scores > 0.48).astype(int)  # Slightly different threshold
        
        st.session_state.compliance_data = sample_data
        st.session_state.compliance_engine.load_data(sample_data)
        st.success("✅ Sample data generated successfully!")

# Load uploaded data
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.session_state.compliance_data = data
        st.session_state.compliance_engine.load_data(data)
        st.success(f"✅ Data loaded successfully! ({len(data)} rows, {len(data.columns)} columns)")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

# Main content area
if st.session_state.compliance_data is not None:
    data = st.session_state.compliance_data
    engine = st.session_state.compliance_engine
    
    # Data overview
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", len(data))
    with col2:
        if 'loan_approved' in data.columns:
            approval_rate = data['loan_approved'].mean() * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        else:
            st.metric("Approval Rate", "N/A")
    with col3:
        protected_attrs = ['race', 'ethnicity', 'sex', 'age_group']
        available_attrs = [attr for attr in protected_attrs if attr in data.columns]
        st.metric("Protected Attributes", len(available_attrs))
    with col4:
        st.metric("Data Quality", f"{((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100):.1f}%")
    
    # Protected attributes selection
    st.subheader("🛡️ Protected Attributes")
    
    all_categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    suggested_protected = ['race', 'ethnicity', 'sex', 'age_group', 'marital_status']
    available_protected = [col for col in suggested_protected if col in all_categorical_cols]
    
    selected_protected = st.multiselect(
        "Select protected class attributes for fair lending analysis:",
        all_categorical_cols,
        default=available_protected,
        help="Choose demographic attributes that are protected under fair lending laws"
    )
    
    if selected_protected:
        engine.identify_protected_attributes(selected_protected)
        
        # Outcome variable selection
        st.subheader("🎯 Outcome Variables")
        
        col1, col2 = st.columns(2)
        with col1:
            outcome_col = st.selectbox(
                "Select actual outcome variable:",
                [col for col in data.columns if data[col].dtype in ['int64', 'float64', 'bool']],
                index=0 if 'loan_approved' in data.columns else 0,
                help="The actual decision/outcome variable"
            )
        
        with col2:
            predicted_col = st.selectbox(
                "Select predicted outcome variable:",
                [col for col in data.columns if data[col].dtype in ['int64', 'float64', 'bool']],
                index=1 if len([col for col in data.columns if data[col].dtype in ['int64', 'float64', 'bool']]) > 1 else 0,
                help="The model's predicted outcome variable"
            )
        
        # Compliance Analysis
        st.subheader("⚖️ Fair Lending Analysis")
        
        if st.button("🔍 Run Compliance Analysis", type="primary"):
            with st.spinner("Performing comprehensive compliance analysis..."):
                
                # Store analysis results for export
                analysis_results = {}
                
                # Adverse Impact Analysis
                st.subheader("📊 Adverse Impact Analysis")
                
                adverse_impact = engine.perform_adverse_impact_analysis(outcome_col, selected_protected)
                analysis_results['adverse_impact'] = adverse_impact
                
                for attr, results in adverse_impact.items():
                    with st.expander(f"📈 {attr.title()} Analysis"):
                        demo_parity = results['demographic_parity']
                        
                        # Display group rates
                        st.markdown("**Group Approval Rates:**")
                        group_rates_df = pd.DataFrame([
                            {'Group': group, 'Approval Rate': f"{rate:.1%}", 'Rate Value': rate}
                            for group, rate in demo_parity['group_rates'].items()
                        ])
                        
                        # Color code based on disparate impact
                        fig_rates = px.bar(
                            group_rates_df, 
                            x='Group', 
                            y='Rate Value',
                            title=f'Approval Rates by {attr.title()}',
                            labels={'Rate Value': 'Approval Rate'}
                        )
                        fig_rates.update_traces(text=[f"{rate:.1%}" for rate in group_rates_df['Rate Value']], 
                                              textposition='outside')
                        fig_rates.update_layout(yaxis_tickformat='.1%')
                        st.plotly_chart(fig_rates, use_container_width=True)
                        
                        # 80% Rule Assessment
                        disparate_impact_ratio = demo_parity.get('disparate_impact_ratio', 0)
                        passes_80_rule = demo_parity.get('passes_80_percent_rule', False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Disparate Impact Ratio",
                                f"{disparate_impact_ratio:.3f}",
                                delta="Pass" if passes_80_rule else "Fail",
                                delta_color="normal" if passes_80_rule else "inverse"
                            )
                        
                        with col2:
                            st.metric(
                                "80% Rule Compliance",
                                "✅ Pass" if passes_80_rule else "❌ Fail",
                                help="Disparate impact ratio should be ≥ 0.80"
                            )
                        
                        # Statistical significance
                        p_value = results['p_value']
                        is_significant = results['statistically_significant']
                        
                        st.markdown("**Statistical Significance:**")
                        st.markdown(f"- Chi-square statistic: {results['chi_square_statistic']:.4f}")
                        st.markdown(f"- P-value: {p_value:.6f}")
                        st.markdown(f"- Statistically significant: {'Yes' if is_significant else 'No'} (α = 0.05)")
                        
                        if is_significant and not passes_80_rule:
                            st.error("⚠️ **Compliance Risk**: Statistically significant disparate impact detected!")
                        elif not passes_80_rule:
                            st.warning("⚠️ **Potential Risk**: Disparate impact ratio below 80% threshold")
                        else:
                            st.success("✅ **Compliant**: No significant disparate impact detected")
                
                # Model Fairness Assessment
                st.subheader("🤖 Model Fairness Assessment")
                
                fairness_results = engine.assess_model_fairness(outcome_col, predicted_col, selected_protected)
                analysis_results['fairness_assessment'] = fairness_results
                
                for attr, metrics in fairness_results['metrics'].items():
                    with st.expander(f"⚖️ {attr.title()} Fairness Metrics"):
                        
                        # Demographic Parity
                        st.markdown("**Demographic Parity (Predicted Outcomes):**")
                        demo_parity = metrics['demographic_parity']
                        
                        if demo_parity:
                            parity_df = pd.DataFrame([
                                {'Group': group, 'Predicted Rate': f"{rate:.1%}"}
                                for group, rate in demo_parity['group_rates'].items()
                            ])
                            st.dataframe(parity_df, use_container_width=True, hide_index=True)
                        
                        # Equalized Odds
                        st.markdown("**Equalized Odds:**")
                        eq_odds = metrics['equalized_odds']
                        
                        if eq_odds:
                            eq_odds_data = []
                            for group, rates in eq_odds.items():
                                eq_odds_data.append({
                                    'Group': group,
                                    'True Positive Rate': f"{rates['true_positive_rate']:.3f}",
                                    'False Positive Rate': f"{rates['false_positive_rate']:.3f}",
                                    'Sample Size': rates['sample_size']
                                })
                            
                            eq_odds_df = pd.DataFrame(eq_odds_data)
                            st.dataframe(eq_odds_df, use_container_width=True, hide_index=True)
                        
                        # Calibration
                        st.markdown("**Calibration Analysis:**")
                        calibration = metrics['calibration']
                        
                        if calibration:
                            cal_data = []
                            for group, cal_metrics in calibration.items():
                                cal_data.append({
                                    'Group': group,
                                    'Predicted Rate': f"{cal_metrics['predicted_rate']:.3f}",
                                    'Actual Rate': f"{cal_metrics['actual_rate']:.3f}",
                                    'Calibration Error': f"{cal_metrics['calibration_error']:.3f}"
                                })
                            
                            cal_df = pd.DataFrame(cal_data)
                            st.dataframe(cal_df, use_container_width=True, hide_index=True)
                            
                            # Calibration plot
                            fig_cal = go.Figure()
                            
                            for group, cal_metrics in calibration.items():
                                fig_cal.add_trace(go.Scatter(
                                    x=[cal_metrics['predicted_rate']],
                                    y=[cal_metrics['actual_rate']],
                                    mode='markers',
                                    name=group,
                                    marker=dict(size=10)
                                ))
                            
                            # Perfect calibration line
                            fig_cal.add_trace(go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode='lines',
                                name='Perfect Calibration',
                                line=dict(dash='dash', color='gray')
                            ))
                            
                            fig_cal.update_layout(
                                title='Model Calibration by Group',
                                xaxis_title='Predicted Rate',
                                yaxis_title='Actual Rate',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_cal, use_container_width=True)
                
                
                # Regulatory Compliance Check
                st.subheader("📋 Regulatory Compliance Status")

                compliance_status = engine.check_regulatory_compliance()
                analysis_results['regulatory_compliance'] = compliance_status

                for regulation, status in compliance_status.items():
                    with st.expander(f"📜 {regulation} - {status['description']}"):

                        if status['compliant']:
                            st.success(f"✅ **{regulation} Compliant**")
                        else:
                            st.error(f"❌ **{regulation} Non-Compliant**")

                        if status['issues']:
                            st.markdown("**Issues Identified:**")
                            for issue in status['issues']:
                                st.markdown(f"- {issue}")
                        else:
                            st.markdown("**No compliance issues detected.**")

                # HMDA Report Generation
                if st.checkbox("📊 Generate HMDA Report"):
                    st.subheader("🏠 HMDA Compliance Report")

                    hmda_report = engine.generate_hmda_report()
                    analysis_results['hmda_report'] = hmda_report

                    if hmda_report:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Report Summary:**")
                            st.json({
                                'Total Applications': hmda_report['total_applications'],
                                'Reporting Period': hmda_report['reporting_period'],
                                'Report Date': hmda_report['report_date'][:10]
                            })

                        with col2:
                            if 'action_taken' in hmda_report:
                                st.markdown("**Action Taken Summary:**")
                                action_data = hmda_report['action_taken']
                                st.json(action_data)

                        # Demographic breakdowns
                        demographic_fields = [key for key in hmda_report.keys() if key.endswith('_breakdown')]

                        if demographic_fields:
                            st.markdown("**Demographic Breakdowns:**")

                            for field in demographic_fields:
                                field_name = field.replace('_breakdown', '').title()
                                breakdown_data = hmda_report[field]

                                fig_demo = px.pie(
                                    values=list(breakdown_data.values()),
                                    names=list(breakdown_data.keys()),
                                    title=f'{field_name} Distribution'
                                )
                                st.plotly_chart(fig_demo, use_container_width=True)

                # Audit Trail
                st.subheader("📝 Audit Trail")

                audit_trail = engine.generate_audit_trail()
                analysis_results['audit_trail'] = audit_trail

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Activities", audit_trail['total_activities'])
                with col2:
                    if audit_trail['audit_period']['start']:
                        period_start = audit_trail['audit_period']['start'][:19]
                        st.metric("Audit Period Start", period_start)

                # Recent activities
                if audit_trail['activities']:
                    st.markdown("**Recent Activities:**")
                    recent_activities = audit_trail['activities'][-10:]  # Last 10 activities

                    activities_df = pd.DataFrame([
                        {
                            'Timestamp': activity['timestamp'][:19],
                            'Activity': activity['activity'],
                            'User': activity['user']
                        }
                        for activity in reversed(recent_activities)
                    ])

                    st.dataframe(activities_df, use_container_width=True, hide_index=True)

                # Export Reports
                st.subheader("📤 Export Reports")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Export adverse impact analysis
                    adverse_impact_json = json.dumps(adverse_impact, indent=2, default=str)
                    st.download_button(
                        label="📊 Download Adverse Impact Analysis",
                        data=adverse_impact_json,
                        file_name=f"adverse_impact_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                with col2:
                    # Export fairness assessment
                    fairness_json = json.dumps(fairness_results, indent=2, default=str)
                    st.download_button(
                        label="⚖️ Download Fairness Assessment",
                        data=fairness_json,
                        file_name=f"fairness_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                with col3:
                    # Export complete compliance report
                    complete_report = json.dumps(analysis_results, indent=2, default=str)
                    st.download_button(
                        label="📋 Download Complete Report",
                        data=complete_report,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                # Executive Summary
                st.subheader("📊 Executive Summary")

                # Calculate overall compliance score
                total_checks = len(compliance_status)
                compliant_checks = sum(1 for status in compliance_status.values() if status['compliant'])
                compliance_score = (compliant_checks / total_checks * 100) if total_checks > 0 else 0

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Overall Compliance Score",
                        f"{compliance_score:.1f}%",
                        delta="Good" if compliance_score >= 80 else "Needs Attention",
                        delta_color="normal" if compliance_score >= 80 else "inverse"
                    )

                with col2:
                    # Count disparate impact issues
                    disparate_impact_issues = 0
                    for attr, results in adverse_impact.items():
                        if not results['demographic_parity'].get('passes_80_percent_rule', True):
                            disparate_impact_issues += 1

                    st.metric(
                        "Disparate Impact Issues",
                        disparate_impact_issues,
                        delta="Critical" if disparate_impact_issues > 0 else "None",
                        delta_color="inverse" if disparate_impact_issues > 0 else "normal"
                    )

                with col3:
                    # Risk assessment
                    risk_level = "Low"
                    if disparate_impact_issues > 0 or compliance_score < 80:
                        risk_level = "High"
                    elif compliance_score < 90:
                        risk_level = "Medium"

                    st.metric(
                        "Risk Level",
                        risk_level,
                        delta="Action Required" if risk_level == "High" else "Monitor",
                        delta_color="inverse" if risk_level == "High" else "normal"
                    )

                # Key Recommendations
                st.markdown("**🎯 Key Recommendations:**")

                recommendations = []

                # Check for disparate impact issues
                for attr, results in adverse_impact.items():
                    if not results['demographic_parity'].get('passes_80_percent_rule', True):
                        recommendations.append(f"Address disparate impact in {attr} - consider model adjustments or additional review processes")

                # Check for compliance issues
                for regulation, status in compliance_status.items():
                    if not status['compliant']:
                        recommendations.append(f"Resolve {regulation} compliance issues - review policies and procedures")

                # General recommendations
                if not recommendations:
                    recommendations.append("Continue monitoring model performance and fairness metrics")
                    recommendations.append("Conduct regular compliance audits and reviews")
                    recommendations.append("Maintain documentation of all compliance activities")

                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")

    else:
        st.warning("⚠️ Please select protected attributes to perform compliance analysis")

else:
    st.info("📁 Please upload loan/credit data or generate sample data to begin compliance analysis")

    # Show example data format
    st.subheader("📋 Expected Data Format")

    example_data = pd.DataFrame({
        'application_id': [1, 2, 3, 4, 5],
        'age': [25, 35, 45, 55, 30],
        'income': [50000, 75000, 100000, 120000, 60000],
        'credit_score': [650, 720, 780, 800, 680],
        'loan_amount': [200000, 300000, 400000, 500000, 250000],
        'loan_approved': [1, 1, 1, 1, 0],
        'predicted_approval': [1, 1, 1, 1, 0],
        'race': ['White', 'Black', 'Hispanic', 'Asian', 'White'],
        'sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'ethnicity': ['Non-Hispanic', 'Non-Hispanic', 'Hispanic', 'Non-Hispanic', 'Non-Hispanic']
    })

    st.dataframe(example_data, use_container_width=True)

    st.markdown("""
    **Required columns for compliance analysis:**
    - **Outcome variables**: loan_approved, predicted_approval (binary: 0/1)
    - **Protected attributes**: race, ethnicity, sex, age_group, marital_status
    - **Application data**: application_id, age, income, credit_score, loan_amount

    **Compliance features:**
    - Adverse impact analysis (80% rule)
    - Fair lending metrics (demographic parity, equalized odds)
    - Regulatory compliance checks (ECOA, FCRA, HMDA, GDPR)
    - Model fairness assessment and calibration
    - Comprehensive audit trail and reporting
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Compliance Report Engine v2.0 | Fair Lending Analysis & Regulatory Compliance</p>
        <p>⚖️ Ensuring fairness and compliance in credit decisions</p>
    </div>
    """,
    unsafe_allow_html=True
)
