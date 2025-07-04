import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

class RiskCalculator:
    """Handles risk calculation and credit scoring."""
    
    def __init__(self):
        """Initialize RiskCalculator."""
        self.risk_weights = {
            'credit_score': 0.35,
            'debt_to_income': 0.25,
            'income': 0.20,
            'employment_length': 0.10,
            'age': 0.05,
            'loan_amount': 0.05
        }
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
    
    def calculate_credit_score_risk(self, credit_score):
        """Calculate risk based on credit score."""
        if credit_score >= 740:
            return 0.1  # Excellent
        elif credit_score >= 670:
            return 0.3  # Good
        elif credit_score >= 580:
            return 0.6  # Fair
        else:
            return 0.9  # Poor
    
    def calculate_debt_to_income_risk(self, debt_to_income):
        """Calculate risk based on debt-to-income ratio."""
        if debt_to_income <= 0.2:
            return 0.1  # Low risk
        elif debt_to_income <= 0.36:
            return 0.4  # Medium-low risk
        elif debt_to_income <= 0.5:
            return 0.7  # Medium-high risk
        else:
            return 0.9  # High risk
    
    def calculate_income_risk(self, income):
        """Calculate risk based on income level."""
        if income >= 80000:
            return 0.1  # High income, low risk
        elif income >= 50000:
            return 0.3  # Medium income
        elif income >= 30000:
            return 0.6  # Lower income
        else:
            return 0.8  # Low income, higher risk
    
    def calculate_employment_risk(self, employment_length):
        """Calculate risk based on employment length."""
        if employment_length >= 10:
            return 0.1  # Very stable
        elif employment_length >= 5:
            return 0.3  # Stable
        elif employment_length >= 2:
            return 0.6  # Somewhat stable
        else:
            return 0.8  # Unstable
    
    def calculate_age_risk(self, age):
        """Calculate risk based on age (experience factor)."""
        if 30 <= age <= 55:
            return 0.2  # Prime working age
        elif 25 <= age <= 65:
            return 0.4  # Good working age
        else:
            return 0.6  # Higher risk age groups
    
    def calculate_loan_amount_risk(self, loan_amount, income):
        """Calculate risk based on loan amount relative to income."""
        if income > 0:
            loan_to_income = loan_amount / income
            if loan_to_income <= 2:
                return 0.2  # Conservative loan
            elif loan_to_income <= 4:
                return 0.5  # Moderate loan
            elif loan_to_income <= 6:
                return 0.7  # Aggressive loan
            else:
                return 0.9  # Very aggressive loan
        else:
            return 0.9  # No income information
    
    def calculate_composite_risk_score(self, applicant_data):
        """Calculate composite risk score for an applicant."""
        risk_components = {}
        
        # Calculate individual risk components
        if 'credit_score' in applicant_data:
            risk_components['credit_score'] = self.calculate_credit_score_risk(applicant_data['credit_score'])
        
        if 'debt_to_income' in applicant_data:
            risk_components['debt_to_income'] = self.calculate_debt_to_income_risk(applicant_data['debt_to_income'])
        
        if 'income' in applicant_data:
            risk_components['income'] = self.calculate_income_risk(applicant_data['income'])
        
        if 'employment_length' in applicant_data:
            risk_components['employment_length'] = self.calculate_employment_risk(applicant_data['employment_length'])
        
        if 'age' in applicant_data:
            risk_components['age'] = self.calculate_age_risk(applicant_data['age'])
        
        if 'loan_amount' in applicant_data and 'income' in applicant_data:
            risk_components['loan_amount'] = self.calculate_loan_amount_risk(
                applicant_data['loan_amount'], applicant_data['income']
            )
        
        # Calculate weighted composite score
        composite_score = 0
        total_weight = 0
        
        for component, risk_value in risk_components.items():
            if component in self.risk_weights:
                weight = self.risk_weights[component]
                composite_score += risk_value * weight
                total_weight += weight
        
        # Normalize by total weight used
        if total_weight > 0:
            composite_score = composite_score / total_weight
        
        return {
            'composite_score': composite_score,
            'risk_components': risk_components,
            'risk_level': self.get_risk_level(composite_score)
        }
    
    def get_risk_level(self, risk_score):
        """Convert risk score to risk level."""
        if risk_score <= self.risk_thresholds['low']:
            return 'Low'
        elif risk_score <= self.risk_thresholds['medium']:
            return 'Medium'
        else:
            return 'High'
    
    def get_risk_color(self, risk_level):
        """Get color for risk level display."""
        colors = {
            'Low': 'green',
            'Medium': 'orange',
            'High': 'red'
        }
        return colors.get(risk_level, 'gray')
    
    def calculate_recommended_interest_rate(self, risk_score, base_rate=3.5):
        """Calculate recommended interest rate based on risk."""
        risk_premium = risk_score * 10  # Scale risk to percentage points
        recommended_rate = base_rate + risk_premium
        return min(recommended_rate, 25.0)  # Cap at 25%
    
    def generate_risk_explanation(self, risk_analysis):
        """Generate human-readable risk explanation."""
        score = risk_analysis['composite_score']
        level = risk_analysis['risk_level']
        components = risk_analysis['risk_components']
        
        explanation = f"**Risk Level: {level}** (Score: {score:.2f})\\n\\n"
        
        explanation += "**Risk Factors:**\\n"
        for component, risk_value in components.items():
            component_name = component.replace('_', ' ').title()
            risk_level = self.get_risk_level(risk_value)
            explanation += f"- {component_name}: {risk_level} ({risk_value:.2f})\\n"
        
        explanation += "\\n**Recommendations:**\\n"
        if level == 'Low':
            explanation += "- Approve loan with standard terms\\n"
            explanation += "- Consider offering premium rates\\n"
        elif level == 'Medium':
            explanation += "- Approve with careful monitoring\\n"
            explanation += "- Consider additional documentation\\n"
            explanation += "- Apply moderate risk premium\\n"
        else:
            explanation += "- High risk - consider rejection\\n"
            explanation += "- If approved, require significant risk premium\\n"
            explanation += "- Implement strict monitoring\\n"
        
        return explanation

def risk_assessment_component():
    """Streamlit component for risk assessment interface."""
    st.subheader("⚠️ Risk Assessment")
    
    calculator = RiskCalculator()
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Manual Input", "Use Loaded Data", "Batch Assessment"]
    )
    
    if input_method == "Manual Input":
        st.write("**Enter applicant information:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01)
            income = st.number_input("Annual Income ($)", 0, 200000, 50000)
        
        with col2:
            employment_length = st.slider("Employment Length (years)", 0, 40, 5)
            age = st.slider("Age", 18, 80, 35)
            loan_amount = st.number_input("Loan Amount ($)", 0, 100000, 25000)
        
        applicant_data = {
            'credit_score': credit_score,
            'debt_to_income': debt_to_income,
            'income': income,
            'employment_length': employment_length,
            'age': age,
            'loan_amount': loan_amount
        }
        
        if st.button("Calculate Risk"):
            risk_analysis = calculator.calculate_composite_risk_score(applicant_data)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_color = calculator.get_risk_color(risk_analysis['risk_level'])
                st.metric(
                    "Risk Level", 
                    risk_analysis['risk_level'],
                    delta=f"Score: {risk_analysis['composite_score']:.2f}"
                )
            
            with col2:
                recommended_rate = calculator.calculate_recommended_interest_rate(
                    risk_analysis['composite_score']
                )
                st.metric("Recommended Rate", f"{recommended_rate:.2f}%")
            
            with col3:
                approval_recommendation = "Approve" if risk_analysis['risk_level'] != 'High' else "Reject"
                st.metric("Recommendation", approval_recommendation)
            
            # Detailed explanation
            st.subheader("📋 Risk Analysis Details")
            explanation = calculator.generate_risk_explanation(risk_analysis)
            st.markdown(explanation)
            
            # Risk components chart
            if risk_analysis['risk_components']:
                st.subheader("📊 Risk Components")
                components_df = pd.DataFrame([
                    {'Component': k.replace('_', ' ').title(), 'Risk Score': v}
                    for k, v in risk_analysis['risk_components'].items()
                ])
                st.bar_chart(components_df.set_index('Component'))
    
    elif input_method == "Use Loaded Data":
        if 'data' not in st.session_state:
            st.warning("Please load data first")
        else:
            data = st.session_state['data']
            
            # Select row for assessment
            row_index = st.selectbox("Select row to assess:", range(len(data)))
            
            if st.button("Assess Selected Row"):
                row_data = data.iloc[row_index].to_dict()
                risk_analysis = calculator.calculate_composite_risk_score(row_data)
                
                # Display results similar to manual input
                st.subheader("📊 Risk Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Level", risk_analysis['risk_level'])
                with col2:
                    st.metric("Risk Score", f"{risk_analysis['composite_score']:.2f}")
                with col3:
                    recommended_rate = calculator.calculate_recommended_interest_rate(
                        risk_analysis['composite_score']
                    )
                    st.metric("Recommended Rate", f"{recommended_rate:.2f}%")
                
                explanation = calculator.generate_risk_explanation(risk_analysis)
                st.markdown(explanation)
    
    elif input_method == "Batch Assessment":
        if 'data' not in st.session_state:
            st.warning("Please load data first")
        else:
            data = st.session_state['data']
            
            if st.button("Assess All Records"):
                with st.spinner("Calculating risk for all records..."):
                    risk_results = []
                    
                    for idx, row in data.iterrows():
                        row_data = row.to_dict()
                        risk_analysis = calculator.calculate_composite_risk_score(row_data)
                        
                        risk_results.append({
                            'Index': idx,
                            'Risk_Score': risk_analysis['composite_score'],
                            'Risk_Level': risk_analysis['risk_level'],
                            'Recommended_Rate': calculator.calculate_recommended_interest_rate(
                                risk_analysis['composite_score']
                            )
                        })
                    
                    results_df = pd.DataFrame(risk_results)
                    st.session_state['risk_results'] = results_df
                    
                    # Display summary
                    st.subheader("📊 Batch Risk Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Assessed", len(results_df))
                    with col2:
                        high_risk_count = len(results_df[results_df['Risk_Level'] == 'High'])
                        st.metric("High Risk", high_risk_count)
                    with col3:
                        avg_score = results_df['Risk_Score'].mean()
                        st.metric("Average Risk Score", f"{avg_score:.2f}")
                    
                    # Risk distribution
                    st.subheader("📈 Risk Distribution")
                    risk_counts = results_df['Risk_Level'].value_counts()
                    st.bar_chart(risk_counts)
                    
                    # Show results table
                    if st.checkbox("Show detailed results"):
                        st.dataframe(results_df)
    
    return calculator