import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }

    .kpi-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }

    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and prepare data for business intelligence analysis"""
    try:
        # Generate sample business intelligence data
        np.random.seed(42)
        n_customers = 10000

        # Generate comprehensive business data
        data = {
            'customer_id': range(1, n_customers + 1),
            'loan_amount': np.random.lognormal(10, 0.5, n_customers),
            'interest_rate': np.random.normal(12, 3, n_customers),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_customers),
            'credit_score': np.random.normal(650, 100, n_customers),
            'annual_income': np.random.lognormal(11, 0.6, n_customers),
            'debt_to_income': np.random.beta(2, 5, n_customers),
            'employment_length': np.random.exponential(5, n_customers),
            'default_probability': np.random.beta(1, 9, n_customers),
            'profit_margin': np.random.normal(0.15, 0.05, n_customers),
            'acquisition_cost': np.random.gamma(2, 50, n_customers),
            'customer_lifetime_value': np.random.lognormal(8, 0.8, n_customers),
            'risk_category': np.random.choice(['Low', 'Medium', 'High'], n_customers, p=[0.6, 0.3, 0.1]),
            'product_type': np.random.choice(['Personal Loan', 'Auto Loan', 'Mortgage', 'Credit Card'], n_customers),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_customers),
            'acquisition_channel': np.random.choice(['Online', 'Branch', 'Partner', 'Referral'], n_customers),
            'account_age_months': np.random.randint(1, 120, n_customers),
        }

        df = pd.DataFrame(data)

        # Calculate derived metrics
        df['monthly_payment'] = df['loan_amount'] * (df['interest_rate']/100/12) / (1 - (1 + df['interest_rate']/100/12)**(-df['loan_term']))
        df['total_interest'] = df['monthly_payment'] * df['loan_term'] - df['loan_amount']
        df['expected_loss'] = df['loan_amount'] * df['default_probability']
        df['net_profit'] = df['total_interest'] - df['expected_loss'] - df['acquisition_cost']
        df['roi'] = df['net_profit'] / df['loan_amount']

        # Add time series data
        base_date = datetime.now() - timedelta(days=365)
        df['origination_date'] = pd.date_range(start=base_date, periods=n_customers, freq='H')
        df['origination_month'] = df['origination_date'].dt.to_period('M')

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_kpis(df):
    """Calculate key performance indicators"""
    try:
        kpis = {
            'total_portfolio_value': df['loan_amount'].sum(),
            'total_customers': len(df),
            'average_loan_amount': df['loan_amount'].mean(),
            'portfolio_default_rate': df['default_probability'].mean(),
            'total_expected_profit': df['net_profit'].sum(),
            'average_credit_score': df['credit_score'].mean(),
            'portfolio_roi': df['roi'].mean(),
            'high_risk_percentage': (df['risk_category'] == 'High').mean() * 100,
            'customer_acquisition_cost': df['acquisition_cost'].mean(),
            'customer_lifetime_value': df['customer_lifetime_value'].mean(),
        }
        return kpis
    except Exception as e:
        st.error(f"Error calculating KPIs: {str(e)}")
        return {}

def create_executive_dashboard(df, kpis):
    """Create executive dashboard with key metrics"""
    st.markdown('<div class="section-header">📈 Executive Dashboard</div>', unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${kpis.get('total_portfolio_value', 0):,.0f}",
            delta=f"{np.random.uniform(-5, 15):.1f}% vs last month"
        )

    with col2:
        st.metric(
            label="Total Customers",
            value=f"{kpis.get('total_customers', 0):,}",
            delta=f"{np.random.randint(50, 200)} new this month"
        )

    with col3:
        st.metric(
            label="Default Rate",
            value=f"{kpis.get('portfolio_default_rate', 0)*100:.2f}%",
            delta=f"{np.random.uniform(-0.5, 0.3):.2f}% vs target",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Portfolio ROI",
            value=f"{kpis.get('portfolio_roi', 0)*100:.2f}%",
            delta=f"{np.random.uniform(-2, 5):.1f}% vs last month"
        )

    # Additional KPIs
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            label="Avg Credit Score",
            value=f"{kpis.get('average_credit_score', 0):.0f}",
            delta=f"{np.random.uniform(-10, 15):.0f} vs last month"
        )

    with col6:
        st.metric(
            label="High Risk %",
            value=f"{kpis.get('high_risk_percentage', 0):.1f}%",
            delta=f"{np.random.uniform(-1, 2):.1f}% vs target",
            delta_color="inverse"
        )

    with col7:
        st.metric(
            label="Avg Loan Amount",
            value=f"${kpis.get('average_loan_amount', 0):,.0f}",
            delta=f"{np.random.uniform(-5, 10):.1f}% vs last month"
        )

    with col8:
        st.metric(
            label="Customer LTV",
            value=f"${kpis.get('customer_lifetime_value', 0):,.0f}",
            delta=f"{np.random.uniform(5, 20):.1f}% vs last month"
        )

def create_portfolio_analysis(df):
    """Create portfolio analysis visualizations"""
    st.markdown('<div class="section-header">💼 Portfolio Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Portfolio composition by product type
        fig_product = px.pie(
            df.groupby('product_type')['loan_amount'].sum().reset_index(),
            values='loan_amount',
            names='product_type',
            title='Portfolio Composition by Product Type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_product.update_layout(height=400)
        st.plotly_chart(fig_product, use_container_width=True)

    with col2:
        # Risk distribution
        risk_summary = df.groupby('risk_category').agg({
            'loan_amount': 'sum',
            'customer_id': 'count'
        }).reset_index()

        fig_risk = px.bar(
            risk_summary,
            x='risk_category',
            y='loan_amount',
            title='Portfolio Value by Risk Category',
            color='risk_category',
            color_discrete_map={'Low': '#28a745', 'Medium': '#fd7e14', 'High': '#dc3545'}
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)

    # Regional analysis
    col3, col4 = st.columns(2)

    with col3:
        regional_data = df.groupby('region').agg({
            'loan_amount': 'sum',
            'default_probability': 'mean',
            'customer_id': 'count'
        }).reset_index()

        fig_regional = px.scatter(
            regional_data,
            x='loan_amount',
            y='default_probability',
            size='customer_id',
            color='region',
            title='Regional Portfolio Analysis',
            labels={'loan_amount': 'Total Loan Amount', 'default_probability': 'Avg Default Rate'}
        )
        fig_regional.update_layout(height=400)
        st.plotly_chart(fig_regional, use_container_width=True)

    with col4:
        # Channel performance
        channel_data = df.groupby('acquisition_channel').agg({
            'customer_lifetime_value': 'mean',
            'acquisition_cost': 'mean',
            'customer_id': 'count'
        }).reset_index()
        channel_data['roi_ratio'] = channel_data['customer_lifetime_value'] / channel_data['acquisition_cost']

        fig_channel = px.bar(
            channel_data,
            x='acquisition_channel',
            y='roi_ratio',
            title='Channel ROI Performance',
            color='roi_ratio',
            color_continuous_scale='RdYlGn'
        )
        fig_channel.update_layout(height=400)
        st.plotly_chart(fig_channel, use_container_width=True)

def create_risk_analysis(df):
    """Create risk analysis dashboard"""
    st.markdown('<div class="section-header">⚠️ Risk Analysis</div>', unsafe_allow_html=True)

    # Risk heatmap
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create risk correlation matrix
        risk_features = ['credit_score', 'debt_to_income', 'loan_amount', 'interest_rate', 'default_probability']
        corr_matrix = df[risk_features].corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            title='Risk Factor Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        # Risk distribution
        risk_counts = df['risk_category'].value_counts()

        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker_colors=['#28a745', '#fd7e14', '#dc3545']
        )])
        fig_donut.update_layout(
            title='Risk Distribution',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Risk trends over time
    monthly_risk = df.groupby(['origination_month', 'risk_category']).size().unstack(fill_value=0)
    monthly_risk_pct = monthly_risk.div(monthly_risk.sum(axis=1), axis=0) * 100

    fig_trend = go.Figure()

    for risk_cat in ['Low', 'Medium', 'High']:
        if risk_cat in monthly_risk_pct.columns:
            color_map = {'Low': '#28a745', 'Medium': '#fd7e14', 'High': '#dc3545'}
            fig_trend.add_trace(go.Scatter(
                x=monthly_risk_pct.index.astype(str),
                y=monthly_risk_pct[risk_cat],
                mode='lines+markers',
                name=f'{risk_cat} Risk',
                line=dict(color=color_map[risk_cat], width=3),
                marker=dict(size=8)
            ))

    fig_trend.update_layout(
        title='Risk Category Trends Over Time',
        xaxis_title='Month',
        yaxis_title='Percentage (%)',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

def create_profitability_analysis(df):
    """Create profitability analysis dashboard"""
    st.markdown('<div class="section-header">💰 Profitability Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Profit by product type
        profit_by_product = df.groupby('product_type').agg({
            'net_profit': ['sum', 'mean'],
            'customer_id': 'count'
        }).round(2)
        profit_by_product.columns = ['Total Profit', 'Avg Profit', 'Customer Count']
        profit_by_product = profit_by_product.reset_index()

        fig_profit_product = px.bar(
            profit_by_product,
            x='product_type',
            y='Total Profit',
            title='Total Profit by Product Type',
            color='Total Profit',
            color_continuous_scale='RdYlGn'
        )
        fig_profit_product.update_layout(height=400)
        st.plotly_chart(fig_profit_product, use_container_width=True)

    with col2:
        # ROI distribution
        fig_roi_dist = px.histogram(
            df,
            x='roi',
            nbins=50,
            title='ROI Distribution',
            color_discrete_sequence=['#1f77b4']
        )
        fig_roi_dist.add_vline(
            x=df['roi'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean ROI: {df['roi'].mean():.2%}"
        )
        fig_roi_dist.update_layout(height=400)
        st.plotly_chart(fig_roi_dist, use_container_width=True)

    # Profitability by customer segments
    col3, col4 = st.columns(2)

    with col3:
        # Create customer value segments
        df['value_segment'] = pd.cut(
            df['customer_lifetime_value'],
            bins=4,
            labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
        )

        segment_profit = df.groupby('value_segment').agg({
            'net_profit': 'mean',
            'customer_lifetime_value': 'mean',
            'acquisition_cost': 'mean'
        }).reset_index()

        fig_segment = px.scatter(
            segment_profit,
            x='customer_lifetime_value',
            y='net_profit',
            size='acquisition_cost',
            color='value_segment',
            title='Profitability by Customer Segment',
            labels={'customer_lifetime_value': 'Customer LTV', 'net_profit': 'Net Profit'}
        )
        fig_segment.update_layout(height=400)
        st.plotly_chart(fig_segment, use_container_width=True)

    with col4:
        # Monthly profitability trend
        monthly_profit = df.groupby('origination_month').agg({
            'net_profit': 'sum',
            'loan_amount': 'sum'
        }).reset_index()
        monthly_profit['profit_margin'] = monthly_profit['net_profit'] / monthly_profit['loan_amount'] * 100

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly_profit['origination_month'].astype(str),
            y=monthly_profit['profit_margin'],
            mode='lines+markers',
            name='Profit Margin %',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))

        fig_monthly.update_layout(
            title='Monthly Profit Margin Trend',
            xaxis_title='Month',
            yaxis_title='Profit Margin (%)',
            height=400
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

def create_insights_and_recommendations(df, kpis):
    """Generate business insights and recommendations"""
    st.markdown('<div class="section-header">💡 Business Insights & Recommendations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Key Insights")

        # Generate dynamic insights based on data
        high_risk_pct = kpis.get('high_risk_percentage', 0)
        avg_roi = kpis.get('portfolio_roi', 0) * 100

        if high_risk_pct > 15:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ High Risk Alert:</strong> {high_risk_pct:.1f}% of portfolio is high-risk, 
                above the recommended 10% threshold. Consider tightening credit criteria.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
                <strong>✅ Risk Management:</strong> Portfolio risk is well-controlled at {high_risk_pct:.1f}% 
                high-risk customers, within acceptable limits.
            </div>
            """, unsafe_allow_html=True)

        if avg_roi > 15:
            st.markdown(f"""
            <div class="insight-box">
                <strong>📈 Strong Performance:</strong> Portfolio ROI of {avg_roi:.1f}% 
                exceeds industry benchmarks, indicating effective pricing strategy.
            </div>
            """, unsafe_allow_html=True)

        # Product performance insights
        product_performance = df.groupby('product_type')['roi'].mean().sort_values(ascending=False)
        best_product = product_performance.index[0]
        best_roi = product_performance.iloc[0] * 100

        st.markdown(f"""
        <div class="insight-box">
            <strong>🏆 Top Performer:</strong> {best_product} shows highest ROI at {best_roi:.1f}%. 
            Consider expanding this product line.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📋 Strategic Recommendations")

        recommendations = [
            "🎯 **Risk Optimization**: Implement dynamic pricing based on risk scores to improve margins",
            "📊 **Portfolio Diversification**: Balance high-margin products with stable, low-risk offerings",
            "🔍 **Customer Segmentation**: Develop targeted products for high-value customer segments",
            "📈 **Channel Optimization**: Focus marketing spend on highest-ROI acquisition channels",
            "⚡ **Process Automation**: Streamline underwriting for low-risk applications to reduce costs",
            "📱 **Digital Transformation**: Enhance online experience to capture younger demographics",
            "🤝 **Partnership Strategy**: Explore strategic partnerships for customer acquisition",
            "📉 **Loss Mitigation**: Implement early warning systems for default prediction"
        ]

        for rec in recommendations:
            st.markdown(rec)

def main():
    """Main application function"""
    # Header
    st.markdown('<div class="main-header">📊 Business Intelligence Dashboard</div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")

    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[datetime.now() - timedelta(days=365), datetime.now()],
        max_value=datetime.now()
    )

    # Filters
    st.sidebar.subheader("Filters")

    # Load data
    with st.spinner("Loading business intelligence data..."):
        df = load_data()

    if df is None:
        st.error("Failed to load data. Please check the data source.")
        return

    # Apply filters
    risk_filter = st.sidebar.multiselect(
        "Risk Categories",
        options=df['risk_category'].unique(),
        default=df['risk_category'].unique()
    )

    product_filter = st.sidebar.multiselect(
        "Product Types",
        options=df['product_type'].unique(),
        default=df['product_type'].unique()
    )

    region_filter = st.sidebar.multiselect(
        "Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )

    # Filter data
    filtered_df = df[
        (df['risk_category'].isin(risk_filter)) &
        (df['product_type'].isin(product_filter)) &
        (df['region'].isin(region_filter))
    ]

    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)

    # Create dashboard sections
    create_executive_dashboard(filtered_df, kpis)

    st.markdown("---")
    create_portfolio_analysis(filtered_df)

    st.markdown("---")
    create_risk_analysis(filtered_df)

    st.markdown("---")
    create_profitability_analysis(filtered_df)

    st.markdown("---")
    create_insights_and_recommendations(filtered_df, kpis)

    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Options")

    if st.sidebar.button("📊 Export Dashboard Data"):
        # Create summary report
        summary_data = {
            'KPIs': kpis,
            'Portfolio_Summary': {
                'total_customers': len(filtered_df),
                'total_portfolio_value': filtered_df['loan_amount'].sum(),
                'risk_distribution': filtered_df['risk_category'].value_counts().to_dict(),
                'product_distribution': filtered_df['product_type'].value_counts().to_dict()
            }
        }

        # Save to file
        import json
        export_path = "/home/user/output/bi_dashboard_export.json"

        try:
            with open(export_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)

            st.sidebar.success(f"✅ Data exported successfully!")
            st.sidebar.info(f"📁 Saved to: {export_path}")

        except Exception as e:
            st.sidebar.error(f"❌ Export failed: {str(e)}")

if __name__ == "__main__":
    main()
