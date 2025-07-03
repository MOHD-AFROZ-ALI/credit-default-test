import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Data Exploration",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Data Exploration & Analysis")
st.markdown("---")

# Check if data exists in session state
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please upload data in the Data Upload section first.")
    st.stop()

df = st.session_state.data.copy()

# Sidebar for exploration options
st.sidebar.header("🔧 Exploration Options")
exploration_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Dataset Overview", "Statistical Analysis", "Distribution Analysis", 
     "Correlation Analysis", "Missing Data Analysis", "Outlier Detection"]
)

# Main content based on selection
if exploration_type == "Dataset Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{len(df.columns):,}")
    with col3:
        st.metric("Numeric Features", f"{len(df.select_dtypes(include=[np.number]).columns):,}")
    with col4:
        st.metric("Categorical Features", f"{len(df.select_dtypes(include=['object']).columns):,}")

    st.subheader("📋 Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif exploration_type == "Statistical Analysis":
    st.header("📈 Statistical Analysis")

    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        st.subheader("🔢 Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # Select column for detailed analysis
        selected_col = st.selectbox("Select column for detailed analysis", numeric_cols)

        if selected_col:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📊 {selected_col} - Statistics")
                stats_data = {
                    'Mean': df[selected_col].mean(),
                    'Median': df[selected_col].median(),
                    'Mode': df[selected_col].mode().iloc[0] if not df[selected_col].mode().empty else 'N/A',
                    'Standard Deviation': df[selected_col].std(),
                    'Variance': df[selected_col].var(),
                    'Skewness': df[selected_col].skew(),
                    'Kurtosis': df[selected_col].kurtosis(),
                    'Min': df[selected_col].min(),
                    'Max': df[selected_col].max(),
                    'Range': df[selected_col].max() - df[selected_col].min()
                }

                for stat, value in stats_data.items():
                    if isinstance(value, (int, float)):
                        st.metric(stat, f"{value:.4f}")
                    else:
                        st.metric(stat, str(value))

            with col2:
                st.subheader(f"📈 {selected_col} - Distribution")
                fig = px.histogram(df, x=selected_col, nbins=30, 
                                 title=f"Distribution of {selected_col}")
                fig.add_vline(x=df[selected_col].mean(), line_dash="dash", 
                            line_color="red", annotation_text="Mean")
                fig.add_vline(x=df[selected_col].median(), line_dash="dash", 
                            line_color="green", annotation_text="Median")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for statistical analysis.")

elif exploration_type == "Distribution Analysis":
    st.header("📊 Distribution Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    analysis_type = st.radio("Select Distribution Type", 
                           ["Numeric Distributions", "Categorical Distributions"])

    if analysis_type == "Numeric Distributions" and numeric_cols:
        selected_cols = st.multiselect("Select numeric columns", numeric_cols, 
                                     default=numeric_cols[:4])

        if selected_cols:
            # Create subplots
            n_cols = min(2, len(selected_cols))
            n_rows = (len(selected_cols) + n_cols - 1) // n_cols

            fig = make_subplots(rows=n_rows, cols=n_cols, 
                              subplot_titles=selected_cols)

            for i, col in enumerate(selected_cols):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1

                fig.add_trace(
                    go.Histogram(x=df[col], name=col, nbinsx=30),
                    row=row, col=col_pos
                )

            fig.update_layout(height=300*n_rows, showlegend=False,
                            title_text="Distribution of Selected Numeric Variables")
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Categorical Distributions" and categorical_cols:
        selected_col = st.selectbox("Select categorical column", categorical_cols)

        if selected_col:
            value_counts = df[selected_col].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Distribution of {selected_col}")
                fig.update_xaxes(title=selected_col)
                fig.update_yaxes(title="Count")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Proportion of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

elif exploration_type == "Correlation Analysis":
    st.header("🔗 Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()

        # Correlation heatmap
        fig = px.imshow(correlation_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Correlation Matrix Heatmap",
                       color_continuous_scale="RdBu_r")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # High correlation pairs
        st.subheader("🔍 High Correlation Pairs")
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for high correlation
                    correlation_pairs.append({
                        'Variable 1': correlation_matrix.columns[i],
                        'Variable 2': correlation_matrix.columns[j],
                        'Correlation': corr_value
                    })

        if correlation_pairs:
            corr_df = pd.DataFrame(correlation_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df, use_container_width=True)
        else:
            st.info("No high correlation pairs found (threshold: |r| > 0.5)")
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

elif exploration_type == "Missing Data Analysis":
    st.header("❓ Missing Data Analysis")

    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    if not missing_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(missing_df, x='Column', y='Missing Count',
                        title="Missing Values by Column")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                        title="Missing Values Percentage by Column")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Missing Data Summary")
        st.dataframe(missing_df, use_container_width=True)

        # Missing data pattern
        st.subheader("🔍 Missing Data Pattern")
        if len(missing_df) > 1:
            # Create a heatmap of missing data patterns
            missing_matrix = df[missing_df['Column']].isnull().astype(int)
            fig = px.imshow(missing_matrix.T, 
                           title="Missing Data Pattern",
                           labels=dict(x="Records", y="Variables"),
                           color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing data found in the dataset!")

elif exploration_type == "Outlier Detection":
    st.header("🎯 Outlier Detection")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)

        if selected_col:
            col1, col2 = st.columns(2)

            with col1:
                # Box plot
                fig = px.box(df, y=selected_col, title=f"Box Plot - {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Scatter plot with outliers highlighted
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                normal_data = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=normal_data.index, y=normal_data[selected_col],
                    mode='markers', name='Normal', marker=dict(color='blue', size=4)
                ))
                fig.add_trace(go.Scatter(
                    x=outliers.index, y=outliers[selected_col],
                    mode='markers', name='Outliers', marker=dict(color='red', size=6)
                ))
                fig.update_layout(title=f"Outlier Detection - {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

            # Outlier statistics
            st.subheader("📊 Outlier Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Outliers", len(outliers))
            with col2:
                st.metric("Outlier Percentage", f"{len(outliers)/len(df)*100:.2f}%")
            with col3:
                st.metric("Lower Bound", f"{lower_bound:.4f}")
            with col4:
                st.metric("Upper Bound", f"{upper_bound:.4f}")

            if len(outliers) > 0:
                st.subheader("🔍 Outlier Details")
                st.dataframe(outliers[[selected_col]], use_container_width=True)
    else:
        st.info("No numeric columns found for outlier analysis.")

# Export functionality
st.markdown("---")
st.subheader("💾 Export Analysis")
if st.button("Generate Analysis Report"):
    # Create a comprehensive analysis report
    report = {
        'dataset_overview': {
            'total_records': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns)
        },
        'missing_data': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }

    # Add numeric statistics if available
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report['descriptive_statistics'] = df[numeric_cols].describe().to_dict()

    st.success("✅ Analysis report generated successfully!")
    st.json(report)
