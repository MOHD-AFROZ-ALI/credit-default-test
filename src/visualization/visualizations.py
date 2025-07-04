import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """Handles data visualization and plotting."""
    
    def __init__(self):
        """Initialize DataVisualizer."""
        self.color_palette = px.colors.qualitative.Set3
        self.default_template = "plotly_white"
    
    def plot_distribution(self, data, column, title=None):
        """Plot distribution of a single column."""
        if column not in data.columns:
            st.error(f"Column '{column}' not found in data")
            return None
        
        title = title or f"Distribution of {column}"
        
        if data[column].dtype in ['object', 'category']:
            # Categorical data - bar chart
            value_counts = data[column].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=title,
                labels={'x': column, 'y': 'Count'},
                template=self.default_template
            )
        else:
            # Numerical data - histogram
            fig = px.histogram(
                data,
                x=column,
                title=title,
                template=self.default_template,
                marginal="box"
            )
        
        return fig
    
    def plot_correlation_matrix(self, data, title="Correlation Matrix"):
        """Plot correlation matrix for numerical columns."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            st.error("No numerical columns found for correlation analysis")
            return None
        
        correlation_matrix = numeric_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title=title,
            template=self.default_template,
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        return fig
    
    def plot_scatter(self, data, x_col, y_col, color_col=None, title=None):
        """Create scatter plot between two columns."""
        if x_col not in data.columns or y_col not in data.columns:
            st.error(f"Columns not found in data")
            return None
        
        title = title or f"{y_col} vs {x_col}"
        
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            template=self.default_template,
            hover_data=data.columns[:5].tolist()  # Show first 5 columns on hover
        )
        
        return fig
    
    def plot_box_plot(self, data, x_col, y_col, title=None):
        """Create box plot for categorical vs numerical data."""
        title = title or f"{y_col} by {x_col}"
        
        fig = px.box(
            data,
            x=x_col,
            y=y_col,
            title=title,
            template=self.default_template
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df, title="Feature Importance", top_n=15):
        """Plot feature importance."""
        if importance_df is None or importance_df.empty:
            st.error("No feature importance data provided")
            return None
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            template=self.default_template
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, top_n * 25)
        )
        
        return fig
    
    def plot_prediction_distribution(self, predictions, probabilities=None, title="Prediction Distribution"):
        """Plot distribution of predictions."""
        pred_counts = pd.Series(predictions).value_counts()
        
        fig = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title=title,
            template=self.default_template
        )
        
        return fig
    
    def plot_probability_distribution(self, probabilities, title="Probability Distribution"):
        """Plot distribution of prediction probabilities."""
        if probabilities is None:
            st.error("No probability data provided")
            return None
        
        fig = px.histogram(
            x=probabilities,
            title=title,
            labels={'x': 'Probability', 'y': 'Count'},
            template=self.default_template,
            marginal="box"
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            title=title,
            labels=dict(x="Predicted", y="Actual"),
            x=['No Default', 'Default'],
            y=['No Default', 'Default'],
            template=self.default_template,
            color_continuous_scale="Blues"
        )
        
        # Add text annotations
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(cm[i][j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
                )
        
        return fig
    
    def plot_roc_curve(self, y_true, y_prob, title="ROC Curve"):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.2f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template=self.default_template
        )
        
        return fig
    
    def plot_risk_distribution(self, risk_data, title="Risk Score Distribution"):
        """Plot risk score distribution."""
        if isinstance(risk_data, dict):
            # Convert dict to DataFrame
            risk_df = pd.DataFrame([risk_data])
        else:
            risk_df = risk_data
        
        if 'Risk_Score' in risk_df.columns:
            fig = px.histogram(
                risk_df,
                x='Risk_Score',
                title=title,
                template=self.default_template,
                marginal="box"
            )
        else:
            st.error("Risk_Score column not found")
            return None
        
        return fig
    
    def create_dashboard_summary(self, data, predictions=None, risk_scores=None):
        """Create a comprehensive dashboard summary."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Overview', 'Target Distribution', 
                          'Feature Correlations', 'Risk Analysis'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "heatmap"}, {"type": "histogram"}]]
        )
        
        # Data overview - column types
        numeric_count = len(data.select_dtypes(include=[np.number]).columns)
        categorical_count = len(data.select_dtypes(include=['object']).columns)
        
        fig.add_trace(
            go.Bar(x=['Numeric', 'Categorical'], y=[numeric_count, categorical_count]),
            row=1, col=1
        )
        
        # Target distribution
        if 'default' in data.columns:
            target_counts = data['default'].value_counts()
            fig.add_trace(
                go.Pie(labels=target_counts.index, values=target_counts.values),
                row=1, col=2
            )
        
        # Feature correlations (simplified)
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.columns,
                          colorscale='RdBu'),
                row=2, col=1
            )
        
        # Risk scores distribution
        if risk_scores is not None:
            fig.add_trace(
                go.Histogram(x=risk_scores),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Data Analysis Dashboard",
            template=self.default_template,
            height=800
        )
        
        return fig

def visualization_component():
    """Streamlit component for data visualization interface."""
    st.subheader("📊 Data Visualization")
    
    if 'data' not in st.session_state:
        st.warning("Please load data first")
        return None
    
    data = st.session_state['data']
    visualizer = DataVisualizer()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select visualization type:",
        [
            "Distribution Plot",
            "Correlation Matrix", 
            "Scatter Plot",
            "Box Plot",
            "Feature Importance",
            "Prediction Analysis",
            "Risk Analysis",
            "Dashboard Summary"
        ]
    )
    
    if viz_type == "Distribution Plot":
        column = st.selectbox("Select column:", data.columns)
        if st.button("Generate Plot"):
            fig = visualizer.plot_distribution(data, column)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Matrix":
        if st.button("Generate Correlation Matrix"):
            fig = visualizer.plot_correlation_matrix(data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", data.columns)
        with col2:
            y_col = st.selectbox("Y-axis:", data.columns)
        
        color_col = st.selectbox("Color by (optional):", [None] + list(data.columns))
        
        if st.button("Generate Scatter Plot"):
            fig = visualizer.plot_scatter(data, x_col, y_col, color_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Categorical column:", data.select_dtypes(include=['object']).columns)
        with col2:
            y_col = st.selectbox("Numerical column:", data.select_dtypes(include=[np.number]).columns)
        
        if st.button("Generate Box Plot"):
            fig = visualizer.plot_box_plot(data, x_col, y_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Feature Importance":
        if 'model_manager' in st.session_state:
            model_manager = st.session_state['model_manager']
            if model_manager.is_trained:
                importance_df = model_manager.get_feature_importance()
                if importance_df is not None:
                    top_n = st.slider("Number of top features:", 5, 20, 10)
                    fig = visualizer.plot_feature_importance(importance_df, top_n=top_n)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Feature importance not available for this model")
            else:
                st.warning("Please train a model first")
        else:
            st.warning("No model manager found")
    
    elif viz_type == "Prediction Analysis":
        if 'model_manager' in st.session_state:
            model_manager = st.session_state['model_manager']
            if model_manager.is_trained:
                if st.button("Analyze Predictions"):
                    predictions_result = model_manager.predict(data)
                    if predictions_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig1 = visualizer.plot_prediction_distribution(predictions_result['predictions'])
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            fig2 = visualizer.plot_probability_distribution(predictions_result['probability_default'])
                            st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Please train a model first")
        else:
            st.warning("No model manager found")
    
    elif viz_type == "Risk Analysis":
        if 'risk_results' in st.session_state:
            risk_results = st.session_state['risk_results']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = visualizer.plot_risk_distribution(risk_results)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Risk level distribution
                risk_counts = risk_results['Risk_Level'].value_counts()
                fig2 = px.pie(values=risk_counts.values, names=risk_counts.index, 
                             title="Risk Level Distribution")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Please run risk assessment first")
    
    elif viz_type == "Dashboard Summary":
        if st.button("Generate Dashboard"):
            predictions = None
            risk_scores = None
            
            if 'model_manager' in st.session_state and st.session_state['model_manager'].is_trained:
                pred_result = st.session_state['model_manager'].predict(data)
                if pred_result:
                    predictions = pred_result['predictions']
            
            if 'risk_results' in st.session_state:
                risk_scores = st.session_state['risk_results']['Risk_Score'].values
            
            fig = visualizer.create_dashboard_summary(data, predictions, risk_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    return visualizer