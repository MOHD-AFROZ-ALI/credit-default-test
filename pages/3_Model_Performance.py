import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report, cross_val_score
)
from sklearn.model_selection import cross_validate
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Model Performance - Credit Default Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
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
        border-left: 5px solid #1f77b4;
    }

    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }

    .performance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
    }

    .comparison-card {
        background: white;
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .comparison-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.2);
    }

    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Model Performance Analysis</div>', unsafe_allow_html=True)

# Sidebar for model selection and options
st.sidebar.header("🔧 Performance Settings")

# Model selection
model_files = []
models_dir = "models"
if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') or f.endswith('.pkl')]

if not model_files:
    st.sidebar.warning("No trained models found. Please train a model first.")
    st.warning("⚠️ No trained models available. Please go to the Model Training page to train a model first.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select Model for Analysis",
    model_files,
    help="Choose a trained model to analyze its performance"
)

# Performance analysis options
st.sidebar.subheader("Analysis Options")
show_detailed_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=True)
show_confusion_matrix = st.sidebar.checkbox("Show Confusion Matrix", value=True)
show_roc_curve = st.sidebar.checkbox("Show ROC Curve", value=True)
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
show_cross_validation = st.sidebar.checkbox("Show Cross-Validation Results", value=True)
show_model_comparison = st.sidebar.checkbox("Enable Model Comparison", value=False)

# Comparison settings
if show_model_comparison and len(model_files) > 1:
    st.sidebar.subheader("Model Comparison")
    comparison_models = st.sidebar.multiselect(
        "Select Models to Compare",
        [f for f in model_files if f != selected_model],
        default=[f for f in model_files if f != selected_model][:2]
    )

# Load model and data function
@st.cache_data
def load_model_and_data(model_file):
    """Load trained model and associated data"""
    try:
        model_path = os.path.join("models", model_file)
        model_data = joblib.load(model_path)

        if isinstance(model_data, dict):
            model = model_data.get('model')
            X_test = model_data.get('X_test')
            y_test = model_data.get('y_test')
            feature_names = model_data.get('feature_names', [])
            model_name = model_data.get('model_name', 'Unknown')
            training_score = model_data.get('training_score', {})
        else:
            model = model_data
            X_test = None
            y_test = None
            feature_names = []
            model_name = 'Unknown'
            training_score = {}

        return model, X_test, y_test, feature_names, model_name, training_score
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, [], 'Unknown', {}

# Calculate performance metrics
def calculate_metrics(model, X_test, y_test):
    """Calculate comprehensive performance metrics"""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        return metrics, y_pred, y_pred_proba
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}, None, None

# Load selected model
with st.spinner("Loading model and calculating performance metrics..."):
    model, X_test, y_test, feature_names, model_name, training_score = load_model_and_data(selected_model)

if model is None or X_test is None or y_test is None:
    st.error("❌ Unable to load model or test data. Please ensure the model was saved with test data.")
    st.stop()

# Calculate metrics
metrics, y_pred, y_pred_proba = calculate_metrics(model, X_test, y_test)

if not metrics:
    st.error("❌ Unable to calculate performance metrics.")
    st.stop()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 Performance Overview - {model_name}")

    # Display key metrics
    if show_detailed_metrics:
        metric_cols = st.columns(5)

        with metric_cols[0]:
            st.markdown(f"""
            <div class="performance-card">
                <h3>Accuracy</h3>
                <h2>{metrics.get('accuracy', 0):.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[1]:
            st.markdown(f"""
            <div class="performance-card">
                <h3>Precision</h3>
                <h2>{metrics.get('precision', 0):.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[2]:
            st.markdown(f"""
            <div class="performance-card">
                <h3>Recall</h3>
                <h2>{metrics.get('recall', 0):.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[3]:
            st.markdown(f"""
            <div class="performance-card">
                <h3>F1-Score</h3>
                <h2>{metrics.get('f1', 0):.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[4]:
            roc_auc = metrics.get('roc_auc', 0)
            st.markdown(f"""
            <div class="performance-card">
                <h3>ROC-AUC</h3>
                <h2>{roc_auc:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.subheader("📋 Model Information")
    st.markdown(f"""
    <div class="info-box">
        <strong>Model:</strong> {model_name}<br>
        <strong>File:</strong> {selected_model}<br>
        <strong>Test Samples:</strong> {len(y_test)}<br>
        <strong>Features:</strong> {len(feature_names) if feature_names else 'Unknown'}<br>
        <strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)

# Detailed Classification Report
if show_detailed_metrics:
    st.subheader("📊 Detailed Classification Report")

    try:
        class_report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(class_report).transpose()

        # Format the dataframe for better display
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating classification report: {str(e)}")

# Visualization section
viz_col1, viz_col2 = st.columns(2)

# Confusion Matrix
if show_confusion_matrix:
    with viz_col1:
        st.subheader("🔍 Confusion Matrix")

        try:
            cm = confusion_matrix(y_test, y_pred)

            # Create confusion matrix heatmap
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                annotation_text=cm,
                colorscale='Blues',
                showscale=True
            )

            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )

            st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating confusion matrix: {str(e)}")

# ROC Curve
if show_roc_curve and y_pred_proba is not None:
    with viz_col2:
        st.subheader("📈 ROC Curve")

        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=2)
            ))

            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))

            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_roc, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating ROC curve: {str(e)}")

# Feature Importance
if show_feature_importance and feature_names:
    st.subheader("🎯 Feature Importance")

    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            fig_importance = px.bar(
                importance_df.tail(20),  # Show top 20 features
                x='importance',
                y='feature',
                orientation='h',
                title='Top 20 Feature Importances',
                labels={'importance': 'Importance Score', 'feature': 'Features'}
            )

            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)

        elif hasattr(model, 'coef_'):
            # For linear models
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            })
            importance_df['abs_coefficient'] = abs(importance_df['coefficient'])
            importance_df = importance_df.sort_values('abs_coefficient', ascending=True)

            fig_coef = px.bar(
                importance_df.tail(20),
                x='coefficient',
                y='feature',
                orientation='h',
                title='Top 20 Feature Coefficients',
                labels={'coefficient': 'Coefficient Value', 'feature': 'Features'},
                color='coefficient',
                color_continuous_scale='RdBu'
            )

            fig_coef.update_layout(height=600)
            st.plotly_chart(fig_coef, use_container_width=True)

        else:
            st.info("Feature importance not available for this model type.")

    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

# Cross-Validation Results
if show_cross_validation:
    st.subheader("🔄 Cross-Validation Analysis")

    try:
        with st.spinner("Performing cross-validation..."):
            # Combine X_test and y_test for cross-validation (in practice, you'd use full dataset)
            cv_scores = cross_validate(
                model, X_test, y_test,
                cv=5,
                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                return_train_score=True
            )

            cv_results = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Mean CV Score': [
                    cv_scores['test_accuracy'].mean(),
                    cv_scores['test_precision_weighted'].mean(),
                    cv_scores['test_recall_weighted'].mean(),
                    cv_scores['test_f1_weighted'].mean()
                ],
                'Std CV Score': [
                    cv_scores['test_accuracy'].std(),
                    cv_scores['test_precision_weighted'].std(),
                    cv_scores['test_recall_weighted'].std(),
                    cv_scores['test_f1_weighted'].std()
                ],
                'Mean Train Score': [
                    cv_scores['train_accuracy'].mean(),
                    cv_scores['train_precision_weighted'].mean(),
                    cv_scores['train_recall_weighted'].mean(),
                    cv_scores['train_f1_weighted'].mean()
                ]
            })

            cv_results = cv_results.round(4)
            st.dataframe(cv_results, use_container_width=True)

            # Visualize CV scores
            fig_cv = go.Figure()

            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            test_scores = [cv_scores['test_accuracy'], cv_scores['test_precision_weighted'], 
                          cv_scores['test_recall_weighted'], cv_scores['test_f1_weighted']]

            for i, (metric, scores) in enumerate(zip(metrics, test_scores)):
                fig_cv.add_trace(go.Box(
                    y=scores,
                    name=metric,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))

            fig_cv.update_layout(
                title='Cross-Validation Score Distribution',
                yaxis_title='Score',
                height=400
            )

            st.plotly_chart(fig_cv, use_container_width=True)

    except Exception as e:
        st.error(f"Error performing cross-validation: {str(e)}")

# Model Comparison
if show_model_comparison and len(model_files) > 1:
    st.subheader("⚖️ Model Comparison")

    if 'comparison_models' in locals() and comparison_models:
        comparison_data = []

        # Add current model
        comparison_data.append({
            'Model': model_name,
            'File': selected_model,
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1', 0),
            'ROC-AUC': metrics.get('roc_auc', 0)
        })

        # Add comparison models
        for comp_model_file in comparison_models:
            try:
                comp_model, comp_X_test, comp_y_test, _, comp_model_name, _ = load_model_and_data(comp_model_file)
                if comp_model and comp_X_test is not None and comp_y_test is not None:
                    comp_metrics, _, _ = calculate_metrics(comp_model, comp_X_test, comp_y_test)

                    comparison_data.append({
                        'Model': comp_model_name,
                        'File': comp_model_file,
                        'Accuracy': comp_metrics.get('accuracy', 0),
                        'Precision': comp_metrics.get('precision', 0),
                        'Recall': comp_metrics.get('recall', 0),
                        'F1-Score': comp_metrics.get('f1', 0),
                        'ROC-AUC': comp_metrics.get('roc_auc', 0)
                    })
            except Exception as e:
                st.warning(f"Could not load comparison model {comp_model_file}: {str(e)}")

        if len(comparison_data) > 1:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.round(4)

            # Display comparison table
            st.dataframe(comparison_df, use_container_width=True)

            # Create comparison chart
            metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

            fig_comparison = go.Figure()

            for metric in metrics_to_compare:
                fig_comparison.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    text=comparison_df[metric].round(3),
                    textposition='auto'
                ))

            fig_comparison.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Score',
                barmode='group',
                height=500
            )

            st.plotly_chart(fig_comparison, use_container_width=True)

            # Highlight best performing model
            best_model_idx = comparison_df['F1-Score'].idxmax()
            best_model = comparison_df.loc[best_model_idx, 'Model']

            st.markdown(f"""
            <div class="success-box">
                <strong>🏆 Best Performing Model:</strong> {best_model}<br>
                <strong>F1-Score:</strong> {comparison_df.loc[best_model_idx, 'F1-Score']:.4f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Select models to compare from the sidebar.")

# Performance Summary and Recommendations
st.subheader("📝 Performance Summary & Recommendations")

# Generate performance insights
performance_insights = []

if metrics.get('accuracy', 0) > 0.9:
    performance_insights.append("✅ Excellent accuracy - Model performs very well overall")
elif metrics.get('accuracy', 0) > 0.8:
    performance_insights.append("✅ Good accuracy - Model performs well")
else:
    performance_insights.append("⚠️ Moderate accuracy - Consider model improvement")

if metrics.get('roc_auc', 0) > 0.9:
    performance_insights.append("✅ Excellent ROC-AUC - Strong discriminative ability")
elif metrics.get('roc_auc', 0) > 0.8:
    performance_insights.append("✅ Good ROC-AUC - Good discriminative ability")
else:
    performance_insights.append("⚠️ Moderate ROC-AUC - Model may need improvement")

precision = metrics.get('precision', 0)
recall = metrics.get('recall', 0)

if precision > recall + 0.1:
    performance_insights.append("📊 High precision, lower recall - Model is conservative in predictions")
elif recall > precision + 0.1:
    performance_insights.append("📊 High recall, lower precision - Model captures most positive cases but with some false positives")
else:
    performance_insights.append("📊 Balanced precision and recall - Good overall performance")

# Display insights
for insight in performance_insights:
    st.markdown(f"- {insight}")

# Export functionality
st.subheader("💾 Export Results")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("📊 Export Performance Report"):
        try:
            # Create comprehensive report
            report_data = {
                'model_info': {
                    'name': model_name,
                    'file': selected_model,
                    'analysis_date': datetime.now().isoformat()
                },
                'performance_metrics': metrics,
                'insights': performance_insights
            }

            # Save report
            import json
            report_path = f"reports/performance_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("reports", exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            st.success(f"✅ Performance report exported to {report_path}")

        except Exception as e:
            st.error(f"Error exporting report: {str(e)}")

with export_col2:
    if st.button("📈 Export Visualizations"):
        try:
            # This would export the charts (implementation depends on requirements)
            st.info("📊 Visualization export functionality can be implemented based on specific requirements")

        except Exception as e:
            st.error(f"Error exporting visualizations: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Credit Default Prediction System - Model Performance Analysis</p>
    <p>Built with Streamlit • Last updated: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
