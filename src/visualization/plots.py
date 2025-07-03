"""
Credit Default Prediction System - Visualization Module
Part 1: Basic plotting utilities and chart creation functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style configurations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PlotConfig:
    """Configuration class for consistent plotting styles"""

    # Color schemes
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#F18F01',
        'danger': '#C73E1D',
        'warning': '#FFB400',
        'info': '#17A2B8',
        'light': '#F8F9FA',
        'dark': '#343A40'
    }

    # Figure sizes
    FIGSIZE = {
        'small': (8, 6),
        'medium': (12, 8),
        'large': (16, 10),
        'wide': (20, 8)
    }

    # Font configurations
    FONTS = {
        'title': {'size': 16, 'weight': 'bold'},
        'subtitle': {'size': 14, 'weight': 'normal'},
        'label': {'size': 12, 'weight': 'normal'},
        'text': {'size': 10, 'weight': 'normal'}
    }

def setup_plot_style():
    """Setup consistent plotting style across all visualizations"""
    plt.rcParams.update({
        'figure.figsize': PlotConfig.FIGSIZE['medium'],
        'axes.titlesize': PlotConfig.FONTS['title']['size'],
        'axes.labelsize': PlotConfig.FONTS['label']['size'],
        'xtick.labelsize': PlotConfig.FONTS['text']['size'],
        'ytick.labelsize': PlotConfig.FONTS['text']['size'],
        'legend.fontsize': PlotConfig.FONTS['text']['size'],
        'figure.titlesize': PlotConfig.FONTS['title']['size']
    })

def save_plot(fig, filename, output_dir='/home/user/output/visualizations', 
              formats=['png', 'pdf'], dpi=300):
    """
    Save plot in multiple formats with consistent naming

    Args:
        fig: matplotlib figure object
        filename: base filename without extension
        output_dir: directory to save plots
        formats: list of formats to save
        dpi: resolution for raster formats
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight')

    print(f"✅ Plot saved: {filename}")

def create_confusion_matrix_plot(y_true, y_pred, class_names=None, 
                               title="Confusion Matrix", save_name=None):
    """
    Create an enhanced confusion matrix visualization

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes
        title: Plot title
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=PlotConfig.FIGSIZE['small'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or ['No Default', 'Default'],
                yticklabels=class_names or ['No Default', 'Default'],
                ax=ax)

    # Customize plot
    ax.set_title(title, **PlotConfig.FONTS['title'])
    ax.set_xlabel('Predicted Label', **PlotConfig.FONTS['label'])
    ax.set_ylabel('True Label', **PlotConfig.FONTS['label'])

    # Add accuracy metrics
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
            transform=ax.transAxes, ha='center')

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def create_roc_curve_plot(y_true, y_prob, title="ROC Curve", save_name=None):
    """
    Create ROC curve visualization

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        title: Plot title
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=PlotConfig.FIGSIZE['small'])

    # Plot ROC curve
    ax.plot(fpr, tpr, color=PlotConfig.COLORS['primary'], 
            lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color=PlotConfig.COLORS['danger'], 
            lw=2, linestyle='--', label='Random Classifier')

    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', **PlotConfig.FONTS['label'])
    ax.set_ylabel('True Positive Rate', **PlotConfig.FONTS['label'])
    ax.set_title(title, **PlotConfig.FONTS['title'])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def create_risk_gauge(risk_score, title="Credit Risk Score", save_name=None):
    """
    Create a risk gauge visualization using Plotly

    Args:
        risk_score: Risk score between 0 and 1
        title: Plot title
        save_name: Filename to save plot

    Returns:
        plotly figure object
    """
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )

    if save_name:
        fig.write_html(f'/home/user/output/visualizations/{save_name}.html')
        fig.write_image(f'/home/user/output/visualizations/{save_name}.png')

    return fig

def plot_feature_importance(feature_names, importance_scores, 
                          title="Feature Importance", top_n=20, save_name=None):
    """
    Create feature importance visualization

    Args:
        feature_names: List of feature names
        importance_scores: Corresponding importance scores
        title: Plot title
        top_n: Number of top features to display
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    # Create DataFrame and sort by importance
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True).tail(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=PlotConfig.FIGSIZE['medium'])

    # Create horizontal bar plot
    bars = ax.barh(df['feature'], df['importance'], 
                   color=PlotConfig.COLORS['primary'], alpha=0.7)

    # Customize plot
    ax.set_xlabel('Importance Score', **PlotConfig.FONTS['label'])
    ax.set_title(title, **PlotConfig.FONTS['title'])
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def create_performance_metrics_plot(metrics_dict, title="Model Performance Metrics", 
                                  save_name=None):
    """
    Create performance metrics visualization

    Args:
        metrics_dict: Dictionary with metric names and values
        title: Plot title
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    # Prepare data
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PlotConfig.FIGSIZE['wide'])

    # Bar chart
    bars = ax1.bar(metrics, values, color=PlotConfig.COLORS['primary'], alpha=0.7)
    ax1.set_title(f"{title} - Bar Chart", **PlotConfig.FONTS['subtitle'])
    ax1.set_ylabel('Score', **PlotConfig.FONTS['label'])
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    # Rotate x-axis labels if needed
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_radar = values + [values[0]]  # Complete the circle
    angles += angles[:1]

    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, values_radar, 'o-', linewidth=2, 
             color=PlotConfig.COLORS['primary'])
    ax2.fill(angles, values_radar, alpha=0.25, 
             color=PlotConfig.COLORS['primary'])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"{title} - Radar Chart", **PlotConfig.FONTS['subtitle'])

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

# Initialize plotting style
setup_plot_style()

print("✅ Part 1 of plots.py created with basic plotting utilities")


# ============================================================================
# PART 2: Additional visualization functions
# ============================================================================

def plot_distribution_comparison(data, column, target_col, title=None, save_name=None):
    """
    Create distribution comparison plots for a feature across different target classes

    Args:
        data: DataFrame containing the data
        column: Column name to plot distribution for
        target_col: Target column name
        title: Plot title
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    if title is None:
        title = f"Distribution of {column} by {target_col}"

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=PlotConfig.FIGSIZE['large'])
    fig.suptitle(title, **PlotConfig.FONTS['title'])

    # Get unique classes
    classes = data[target_col].unique()
    colors = [PlotConfig.COLORS['primary'], PlotConfig.COLORS['secondary']]

    # Histogram comparison
    for i, cls in enumerate(classes):
        subset = data[data[target_col] == cls][column]
        axes[0, 0].hist(subset, alpha=0.7, label=f'Class {cls}', 
                       color=colors[i % len(colors)], bins=30)
    axes[0, 0].set_title('Histogram Comparison')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot comparison
    data.boxplot(column=column, by=target_col, ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot Comparison')
    axes[0, 1].set_xlabel(target_col)
    axes[0, 1].set_ylabel(column)

    # Density plot comparison
    for i, cls in enumerate(classes):
        subset = data[data[target_col] == cls][column]
        subset.plot.density(ax=axes[1, 0], alpha=0.7, label=f'Class {cls}',
                           color=colors[i % len(colors)])
    axes[1, 0].set_title('Density Plot Comparison')
    axes[1, 0].set_xlabel(column)
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Violin plot
    sns.violinplot(data=data, x=target_col, y=column, ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot Comparison')

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def create_correlation_heatmap(data, title="Correlation Matrix", 
                             figsize='large', save_name=None):
    """
    Create correlation heatmap with enhanced styling

    Args:
        data: DataFrame to calculate correlations
        title: Plot title
        figsize: Figure size key from PlotConfig
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=PlotConfig.FIGSIZE[figsize])

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                center=0, square=True, linewidths=0.5, 
                cbar_kws={"shrink": .8}, ax=ax)

    ax.set_title(title, **PlotConfig.FONTS['title'])
    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def plot_categorical_analysis(data, cat_col, target_col, title=None, save_name=None):
    """
    Create comprehensive categorical variable analysis

    Args:
        data: DataFrame containing the data
        cat_col: Categorical column to analyze
        target_col: Target column
        title: Plot title
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    if title is None:
        title = f"Analysis of {cat_col} vs {target_col}"

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=PlotConfig.FIGSIZE['large'])
    fig.suptitle(title, **PlotConfig.FONTS['title'])

    # Count plot
    sns.countplot(data=data, x=cat_col, hue=target_col, ax=axes[0, 0])
    axes[0, 0].set_title('Count Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Percentage stacked bar
    ct = pd.crosstab(data[cat_col], data[target_col], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=axes[0, 1], 
            color=[PlotConfig.COLORS['primary'], PlotConfig.COLORS['secondary']])
    axes[0, 1].set_title('Percentage Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylabel('Percentage')

    # Default rate by category
    default_rate = data.groupby(cat_col)[target_col].mean()
    default_rate.plot(kind='bar', ax=axes[1, 0], color=PlotConfig.COLORS['warning'])
    axes[1, 0].set_title('Default Rate by Category')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylabel('Default Rate')

    # Chi-square test visualization
    from scipy.stats import chi2_contingency
    ct_raw = pd.crosstab(data[cat_col], data[target_col])
    chi2, p_value, dof, expected = chi2_contingency(ct_raw)

    # Residuals heatmap
    residuals = (ct_raw - expected) / np.sqrt(expected)
    sns.heatmap(residuals, annot=True, cmap='RdBu_r', center=0, ax=axes[1, 1])
    axes[1, 1].set_title(f'Standardized Residuals\n(Chi2={chi2:.2f}, p={p_value:.4f})')

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def create_model_comparison_plot(models_performance, title="Model Performance Comparison", 
                               save_name=None):
    """
    Create comprehensive model comparison visualization

    Args:
        models_performance: Dict with model names as keys and metrics dict as values
        title: Plot title
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    # Prepare data
    df_metrics = pd.DataFrame(models_performance).T
    metrics = df_metrics.columns.tolist()
    models = df_metrics.index.tolist()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=PlotConfig.FIGSIZE['large'])
    fig.suptitle(title, **PlotConfig.FONTS['title'])

    # Bar plot for each metric
    df_metrics.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('All Metrics Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Heatmap of all metrics
    sns.heatmap(df_metrics, annot=True, cmap='YlOrRd', ax=axes[0, 1])
    axes[0, 1].set_title('Metrics Heatmap')

    # Radar chart for top 2 models
    if len(models) >= 2:
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        ax_radar = plt.subplot(223, projection='polar')

        for i, model in enumerate(models[:2]):
            values = df_metrics.loc[model].tolist()
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', linewidth=2, 
                         label=model, alpha=0.8)
            ax_radar.fill(angles, values, alpha=0.25)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Top 2 Models Comparison')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Best model highlight
    best_model_idx = df_metrics.mean(axis=1).idxmax()
    best_metrics = df_metrics.loc[best_model_idx]

    axes[1, 1].bar(metrics, best_metrics, color=PlotConfig.COLORS['success'], alpha=0.7)
    axes[1, 1].set_title(f'Best Model: {best_model_idx}')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim(0, 1)

    # Add value labels
    for i, v in enumerate(best_metrics):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def create_prediction_distribution_plot(y_true, y_pred_proba, threshold=0.5, 
                                      title="Prediction Distribution Analysis", 
                                      save_name=None):
    """
    Create prediction distribution analysis plots

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        title: Plot title
        save_name: Filename to save plot

    Returns:
        matplotlib figure object
    """
    setup_plot_style()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=PlotConfig.FIGSIZE['large'])
    fig.suptitle(title, **PlotConfig.FONTS['title'])

    # Prediction probability distribution by true class
    for cls in [0, 1]:
        mask = y_true == cls
        axes[0, 0].hist(y_pred_proba[mask], alpha=0.7, bins=30, 
                       label=f'True Class {cls}', density=True)
    axes[0, 0].axvline(threshold, color='red', linestyle='--', 
                      label=f'Threshold ({threshold})')
    axes[0, 0].set_title('Prediction Probability Distribution')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Calibration plot
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10)

    axes[0, 1].plot(mean_predicted_value, fraction_of_positives, "s-",
                   label="Model", color=PlotConfig.COLORS['primary'])
    axes[0, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes[0, 1].set_title('Calibration Plot')
    axes[0, 1].set_xlabel('Mean Predicted Probability')
    axes[0, 1].set_ylabel('Fraction of Positives')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision-Recall curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    axes[1, 0].plot(recall, precision, color=PlotConfig.COLORS['secondary'],
                   label=f'AP = {avg_precision:.3f}')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Threshold analysis
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1_scores = [], [], []

    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        if len(np.unique(y_pred_t)) > 1:
            from sklearn.metrics import precision_score, recall_score, f1_score
            precisions.append(precision_score(y_true, y_pred_t))
            recalls.append(recall_score(y_true, y_pred_t))
            f1_scores.append(f1_score(y_true, y_pred_t))
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)

    axes[1, 1].plot(thresholds, precisions, label='Precision', 
                   color=PlotConfig.COLORS['primary'])
    axes[1, 1].plot(thresholds, recalls, label='Recall', 
                   color=PlotConfig.COLORS['secondary'])
    axes[1, 1].plot(thresholds, f1_scores, label='F1-Score', 
                   color=PlotConfig.COLORS['success'])
    axes[1, 1].axvline(threshold, color='red', linestyle='--', 
                      label=f'Current Threshold')
    axes[1, 1].set_title('Threshold Analysis')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)

    return fig

def create_interactive_dashboard_data(data, target_col, save_name=None):
    """
    Create data structure for interactive dashboard using Plotly

    Args:
        data: DataFrame containing the data
        target_col: Target column name
        save_name: Filename to save the dashboard

    Returns:
        plotly figure object
    """
    from plotly.subplots import make_subplots

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Target Distribution', 'Feature Correlations', 
                       'Missing Values', 'Data Summary'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]]
    )

    # Target distribution pie chart
    target_counts = data[target_col].value_counts()
    fig.add_trace(
        go.Pie(labels=target_counts.index, values=target_counts.values,
               name="Target Distribution"),
        row=1, col=1
    )

    # Top correlations with target
    correlations = data.corr()[target_col].abs().sort_values(ascending=False)[1:11]
    fig.add_trace(
        go.Bar(x=correlations.values, y=correlations.index, 
               orientation='h', name="Top Correlations"),
        row=1, col=2
    )

    # Missing values
    missing_data = data.isnull().sum().sort_values(ascending=False)[:10]
    if missing_data.sum() > 0:
        fig.add_trace(
            go.Bar(x=missing_data.index, y=missing_data.values,
                   name="Missing Values"),
            row=2, col=1
        )

    # Data summary table
    summary_stats = data.describe().round(3)
    fig.add_trace(
        go.Table(
            header=dict(values=['Statistic'] + list(summary_stats.columns)),
            cells=dict(values=[summary_stats.index] + 
                      [summary_stats[col] for col in summary_stats.columns])
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Credit Default Prediction - Data Overview Dashboard",
        showlegend=False
    )

    if save_name:
        fig.write_html(f'/home/user/output/visualizations/{save_name}.html')

    return fig

