"""
Chart Components for Credit Default Prediction System
Provides reusable chart components for data visualization and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set default color palette
DEFAULT_COLORS = px.colors.qualitative.Set3
RISK_COLORS = {'Low': '#2E8B57', 'Medium': '#FFD700', 'High': '#DC143C'}

def create_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    orientation: str = 'vertical',
    height: int = 400,
    show_values: bool = True
) -> go.Figure:
    """
    Create interactive bar chart

    Args:
        data: DataFrame with chart data
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Chart title
        color_col: Optional column for color coding
        orientation: 'vertical' or 'horizontal'
        height: Chart height
        show_values: Whether to show values on bars

    Returns:
        Plotly figure object
    """
    if orientation == 'horizontal':
        fig = px.bar(
            data, 
            x=y_col, 
            y=x_col, 
            color=color_col,
            title=title,
            orientation='h',
            color_discrete_sequence=DEFAULT_COLORS
        )
    else:
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            color_discrete_sequence=DEFAULT_COLORS
        )

    if show_values:
        fig.update_traces(texttemplate='%{y}', textposition='outside')

    fig.update_layout(
        height=height,
        showlegend=bool(color_col),
        hovermode='x unified'
    )

    return fig

def create_line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_cols: Union[str, List[str]],
    title: str,
    height: int = 400,
    show_markers: bool = True,
    line_styles: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Create interactive line chart

    Args:
        data: DataFrame with chart data
        x_col: Column for x-axis
        y_cols: Column(s) for y-axis
        title: Chart title
        height: Chart height
        show_markers: Whether to show markers
        line_styles: Optional line style configurations

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    for i, y_col in enumerate(y_cols):
        mode = 'lines+markers' if show_markers else 'lines'
        line_style = line_styles.get(y_col, {}) if line_styles else {}

        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode=mode,
            name=y_col.replace('_', ' ').title(),
            line=dict(
                color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                **line_style
            ),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=title,
        height=height,
        hovermode='x unified',
        showlegend=len(y_cols) > 1
    )

    return fig

def create_scatter_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    height: int = 400,
    add_trendline: bool = False
) -> go.Figure:
    """
    Create interactive scatter plot

    Args:
        data: DataFrame with chart data
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Chart title
        color_col: Optional column for color coding
        size_col: Optional column for bubble size
        height: Chart height
        add_trendline: Whether to add trend line

    Returns:
        Plotly figure object
    """
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        color_discrete_sequence=DEFAULT_COLORS,
        trendline='ols' if add_trendline else None
    )

    fig.update_layout(
        height=height,
        showlegend=bool(color_col)
    )

    return fig

def create_pie_chart(
    data: pd.DataFrame,
    values_col: str,
    names_col: str,
    title: str,
    height: int = 400,
    show_percentages: bool = True,
    hole_size: float = 0.0
) -> go.Figure:
    """
    Create interactive pie chart

    Args:
        data: DataFrame with chart data
        values_col: Column with values
        names_col: Column with category names
        title: Chart title
        height: Chart height
        show_percentages: Whether to show percentages
        hole_size: Size of center hole (0 for pie, >0 for donut)

    Returns:
        Plotly figure object
    """
    fig = px.pie(
        data,
        values=values_col,
        names=names_col,
        title=title,
        color_discrete_sequence=DEFAULT_COLORS,
        hole=hole_size
    )

    if show_percentages:
        fig.update_traces(textposition='inside', textinfo='percent+label')

    fig.update_layout(height=height)

    return fig

def create_histogram(
    data: pd.DataFrame,
    column: str,
    title: str,
    bins: int = 30,
    color_col: Optional[str] = None,
    height: int = 400,
    show_distribution_curve: bool = False
) -> go.Figure:
    """
    Create interactive histogram

    Args:
        data: DataFrame with chart data
        column: Column to plot
        title: Chart title
        bins: Number of bins
        color_col: Optional column for color coding
        height: Chart height
        show_distribution_curve: Whether to overlay distribution curve

    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        data,
        x=column,
        color=color_col,
        title=title,
        nbins=bins,
        color_discrete_sequence=DEFAULT_COLORS
    )

    if show_distribution_curve and not color_col:
        # Add normal distribution curve
        mean_val = data[column].mean()
        std_val = data[column].std()
        x_range = np.linspace(data[column].min(), data[column].max(), 100)
        y_range = ((1/(std_val * np.sqrt(2 * np.pi))) * 
                  np.exp(-0.5 * ((x_range - mean_val) / std_val) ** 2))

        # Scale to match histogram
        y_range = y_range * len(data) * (data[column].max() - data[column].min()) / bins

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))

    fig.update_layout(
        height=height,
        showlegend=bool(color_col or show_distribution_curve)
    )

    return fig

def create_box_plot(
    data: pd.DataFrame,
    y_col: str,
    title: str,
    x_col: Optional[str] = None,
    height: int = 400,
    show_points: bool = False
) -> go.Figure:
    """
    Create interactive box plot

    Args:
        data: DataFrame with chart data
        y_col: Column for y-axis (values)
        x_col: Optional column for x-axis (categories)
        title: Chart title
        height: Chart height
        show_points: Whether to show individual points

    Returns:
        Plotly figure object
    """
    if x_col:
        fig = px.box(
            data,
            x=x_col,
            y=y_col,
            title=title,
            color=x_col,
            color_discrete_sequence=DEFAULT_COLORS,
            points='all' if show_points else False
        )
    else:
        fig = px.box(
            data,
            y=y_col,
            title=title,
            points='all' if show_points else False
        )

    fig.update_layout(height=height)

    return fig

def create_correlation_heatmap(
    data: pd.DataFrame,
    title: str,
    height: int = 500,
    color_scale: str = 'RdBu_r',
    show_values: bool = True
) -> go.Figure:
    """
    Create correlation heatmap

    Args:
        data: DataFrame with numeric data
        title: Chart title
        height: Chart height
        color_scale: Color scale for heatmap
        show_values: Whether to show correlation values

    Returns:
        Plotly figure object
    """
    # Calculate correlation matrix
    corr_matrix = data.select_dtypes(include=[np.number]).corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=color_scale,
        zmid=0,
        text=corr_matrix.round(2).values if show_values else None,
        texttemplate='%{text}' if show_values else None,
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Features",
        yaxis_title="Features"
    )

    return fig

def create_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    height: int = 400
) -> go.Figure:
    """
    Create confusion matrix heatmap

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)",
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations
    )

    return fig

def create_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    height: int = 400
) -> go.Figure:
    """
    Create ROC curve

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))

    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=1, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=height,
        showlegend=True
    )

    return fig

def create_feature_importance_chart(
    feature_names: List[str],
    importance_values: List[float],
    title: str = "Feature Importance",
    height: int = 400,
    max_features: int = 20
) -> go.Figure:
    """
    Create feature importance chart

    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Chart title
        height: Chart height
        max_features: Maximum number of features to display

    Returns:
        Plotly figure object
    """
    # Create DataFrame and sort by importance
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=True).tail(max_features)

    fig = px.bar(
        df,
        x='importance',
        y='feature',
        orientation='h',
        title=title,
        color='importance',
        color_continuous_scale='viridis'
    )

    fig.update_layout(
        height=height,
        showlegend=False,
        yaxis_title="Features",
        xaxis_title="Importance"
    )

    return fig


def create_revenue_trend_chart(
    data: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    title: str = "Revenue Trend Analysis",
    height: int = 500,
    show_forecast: bool = False,
    forecast_periods: int = 12
) -> go.Figure:
    """
    Create revenue trend chart with optional forecasting

    Args:
        data: DataFrame with revenue data
        date_col: Date column name
        revenue_col: Revenue column name
        title: Chart title
        height: Chart height
        show_forecast: Whether to show forecast
        forecast_periods: Number of periods to forecast

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Revenue Trend", "Monthly Growth Rate"),
        vertical_spacing=0.1
    )

    # Main revenue trend
    fig.add_trace(
        go.Scatter(
            x=data[date_col],
            y=data[revenue_col],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Calculate and add moving average
    if len(data) >= 3:
        data['ma_3'] = data[revenue_col].rolling(window=3).mean()
        fig.add_trace(
            go.Scatter(
                x=data[date_col],
                y=data['ma_3'],
                mode='lines',
                name='3-Month MA',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )

    # Growth rate calculation
    data['growth_rate'] = data[revenue_col].pct_change() * 100
    fig.add_trace(
        go.Bar(
            x=data[date_col],
            y=data['growth_rate'],
            name='Growth Rate (%)',
            marker_color=['green' if x >= 0 else 'red' for x in data['growth_rate']]
        ),
        row=2, col=1
    )

    # Add forecast if requested
    if show_forecast and len(data) >= 12:
        from sklearn.linear_model import LinearRegression

        # Simple linear forecast
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[revenue_col].values

        model = LinearRegression()
        model.fit(X, y)

        # Generate forecast
        future_X = np.arange(len(data), len(data) + forecast_periods).reshape(-1, 1)
        forecast = model.predict(future_X)

        # Create future dates
        last_date = pd.to_datetime(data[date_col].iloc[-1])
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='M'
        )

        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='orange', width=2, dash='dot'),
                marker=dict(size=4)
            ),
            row=1, col=1
        )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        hovermode='x unified'
    )

    return fig

def create_risk_distribution_chart(
    data: pd.DataFrame,
    risk_col: str,
    amount_col: str,
    title: str = "Risk Distribution Analysis",
    height: int = 400
) -> go.Figure:
    """
    Create risk distribution visualization

    Args:
        data: DataFrame with risk data
        risk_col: Risk category column
        amount_col: Amount/value column
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Risk Distribution by Count", "Risk Distribution by Amount"),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )

    # Count distribution
    risk_counts = data[risk_col].value_counts()
    fig.add_trace(
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            name="Count",
            marker_colors=[RISK_COLORS.get(risk, '#808080') for risk in risk_counts.index]
        ),
        row=1, col=1
    )

    # Amount distribution
    risk_amounts = data.groupby(risk_col)[amount_col].sum()
    fig.add_trace(
        go.Pie(
            labels=risk_amounts.index,
            values=risk_amounts.values,
            name="Amount",
            marker_colors=[RISK_COLORS.get(risk, '#808080') for risk in risk_amounts.index]
        ),
        row=1, col=2
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title=title, height=height)

    return fig

def create_violin_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    height: int = 400,
    show_box: bool = True
) -> go.Figure:
    """
    Create violin plot for distribution analysis

    Args:
        data: DataFrame with data
        x_col: Categorical column
        y_col: Numeric column
        title: Chart title
        height: Chart height
        show_box: Whether to show box plot inside violin

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    categories = data[x_col].unique()
    colors = DEFAULT_COLORS[:len(categories)]

    for i, category in enumerate(categories):
        category_data = data[data[x_col] == category][y_col]

        fig.add_trace(go.Violin(
            y=category_data,
            name=str(category),
            box_visible=show_box,
            meanline_visible=True,
            fillcolor=colors[i],
            opacity=0.6,
            x0=str(category)
        ))

    fig.update_layout(
        title=title,
        height=height,
        yaxis_title=y_col.replace('_', ' ').title(),
        xaxis_title=x_col.replace('_', ' ').title()
    )

    return fig

def create_density_plot(
    data: pd.DataFrame,
    columns: List[str],
    title: str,
    height: int = 400,
    fill_alpha: float = 0.3
) -> go.Figure:
    """
    Create density plot for multiple distributions

    Args:
        data: DataFrame with data
        columns: List of columns to plot
        title: Chart title
        height: Chart height
        fill_alpha: Fill transparency

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    for i, col in enumerate(columns):
        # Calculate density using numpy
        values = data[col].dropna()
        hist, bin_edges = np.histogram(values, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=hist,
            mode='lines',
            fill='tonexty' if i > 0 else 'tozeroy',
            name=col.replace('_', ' ').title(),
            line=dict(color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])) + [fill_alpha])}"
        ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Value",
        yaxis_title="Density"
    )

    return fig

def create_multi_panel_dashboard(
    data: pd.DataFrame,
    config: Dict[str, Dict[str, Any]],
    title: str = "Multi-Panel Dashboard",
    height: int = 800
) -> go.Figure:
    """
    Create multi-panel dashboard with various chart types

    Args:
        data: DataFrame with data
        config: Configuration for each panel
        title: Dashboard title
        height: Total height

    Returns:
        Plotly figure object
    """
    n_panels = len(config)
    rows = (n_panels + 1) // 2  # 2 columns layout

    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=list(config.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    for i, (panel_name, panel_config) in enumerate(config.items()):
        row = (i // 2) + 1
        col = (i % 2) + 1

        chart_type = panel_config.get('type', 'bar')
        x_col = panel_config.get('x_col')
        y_col = panel_config.get('y_col')

        if chart_type == 'bar':
            fig.add_trace(
                go.Bar(
                    x=data[x_col],
                    y=data[y_col],
                    name=panel_name,
                    showlegend=False
                ),
                row=row, col=col
            )

        elif chart_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name=panel_name,
                    showlegend=False
                ),
                row=row, col=col
            )

        elif chart_type == 'scatter':
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    name=panel_name,
                    showlegend=False
                ),
                row=row, col=col
            )

    fig.update_layout(title=title, height=height)

    return fig

def create_demographic_chart(
    data: pd.DataFrame,
    demographic_col: str,
    metric_col: str,
    title: str,
    chart_type: str = 'bar',
    height: int = 400
) -> go.Figure:
    """
    Create demographic analysis chart

    Args:
        data: DataFrame with demographic data
        demographic_col: Demographic category column
        metric_col: Metric to analyze
        title: Chart title
        chart_type: Type of chart ('bar', 'pie', 'treemap')
        height: Chart height

    Returns:
        Plotly figure object
    """
    # Aggregate data by demographic
    agg_data = data.groupby(demographic_col)[metric_col].agg(['count', 'mean', 'sum']).reset_index()

    if chart_type == 'bar':
        fig = px.bar(
            agg_data,
            x=demographic_col,
            y='sum',
            title=title,
            color='mean',
            color_continuous_scale='viridis'
        )

    elif chart_type == 'pie':
        fig = px.pie(
            agg_data,
            values='sum',
            names=demographic_col,
            title=title,
            color_discrete_sequence=DEFAULT_COLORS
        )

    elif chart_type == 'treemap':
        fig = px.treemap(
            agg_data,
            path=[demographic_col],
            values='sum',
            color='mean',
            title=title,
            color_continuous_scale='viridis'
        )

    fig.update_layout(height=height)

    return fig

def create_time_series_decomposition(
    data: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "Time Series Decomposition",
    height: int = 600
) -> go.Figure:
    """
    Create time series decomposition chart

    Args:
        data: DataFrame with time series data
        date_col: Date column name
        value_col: Value column name
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    # Ensure data is sorted by date
    data_sorted = data.sort_values(date_col).copy()

    # Calculate moving averages for trend
    data_sorted['trend'] = data_sorted[value_col].rolling(window=12, center=True).mean()

    # Calculate seasonal component (simplified)
    data_sorted['seasonal'] = data_sorted[value_col] - data_sorted['trend']

    # Calculate residual
    data_sorted['residual'] = data_sorted[value_col] - data_sorted['trend'] - data_sorted['seasonal']

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.05
    )

    # Original series
    fig.add_trace(
        go.Scatter(
            x=data_sorted[date_col],
            y=data_sorted[value_col],
            mode='lines',
            name='Original',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Trend
    fig.add_trace(
        go.Scatter(
            x=data_sorted[date_col],
            y=data_sorted['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='red')
        ),
        row=2, col=1
    )

    # Seasonal
    fig.add_trace(
        go.Scatter(
            x=data_sorted[date_col],
            y=data_sorted['seasonal'],
            mode='lines',
            name='Seasonal',
            line=dict(color='green')
        ),
        row=3, col=1
    )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=data_sorted[date_col],
            y=data_sorted['residual'],
            mode='lines',
            name='Residual',
            line=dict(color='orange')
        ),
        row=4, col=1
    )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False
    )

    return fig

def create_cohort_analysis_chart(
    data: pd.DataFrame,
    user_col: str,
    date_col: str,
    value_col: str,
    title: str = "Cohort Analysis",
    height: int = 500
) -> go.Figure:
    """
    Create cohort analysis heatmap

    Args:
        data: DataFrame with user activity data
        user_col: User identifier column
        date_col: Date column
        value_col: Value/metric column
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    # Convert date column to datetime
    data[date_col] = pd.to_datetime(data[date_col])

    # Create cohort month (first activity month for each user)
    data['cohort_month'] = data.groupby(user_col)[date_col].transform('min').dt.to_period('M')
    data['activity_month'] = data[date_col].dt.to_period('M')

    # Calculate period number
    data['period_number'] = (data['activity_month'] - data['cohort_month']).apply(attrgetter('n'))

    # Create cohort table
    cohort_data = data.groupby(['cohort_month', 'period_number'])[user_col].nunique().reset_index()
    cohort_table = cohort_data.pivot(index='cohort_month', columns='period_number', values=user_col)

    # Calculate retention rates
    cohort_sizes = data.groupby('cohort_month')[user_col].nunique()
    retention_table = cohort_table.divide(cohort_sizes, axis=0)

    fig = go.Figure(data=go.Heatmap(
        z=retention_table.values,
        x=[f"Period {i}" for i in retention_table.columns],
        y=[str(idx) for idx in retention_table.index],
        colorscale='Blues',
        text=np.round(retention_table.values * 100, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Period",
        yaxis_title="Cohort Month"
    )

    return fig
