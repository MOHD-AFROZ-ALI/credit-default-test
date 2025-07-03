"""
Metrics Display Components for Credit Default Prediction System
Provides KPI widgets, performance metrics, and comparison displays
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import time

def display_basic_metric(
    title: str,
    value: Union[int, float, str],
    delta: Optional[Union[int, float, str]] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Display a basic metric with optional delta and formatting

    Args:
        title: Metric title
        value: Metric value
        delta: Change from previous period
        delta_color: Color for delta ("normal", "inverse", "off")
        help_text: Optional help tooltip
        format_string: Optional format string for value
    """
    if format_string and isinstance(value, (int, float)):
        formatted_value = format_string.format(value)
    else:
        formatted_value = value

    st.metric(
        label=title,
        value=formatted_value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )

def display_kpi_grid(kpis: Dict[str, Dict[str, Any]], columns: int = 4) -> None:
    """
    Display KPIs in a responsive grid layout

    Args:
        kpis: Dictionary of KPI configurations
        columns: Number of columns in grid
    """
    kpi_items = list(kpis.items())

    # Create rows based on number of columns
    for i in range(0, len(kpi_items), columns):
        cols = st.columns(columns)

        for j, col in enumerate(cols):
            if i + j < len(kpi_items):
                kpi_name, kpi_config = kpi_items[i + j]

                with col:
                    display_basic_metric(
                        title=kpi_config.get('title', kpi_name),
                        value=kpi_config.get('value', 0),
                        delta=kpi_config.get('delta'),
                        delta_color=kpi_config.get('delta_color', 'normal'),
                        help_text=kpi_config.get('help'),
                        format_string=kpi_config.get('format')
                    )

def create_gauge_chart(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    threshold_ranges: Optional[List[Tuple[float, float, str]]] = None,
    height: int = 300
) -> go.Figure:
    """
    Create a gauge chart for KPI visualization

    Args:
        value: Current value
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        threshold_ranges: List of (min, max, color) tuples for ranges
        height: Chart height

    Returns:
        Plotly figure object
    """
    if threshold_ranges is None:
        threshold_ranges = [
            (0, 33, "red"),
            (33, 66, "yellow"), 
            (66, 100, "green")
        ]

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': (min_val + max_val) / 2},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [r[0], r[1]], 'color': r[2]} 
                for r in threshold_ranges
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig

def display_progress_metric(
    title: str,
    current: float,
    target: float,
    unit: str = "",
    color: str = "blue",
    show_percentage: bool = True
) -> None:
    """
    Display a metric with progress bar

    Args:
        title: Metric title
        current: Current value
        target: Target value
        unit: Unit of measurement
        color: Progress bar color
        show_percentage: Whether to show percentage
    """
    progress = min(current / target, 1.0) if target > 0 else 0
    percentage = progress * 100

    st.markdown(f"**{title}**")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.progress(progress)

    with col2:
        if show_percentage:
            st.write(f"{percentage:.1f}%")
        else:
            st.write(f"{current:.1f}/{target:.1f} {unit}")

    st.caption(f"Current: {current:.1f} {unit} | Target: {target:.1f} {unit}")

def create_comparison_metrics(
    metrics_data: Dict[str, Dict[str, float]],
    comparison_type: str = "period"
) -> None:
    """
    Display comparison metrics between different periods or models

    Args:
        metrics_data: Dictionary with metric categories and values
        comparison_type: Type of comparison ("period", "model", "segment")
    """
    st.markdown(f"### 📊 {comparison_type.title()} Comparison")

    if len(metrics_data) < 2:
        st.warning("Need at least 2 data points for comparison")
        return

    # Create comparison table
    df_comparison = pd.DataFrame(metrics_data).T

    # Calculate changes
    if len(df_comparison.columns) >= 2:
        base_col = df_comparison.columns[0]
        current_col = df_comparison.columns[-1]

        df_comparison['Change'] = df_comparison[current_col] - df_comparison[base_col]
        df_comparison['Change %'] = (
            (df_comparison[current_col] - df_comparison[base_col]) / 
            df_comparison[base_col] * 100
        ).round(2)

    # Display metrics with color coding
    for idx, (metric_name, row) in enumerate(df_comparison.iterrows()):
        cols = st.columns(len(df_comparison.columns))

        for col_idx, (col_name, value) in enumerate(row.items()):
            with cols[col_idx]:
                if col_name == 'Change %':
                    delta_color = "normal" if value >= 0 else "inverse"
                    st.metric(
                        label=f"{metric_name} - {col_name}",
                        value=f"{value:.2f}%",
                        delta_color=delta_color
                    )
                elif col_name == 'Change':
                    delta_color = "normal" if value >= 0 else "inverse"
                    st.metric(
                        label=f"{metric_name} - {col_name}",
                        value=f"{value:.4f}",
                        delta_color=delta_color
                    )
                else:
                    st.metric(
                        label=f"{metric_name} - {col_name}",
                        value=f"{value:.4f}"
                    )

def display_performance_dashboard(
    model_metrics: Dict[str, float],
    business_metrics: Dict[str, float],
    system_metrics: Dict[str, float]
) -> None:
    """
    Display comprehensive performance dashboard

    Args:
        model_metrics: ML model performance metrics
        business_metrics: Business KPIs
        system_metrics: System performance metrics
    """
    st.markdown("## 🎯 Performance Dashboard")

    # Model Performance Section
    with st.expander("🤖 Model Performance", expanded=True):
        model_cols = st.columns(4)

        metrics_config = [
            ("Accuracy", model_metrics.get('accuracy', 0), "{:.2%}", "Model prediction accuracy"),
            ("Precision", model_metrics.get('precision', 0), "{:.2%}", "Positive prediction precision"),
            ("Recall", model_metrics.get('recall', 0), "{:.2%}", "True positive detection rate"),
            ("F1-Score", model_metrics.get('f1_score', 0), "{:.3f}", "Harmonic mean of precision and recall")
        ]

        for i, (title, value, fmt, help_text) in enumerate(metrics_config):
            with model_cols[i]:
                display_basic_metric(
                    title=title,
                    value=value,
                    format_string=fmt,
                    help_text=help_text
                )

    # Business Metrics Section
    with st.expander("💼 Business Metrics", expanded=True):
        business_cols = st.columns(3)

        with business_cols[0]:
            display_basic_metric(
                title="Default Rate",
                value=business_metrics.get('default_rate', 0),
                format_string="{:.2%}",
                delta=business_metrics.get('default_rate_delta'),
                help_text="Percentage of loans that defaulted"
            )

        with business_cols[1]:
            display_basic_metric(
                title="Revenue Impact",
                value=business_metrics.get('revenue_impact', 0),
                format_string="${:,.0f}",
                delta=business_metrics.get('revenue_delta'),
                help_text="Revenue impact from predictions"
            )

        with business_cols[2]:
            display_basic_metric(
                title="Risk Reduction",
                value=business_metrics.get('risk_reduction', 0),
                format_string="{:.1%}",
                delta=business_metrics.get('risk_reduction_delta'),
                help_text="Risk reduction achieved"
            )

    # System Performance Section
    with st.expander("⚙️ System Performance", expanded=True):
        system_cols = st.columns(4)

        with system_cols[0]:
            display_basic_metric(
                title="Response Time",
                value=system_metrics.get('response_time', 0),
                format_string="{:.0f}ms",
                help_text="Average API response time"
            )

        with system_cols[1]:
            display_basic_metric(
                title="Throughput",
                value=system_metrics.get('throughput', 0),
                format_string="{:.0f}/min",
                help_text="Predictions per minute"
            )

        with system_cols[2]:
            display_basic_metric(
                title="Uptime",
                value=system_metrics.get('uptime', 0),
                format_string="{:.2%}",
                help_text="System uptime percentage"
            )

        with system_cols[3]:
            display_basic_metric(
                title="Error Rate",
                value=system_metrics.get('error_rate', 0),
                format_string="{:.2%}",
                delta_color="inverse",
                help_text="System error rate"
            )

def create_trend_metrics(
    data: pd.DataFrame,
    metric_column: str,
    date_column: str,
    title: str,
    height: int = 300
) -> go.Figure:
    """
    Create trend visualization for metrics over time

    Args:
        data: DataFrame with time series data
        metric_column: Column name for metric values
        date_column: Column name for dates
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data[date_column],
        y=data[metric_column],
        mode='lines+markers',
        name=title,
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Add trend line
    if len(data) > 1:
        z = np.polyfit(range(len(data)), data[metric_column], 1)
        p = np.poly1d(z)

        fig.add_trace(go.Scatter(
            x=data[date_column],
            y=p(range(len(data))),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=1, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=metric_column.replace('_', ' ').title(),
        height=height,
        showlegend=True,
        hovermode='x unified'
    )

    return fig

def display_real_time_metrics(
    metrics_container: st.container,
    refresh_interval: int = 5
) -> None:
    """
    Display real-time updating metrics

    Args:
        metrics_container: Streamlit container for metrics
        refresh_interval: Refresh interval in seconds
    """
    # Placeholder for real-time data (in production, this would connect to live data)
    placeholder = metrics_container.empty()

    # Simulate real-time updates
    while True:
        with placeholder.container():
            # Generate mock real-time data
            current_time = datetime.now()

            # Mock metrics that change over time
            accuracy = 0.85 + np.random.normal(0, 0.02)
            throughput = 150 + np.random.normal(0, 10)
            response_time = 45 + np.random.normal(0, 5)

            st.markdown(f"**Last Updated:** {current_time.strftime('%H:%M:%S')}")

            cols = st.columns(3)

            with cols[0]:
                st.metric("Live Accuracy", f"{accuracy:.2%}")

            with cols[1]:
                st.metric("Live Throughput", f"{throughput:.0f}/min")

            with cols[2]:
                st.metric("Live Response Time", f"{response_time:.0f}ms")

        time.sleep(refresh_interval)
