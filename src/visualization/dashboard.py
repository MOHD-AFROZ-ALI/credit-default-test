"""
Dashboard Components and Layout Management System
Provides comprehensive dashboard functionality with KPI calculations,
layout management, and interactive components for Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardLayout(Enum):
    """Dashboard layout types"""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    GRID = "grid"
    SIDEBAR = "sidebar"
    TABS = "tabs"

class KPIType(Enum):
    """KPI display types"""
    METRIC = "metric"
    GAUGE = "gauge"
    PROGRESS = "progress"
    TREND = "trend"
    COMPARISON = "comparison"

@dataclass
class KPIConfig:
    """Configuration for KPI display"""
    name: str
    value: Union[int, float, str]
    delta: Optional[Union[int, float]] = None
    delta_color: str = "normal"
    format_string: str = "{:.2f}"
    prefix: str = ""
    suffix: str = ""
    help_text: Optional[str] = None
    kpi_type: KPIType = KPIType.METRIC

@dataclass
class DashboardSection:
    """Dashboard section configuration"""
    title: str
    content_type: str
    data: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    layout: Optional[DashboardLayout] = None
    order: int = 0
    visible: bool = True

class DashboardState:
    """Dashboard state management"""

    def __init__(self):
        self.sections: Dict[str, DashboardSection] = {}
        self.kpis: Dict[str, KPIConfig] = {}
        self.filters: Dict[str, Any] = {}
        self.layout: DashboardLayout = DashboardLayout.SINGLE_COLUMN
        self.theme: str = "default"
        self.refresh_interval: int = 300  # seconds
        self.last_refresh: Optional[datetime] = None

    def add_section(self, section_id: str, section: DashboardSection):
        """Add a section to the dashboard"""
        self.sections[section_id] = section
        logger.info(f"Added section: {section_id}")

    def remove_section(self, section_id: str):
        """Remove a section from the dashboard"""
        if section_id in self.sections:
            del self.sections[section_id]
            logger.info(f"Removed section: {section_id}")

    def add_kpi(self, kpi_id: str, kpi: KPIConfig):
        """Add a KPI to the dashboard"""
        self.kpis[kpi_id] = kpi
        logger.info(f"Added KPI: {kpi_id}")

    def update_filter(self, filter_name: str, value: Any):
        """Update dashboard filter"""
        self.filters[filter_name] = value
        logger.info(f"Updated filter {filter_name}: {value}")

    def get_filtered_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply current filters to data"""
        filtered_data = data.copy()

        for filter_name, filter_value in self.filters.items():
            if filter_name in filtered_data.columns and filter_value is not None:
                if isinstance(filter_value, (list, tuple)):
                    filtered_data = filtered_data[filtered_data[filter_name].isin(filter_value)]
                else:
                    filtered_data = filtered_data[filtered_data[filter_name] == filter_value]

        return filtered_data

class KPICalculator:
    """KPI calculation utilities"""

    @staticmethod
    def calculate_growth_rate(current: float, previous: float) -> float:
        """Calculate growth rate percentage"""
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100

    @staticmethod
    def calculate_moving_average(data: pd.Series, window: int = 7) -> pd.Series:
        """Calculate moving average"""
        return data.rolling(window=window).mean()

    @staticmethod
    def calculate_trend(data: pd.Series) -> str:
        """Determine trend direction"""
        if len(data) < 2:
            return "stable"

        recent_avg = data.tail(3).mean()
        previous_avg = data.head(3).mean()

        if recent_avg > previous_avg * 1.05:
            return "increasing"
        elif recent_avg < previous_avg * 0.95:
            return "decreasing"
        else:
            return "stable"

    @staticmethod
    def calculate_percentile(data: pd.Series, percentile: float) -> float:
        """Calculate percentile value"""
        return np.percentile(data.dropna(), percentile)

    @staticmethod
    def calculate_variance_from_target(actual: float, target: float) -> Dict[str, Any]:
        """Calculate variance from target"""
        variance = actual - target
        variance_pct = (variance / target * 100) if target != 0 else 0

        return {
            'variance': variance,
            'variance_pct': variance_pct,
            'status': 'above' if variance > 0 else 'below' if variance < 0 else 'on_target'
        }

class DashboardLayoutManager:
    """Manages dashboard layouts and rendering"""

    def __init__(self, state: DashboardState):
        self.state = state

    def render_single_column(self, sections: List[DashboardSection]):
        """Render single column layout"""
        for section in sorted(sections, key=lambda x: x.order):
            if section.visible:
                self._render_section(section)

    def render_two_column(self, sections: List[DashboardSection]):
        """Render two column layout"""
        col1, col2 = st.columns(2)

        left_sections = [s for s in sections if s.order % 2 == 0 and s.visible]
        right_sections = [s for s in sections if s.order % 2 == 1 and s.visible]

        with col1:
            for section in sorted(left_sections, key=lambda x: x.order):
                self._render_section(section)

        with col2:
            for section in sorted(right_sections, key=lambda x: x.order):
                self._render_section(section)

    def render_three_column(self, sections: List[DashboardSection]):
        """Render three column layout"""
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]

        for i, section in enumerate(sorted([s for s in sections if s.visible], key=lambda x: x.order)):
            with columns[i % 3]:
                self._render_section(section)

    def render_grid(self, sections: List[DashboardSection], cols: int = 2):
        """Render grid layout"""
        visible_sections = [s for s in sections if s.visible]
        visible_sections.sort(key=lambda x: x.order)

        for i in range(0, len(visible_sections), cols):
            columns = st.columns(cols)
            for j in range(cols):
                if i + j < len(visible_sections):
                    with columns[j]:
                        self._render_section(visible_sections[i + j])

    def render_tabs(self, sections: List[DashboardSection]):
        """Render tabbed layout"""
        visible_sections = [s for s in sections if s.visible]
        tab_names = [s.title for s in sorted(visible_sections, key=lambda x: x.order)]

        if tab_names:
            tabs = st.tabs(tab_names)
            for tab, section in zip(tabs, sorted(visible_sections, key=lambda x: x.order)):
                with tab:
                    self._render_section(section)

    def render_sidebar(self, sections: List[DashboardSection]):
        """Render layout with sidebar"""
        # Sidebar sections
        sidebar_sections = [s for s in sections if s.config.get('sidebar', False) and s.visible]
        main_sections = [s for s in sections if not s.config.get('sidebar', False) and s.visible]

        # Render sidebar sections
        with st.sidebar:
            for section in sorted(sidebar_sections, key=lambda x: x.order):
                self._render_section(section)

        # Render main sections
        for section in sorted(main_sections, key=lambda x: x.order):
            self._render_section(section)

    def _render_section(self, section: DashboardSection):
        """Render individual section"""
        if section.title:
            st.subheader(section.title)

        try:
            if section.content_type == "kpi":
                self._render_kpi_section(section)
            elif section.content_type == "chart":
                self._render_chart_section(section)
            elif section.content_type == "table":
                self._render_table_section(section)
            elif section.content_type == "text":
                self._render_text_section(section)
            elif section.content_type == "custom":
                self._render_custom_section(section)
            else:
                st.warning(f"Unknown content type: {section.content_type}")

        except Exception as e:
            st.error(f"Error rendering section {section.title}: {str(e)}")
            logger.error(f"Section rendering error: {e}")

    def _render_kpi_section(self, section: DashboardSection):
        """Render KPI section"""
        kpi_data = section.data
        if isinstance(kpi_data, dict):
            cols = st.columns(len(kpi_data))
            for i, (kpi_id, kpi_config) in enumerate(kpi_data.items()):
                with cols[i]:
                    self._render_single_kpi(kpi_config)
        elif isinstance(kpi_data, KPIConfig):
            self._render_single_kpi(kpi_data)

    def _render_single_kpi(self, kpi: KPIConfig):
        """Render single KPI"""
        if kpi.kpi_type == KPIType.METRIC:
            formatted_value = f"{kpi.prefix}{kpi.format_string.format(kpi.value)}{kpi.suffix}"
            st.metric(
                label=kpi.name,
                value=formatted_value,
                delta=kpi.delta,
                delta_color=kpi.delta_color,
                help=kpi.help_text
            )
        elif kpi.kpi_type == KPIType.GAUGE:
            self._render_gauge_kpi(kpi)
        elif kpi.kpi_type == KPIType.PROGRESS:
            self._render_progress_kpi(kpi)

    def _render_gauge_kpi(self, kpi: KPIConfig):
        """Render gauge KPI"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=kpi.value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': kpi.name},
            delta={'reference': kpi.delta} if kpi.delta else None,
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    def _render_progress_kpi(self, kpi: KPIConfig):
        """Render progress bar KPI"""
        st.write(kpi.name)
        progress_value = min(max(kpi.value / 100, 0), 1)  # Normalize to 0-1
        st.progress(progress_value)
        st.write(f"{kpi.prefix}{kpi.format_string.format(kpi.value)}{kpi.suffix}")

    def _render_chart_section(self, section: DashboardSection):
        """Render chart section"""
        if section.data is None:
            st.warning("No data available for chart")
            return

        chart_type = section.config.get('chart_type', 'line')
        data = section.data

        if isinstance(data, pd.DataFrame):
            if chart_type == 'line':
                fig = px.line(data, **section.config.get('chart_params', {}))
            elif chart_type == 'bar':
                fig = px.bar(data, **section.config.get('chart_params', {}))
            elif chart_type == 'scatter':
                fig = px.scatter(data, **section.config.get('chart_params', {}))
            elif chart_type == 'pie':
                fig = px.pie(data, **section.config.get('chart_params', {}))
            elif chart_type == 'histogram':
                fig = px.histogram(data, **section.config.get('chart_params', {}))
            elif chart_type == 'box':
                fig = px.box(data, **section.config.get('chart_params', {}))
            else:
                st.warning(f"Unsupported chart type: {chart_type}")
                return

            fig.update_layout(height=section.config.get('height', 400))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Chart data must be a pandas DataFrame")

    def _render_table_section(self, section: DashboardSection):
        """Render table section"""
        if section.data is None:
            st.warning("No data available for table")
            return

        data = section.data
        if isinstance(data, pd.DataFrame):
            # Apply table configuration
            config = section.config

            # Pagination
            if config.get('paginate', False):
                page_size = config.get('page_size', 10)
                total_rows = len(data)
                total_pages = (total_rows - 1) // page_size + 1

                page = st.selectbox(
                    f"Page (1-{total_pages})",
                    range(1, total_pages + 1),
                    key=f"table_page_{section.title}"
                )

                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                data = data.iloc[start_idx:end_idx]

            # Column selection
            if config.get('column_selector', False):
                selected_columns = st.multiselect(
                    "Select columns to display",
                    data.columns.tolist(),
                    default=data.columns.tolist(),
                    key=f"table_cols_{section.title}"
                )
                if selected_columns:
                    data = data[selected_columns]

            # Display table
            st.dataframe(
                data,
                use_container_width=config.get('use_container_width', True),
                height=config.get('height', None)
            )

            # Export option
            if config.get('export', False):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{section.title.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Table data must be a pandas DataFrame")

    def _render_text_section(self, section: DashboardSection):
        """Render text section"""
        content = section.data or section.config.get('content', '')
        text_type = section.config.get('text_type', 'markdown')

        if text_type == 'markdown':
            st.markdown(content)
        elif text_type == 'html':
            st.html(content)
        elif text_type == 'code':
            language = section.config.get('language', 'python')
            st.code(content, language=language)
        elif text_type == 'latex':
            st.latex(content)
        else:
            st.text(content)

    def _render_custom_section(self, section: DashboardSection):
        """Render custom section"""
        custom_function = section.config.get('render_function')
        if custom_function and callable(custom_function):
            try:
                custom_function(section.data, section.config)
            except Exception as e:
                st.error(f"Error in custom render function: {str(e)}")
        else:
            st.warning("No valid render function provided for custom section")

class InteractiveComponents:
    """Interactive filter and control components"""

    @staticmethod
    def date_range_filter(key: str, default_start: datetime = None, default_end: datetime = None) -> Tuple[datetime, datetime]:
        """Date range filter component"""
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start or datetime.now() - timedelta(days=30),
                key=f"{key}_start"
            )

        with col2:
            end_date = st.date_input(
                "End Date", 
                value=default_end or datetime.now(),
                key=f"{key}_end"
            )

        return start_date, end_date

    @staticmethod
    def multi_select_filter(label: str, options: List[str], key: str, default: List[str] = None) -> List[str]:
        """Multi-select filter component"""
        return st.multiselect(
            label,
            options,
            default=default or [],
            key=key
        )

    @staticmethod
    def numeric_range_filter(label: str, min_val: float, max_val: float, key: str) -> Tuple[float, float]:
        """Numeric range filter component"""
        return st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            key=key
        )

    @staticmethod
    def category_filter(label: str, options: List[str], key: str, default: str = None) -> str:
        """Category filter component"""
        return st.selectbox(
            label,
            options,
            index=options.index(default) if default and default in options else 0,
            key=key
        )

    @staticmethod
    def search_filter(label: str, key: str, placeholder: str = "Search...") -> str:
        """Search filter component"""
        return st.text_input(
            label,
            placeholder=placeholder,
            key=key
        )

class DashboardRefreshManager:
    """Manages dashboard refresh and update mechanisms"""

    def __init__(self, state: DashboardState):
        self.state = state

    def auto_refresh_component(self):
        """Auto-refresh component"""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            auto_refresh = st.checkbox("Auto Refresh", key="auto_refresh")

        with col2:
            refresh_interval = st.selectbox(
                "Interval (seconds)",
                [30, 60, 300, 600, 1800],
                index=2,
                key="refresh_interval"
            )

        with col3:
            manual_refresh = st.button("Refresh Now", key="manual_refresh")

        if auto_refresh:
            self.state.refresh_interval = refresh_interval
            # Note: Actual auto-refresh would require additional Streamlit configuration

        if manual_refresh:
            self.refresh_dashboard()

    def refresh_dashboard(self):
        """Refresh dashboard data"""
        self.state.last_refresh = datetime.now()
        st.rerun()

    def show_last_refresh(self):
        """Display last refresh time"""
        if self.state.last_refresh:
            st.caption(f"Last refreshed: {self.state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("Not refreshed yet")

class CreditKPICalculator(KPICalculator):
    """Credit-specific KPI calculations"""

    @staticmethod
    def calculate_default_rate(total_loans: int, defaulted_loans: int) -> float:
        """Calculate default rate percentage"""
        if total_loans == 0:
            return 0.0
        return (defaulted_loans / total_loans) * 100

    @staticmethod
    def calculate_recovery_rate(total_defaulted_amount: float, recovered_amount: float) -> float:
        """Calculate recovery rate percentage"""
        if total_defaulted_amount == 0:
            return 0.0
        return (recovered_amount / total_defaulted_amount) * 100

    @staticmethod
    def calculate_portfolio_at_risk(total_portfolio: float, at_risk_amount: float) -> float:
        """Calculate portfolio at risk percentage"""
        if total_portfolio == 0:
            return 0.0
        return (at_risk_amount / total_portfolio) * 100

    @staticmethod
    def calculate_credit_utilization(credit_limit: float, outstanding_balance: float) -> float:
        """Calculate credit utilization percentage"""
        if credit_limit == 0:
            return 0.0
        return (outstanding_balance / credit_limit) * 100

    @staticmethod
    def calculate_vintage_analysis(data: pd.DataFrame, date_col: str, amount_col: str, 
                                 status_col: str, vintage_months: int = 12) -> pd.DataFrame:
        """Calculate vintage analysis for credit portfolio"""
        data[date_col] = pd.to_datetime(data[date_col])
        data['vintage'] = data[date_col].dt.to_period('M')

        vintage_stats = data.groupby('vintage').agg({
            amount_col: ['sum', 'count'],
            status_col: lambda x: (x == 'default').sum()
        }).round(2)

        vintage_stats.columns = ['total_amount', 'total_count', 'default_count']
        vintage_stats['default_rate'] = (vintage_stats['default_count'] / vintage_stats['total_count'] * 100).round(2)

        return vintage_stats.tail(vintage_months)

    @staticmethod
    def calculate_loss_given_default(exposure_at_default: float, recovery_amount: float) -> float:
        """Calculate Loss Given Default (LGD)"""
        if exposure_at_default == 0:
            return 0.0
        return ((exposure_at_default - recovery_amount) / exposure_at_default) * 100

class BusinessIntelligenceComponents:
    """Business intelligence dashboard components"""

    @staticmethod
    def render_executive_summary(data: Dict[str, Any]):
        """Render executive summary section"""
        st.header("Executive Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Portfolio",
                f"${data.get('total_portfolio', 0):,.0f}",
                delta=f"{data.get('portfolio_growth', 0):+.1f}%"
            )

        with col2:
            st.metric(
                "Default Rate",
                f"{data.get('default_rate', 0):.2f}%",
                delta=f"{data.get('default_rate_change', 0):+.2f}%",
                delta_color="inverse"
            )

        with col3:
            st.metric(
                "Recovery Rate",
                f"{data.get('recovery_rate', 0):.2f}%",
                delta=f"{data.get('recovery_rate_change', 0):+.2f}%"
            )

        with col4:
            st.metric(
                "Active Accounts",
                f"{data.get('active_accounts', 0):,}",
                delta=f"{data.get('account_growth', 0):+.0f}"
            )

    @staticmethod
    def render_risk_dashboard(risk_data: pd.DataFrame):
        """Render risk management dashboard"""
        st.header("Risk Management Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            # Risk distribution pie chart
            if 'risk_category' in risk_data.columns and 'amount' in risk_data.columns:
                risk_summary = risk_data.groupby('risk_category')['amount'].sum().reset_index()
                fig = px.pie(risk_summary, values='amount', names='risk_category', 
                           title="Portfolio Risk Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Risk trend over time
            if 'date' in risk_data.columns and 'risk_score' in risk_data.columns:
                risk_trend = risk_data.groupby('date')['risk_score'].mean().reset_index()
                fig = px.line(risk_trend, x='date', y='risk_score', 
                            title="Average Risk Score Trend")
                st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_performance_metrics(performance_data: Dict[str, Any]):
        """Render performance metrics section"""
        st.header("Performance Metrics")

        # Create performance gauge charts
        col1, col2, col3 = st.columns(3)

        metrics = [
            ("Collection Efficiency", performance_data.get('collection_efficiency', 0)),
            ("Customer Satisfaction", performance_data.get('customer_satisfaction', 0)),
            ("Process Automation", performance_data.get('automation_rate', 0))
        ]

        for i, (metric_name, value) in enumerate(metrics):
            with [col1, col2, col3][i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': metric_name},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

class DashboardBuilder:
    """Main dashboard builder class"""

    def __init__(self):
        self.state = DashboardState()
        self.layout_manager = DashboardLayoutManager(self.state)
        self.refresh_manager = DashboardRefreshManager(self.state)
        self.interactive = InteractiveComponents()
        self.bi_components = BusinessIntelligenceComponents()

    def build_dashboard(self, config: Dict[str, Any]):
        """Build complete dashboard from configuration"""
        # Set page configuration
        st.set_page_config(
            page_title=config.get('title', 'Dashboard'),
            page_icon=config.get('icon', '📊'),
            layout=config.get('layout', 'wide'),
            initial_sidebar_state=config.get('sidebar_state', 'expanded')
        )

        # Dashboard header
        st.title(config.get('title', 'Dashboard'))

        # Refresh controls
        if config.get('show_refresh_controls', True):
            self.refresh_manager.auto_refresh_component()
            self.refresh_manager.show_last_refresh()

        # Filters section
        if config.get('filters'):
            with st.expander("Filters", expanded=True):
                self._render_filters(config['filters'])

        # Main dashboard content
        sections = [DashboardSection(**section_config) for section_config in config.get('sections', [])]

        # Apply layout
        layout_type = DashboardLayout(config.get('dashboard_layout', 'single_column'))

        if layout_type == DashboardLayout.SINGLE_COLUMN:
            self.layout_manager.render_single_column(sections)
        elif layout_type == DashboardLayout.TWO_COLUMN:
            self.layout_manager.render_two_column(sections)
        elif layout_type == DashboardLayout.THREE_COLUMN:
            self.layout_manager.render_three_column(sections)
        elif layout_type == DashboardLayout.GRID:
            self.layout_manager.render_grid(sections, config.get('grid_cols', 2))
        elif layout_type == DashboardLayout.TABS:
            self.layout_manager.render_tabs(sections)
        elif layout_type == DashboardLayout.SIDEBAR:
            self.layout_manager.render_sidebar(sections)

    def _render_filters(self, filters_config: List[Dict[str, Any]]):
        """Render filter components"""
        for filter_config in filters_config:
            filter_type = filter_config.get('type')

            if filter_type == 'date_range':
                start, end = self.interactive.date_range_filter(
                    filter_config['key'],
                    filter_config.get('default_start'),
                    filter_config.get('default_end')
                )
                self.state.update_filter(f"{filter_config['key']}_start", start)
                self.state.update_filter(f"{filter_config['key']}_end", end)

            elif filter_type == 'multi_select':
                selected = self.interactive.multi_select_filter(
                    filter_config['label'],
                    filter_config['options'],
                    filter_config['key'],
                    filter_config.get('default')
                )
                self.state.update_filter(filter_config['key'], selected)

            elif filter_type == 'category':
                selected = self.interactive.category_filter(
                    filter_config['label'],
                    filter_config['options'],
                    filter_config['key'],
                    filter_config.get('default')
                )
                self.state.update_filter(filter_config['key'], selected)
