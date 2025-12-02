"""
UI Enhancements Module - Glassmorphism Agritech Design
=======================================================
Professional SaaS-like styling for Spectral Soil Modeler.

Features:
- Glassmorphism design with semi-transparent cards
- Emerald Green + Electric Blue color palette
- Fixed contrast issues (file uploader visibility)
- Custom metric cards
- Professional typography (Inter font)
- Dark mode optimized for data dashboards
"""

import streamlit as st
from typing import Optional


def apply_modern_style():
    """
    Applies a complete CSS overhaul to the Streamlit app.
    Fixes contrast issues and applies a Glassmorphism dashboard look.
    
    Call this function at the very start of your app (after st.set_page_config).
    """
    st.markdown("""
        <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

        /* ROOT VARIABLES - Glassmorphism Agritech Theme */
        :root {
            --bg-color: #0e1117;
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-bg-hover: rgba(255, 255, 255, 0.08);
            --border-color: rgba(255, 255, 255, 0.1);
            --border-color-accent: rgba(46, 134, 171, 0.3);
            --primary-color: #2E86AB;
            --primary-light: #1a5f7a;
            --accent-color: #06D6A0;
            --accent-dark: #04a071;
            --success-color: #06D6A0;
            --warning-color: #FFD166;
            --error-color: #EF476F;
            --text-primary: #E0E0E0;
            --text-secondary: #A0A0A0;
            --text-tertiary: #696969;
        }

        /* GLOBAL RESET */
        * {
            box-sizing: border-box;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--text-primary);
        }

        /* CODE BLOCKS */
        code, pre {
            font-family: 'JetBrains Mono', 'Courier New', monospace;
        }

        /* MAIN APP CONTAINER */
        .stApp {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 80% 20%, rgba(46, 134, 171, 0.08), transparent 40%),
                radial-gradient(circle at 20% 80%, rgba(6, 214, 160, 0.05), transparent 40%);
            background-attachment: fixed;
        }

        /* HIDE DEFAULT STREAMLIT UI CRUFT */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        .viewerBadge_container { display: none; }

        /* MAIN CONTENT AREA */
        [data-testid="stAppViewContainer"] {
            background-color: transparent;
        }

        [data-testid="stMainBlockContainer"] {
            background-color: transparent;
            padding: 2rem 1rem;
        }

        /* === TYPOGRAPHY === */
        h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
            font-size: 2.2rem !important;
            line-height: 1.2 !important;
            background: linear-gradient(90deg, var(--accent-color) 0%, var(--primary-color) 100%);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            margin-bottom: 1.5rem !important;
        }

        h2 {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1.6rem !important;
            letter-spacing: -0.01em !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.8rem !important;
            border-bottom: 2px solid var(--accent-color) !important;
            padding-bottom: 0.5rem !important;
        }

        h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1.2rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.6rem !important;
        }

        p, span, label, div {
            color: var(--text-primary) !important;
        }

        /* === SIDEBAR === */
        section[data-testid="stSidebar"] {
            background-color: rgba(9, 11, 14, 0.95);
            border-right: 1px solid var(--border-color);
        }

        section[data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }

        /* Sidebar header */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2 {
            border: none;
            margin-bottom: 0.5rem !important;
        }

        /* === FILE UPLOADER FIX === */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }

        section[data-testid="stFileUploadDropzone"] {
            background-color: rgba(46, 134, 171, 0.05) !important;
            border: 2px dashed var(--primary-color) !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }

        section[data-testid="stFileUploadDropzone"]:hover {
            background-color: rgba(46, 134, 171, 0.1) !important;
            border-color: var(--accent-color) !important;
        }

        /* Ensure text inside file uploader is visible */
        section[data-testid="stFileUploadDropzone"] div,
        section[data-testid="stFileUploadDropzone"] span,
        section[data-testid="stFileUploadDropzone"] small,
        section[data-testid="stFileUploadDropzone"] p {
            color: var(--text-primary) !important;
        }

        /* The 'Browse files' button styling */
        section[data-testid="stFileUploadDropzone"] button,
        button[data-testid="baseButton-secondary"] {
            background-color: transparent !important;
            border: 1px solid var(--primary-color) !important;
            color: var(--primary-color) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        section[data-testid="stFileUploadDropzone"] button:hover,
        button[data-testid="baseButton-secondary"]:hover {
            background-color: rgba(46, 134, 171, 0.1) !important;
            border-color: var(--accent-color) !important;
            color: var(--accent-color) !important;
        }

        /* === METRIC CARDS === */
        [data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--border-color-accent);
            border-radius: 12px;
            padding: 20px !important;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        [data-testid="stMetric"]:hover {
            background: var(--card-bg-hover);
            border-color: var(--accent-color);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(46, 134, 171, 0.15);
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary);
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 0.5rem !important;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 2rem !important;
            font-weight: 800 !important;
        }

        [data-testid="stMetricDelta"] {
            color: var(--success-color) !important;
            font-weight: 600 !important;
        }

        /* === BUTTONS === */
        .stButton > button,
        button[data-testid="baseButton-primary"] {
            background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.7rem 1.2rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.2);
        }

        .stButton > button:hover,
        button[data-testid="baseButton-primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(46, 134, 171, 0.4);
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        }

        /* Secondary buttons */
        button[data-testid="baseButton-secondary"] {
            background-color: transparent !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        button[data-testid="baseButton-secondary"]:hover {
            border-color: var(--accent-color) !important;
            color: var(--accent-color) !important;
            background-color: rgba(6, 214, 160, 0.1) !important;
        }

        /* === EXPANDABLE SECTIONS === */
        div[data-testid="stExpander"] {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px !important;
            overflow: hidden;
        }

        div[data-testid="stExpander"] > button {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        div[data-testid="stExpander"] > button:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
        }

        /* === DATA FRAMES & TABLES === */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(10px) !important;
        }

        [data-testid="stDataFrame"] table,
        [data-testid="stTable"] table {
            background-color: transparent !important;
        }

        [data-testid="stDataFrame"] th,
        [data-testid="stTable"] th {
            background-color: rgba(46, 134, 171, 0.1) !important;
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            border-bottom: 2px solid var(--border-color-accent) !important;
        }

        [data-testid="stDataFrame"] td,
        [data-testid="stTable"] td {
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }

        /* === INPUTS & SELECTORS === */
        input, textarea, select {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
            padding: 0.6rem 0.8rem !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.3s ease !important;
        }

        input:focus, textarea:focus, select:focus {
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 2px rgba(6, 214, 160, 0.1) !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }

        /* === SLIDERS === */
        [data-testid="stSlider"] .stSlider {
            color: var(--text-primary) !important;
        }

        /* === RADIO BUTTONS & CHECKBOXES === */
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }

        /* === ALERTS & STATUS === */
        .stSuccess, [data-testid="stAlert"] {
            background-color: rgba(6, 214, 160, 0.1) !important;
            color: var(--success-color) !important;
            border-left: 4px solid var(--success-color) !important;
            border-radius: 8px !important;
        }

        .stWarning {
            background-color: rgba(255, 209, 102, 0.1) !important;
            color: var(--warning-color) !important;
            border-left: 4px solid var(--warning-color) !important;
            border-radius: 8px !important;
        }

        .stError {
            background-color: rgba(239, 71, 111, 0.1) !important;
            color: var(--error-color) !important;
            border-left: 4px solid var(--error-color) !important;
            border-radius: 8px !important;
        }

        .stInfo {
            background-color: rgba(46, 134, 171, 0.1) !important;
            color: var(--primary-color) !important;
            border-left: 4px solid var(--primary-color) !important;
            border-radius: 8px !important;
        }

        /* === SPINNER & LOADING === */
        .stSpinner > div {
            border-top-color: var(--accent-color) !important;
        }

        /* === PROGRESS BAR === */
        progress {
            accent-color: var(--accent-color) !important;
        }

        /* === PLOTLY CHARTS === */
        .plotly-graph-div {
            background-color: transparent !important;
        }

        /* === SELECTBOX & MULTISELECT === */
        [data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }

        /* === CUSTOM CARD CONTAINERS === */
        .custom-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }

        .custom-card:hover {
            background: var(--card-bg-hover);
            border-color: var(--accent-color);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(46, 134, 171, 0.15);
        }

        .custom-card-title {
            color: #ffffff;
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
        }

        /* === SECTION SEPARATOR === */
        .section-separator {
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-color), transparent);
            margin: 2rem 0;
            border: none;
        }

        </style>
    """, unsafe_allow_html=True)


def render_custom_metric(label: str, value: str, delta: Optional[str] = None, color: str = "blue"):
    """
    Renders a custom HTML metric card that looks better than st.metric.
    
    Parameters:
    -----------
    label : str
        The metric label (e.g., "Best R²")
    value : str
        The metric value (e.g., "0.8945")
    delta : str, optional
        Optional delta/change (e.g., "+0.05" or "-2%")
    color : str, optional
        Color theme: "blue", "green", "red", "orange", "purple"
    """
    color_map = {
        "blue": ("2E86AB", "46, 134, 171"),
        "green": ("06D6A0", "6, 214, 160"),
        "red": ("EF476F", "239, 71, 111"),
        "orange": ("FFD166", "255, 209, 102"),
        "purple": ("8338EC", "131, 56, 236"),
    }
    
    hex_color, rgb_color = color_map.get(color, color_map["blue"])
    
    delta_html = ""
    if delta:
        is_positive = "+" in delta or (delta.replace("%", "").replace("-", "").replace(".", "").isdigit() and float(delta.replace("%", "")) > 0)
        delta_color = "#06D6A0" if is_positive else "#EF476F"
        delta_sign = "↑" if is_positive else "↓"
        delta_html = f'<span style="color: {delta_color}; font-size: 0.85rem; font-weight: bold; margin-top: 4px; display: block;">{delta_sign} {delta}</span>'

    html = f"""
    <div class="custom-card" style="border-left: 4px solid #{hex_color};">
        <div style="color: var(--text-secondary); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
            {label}
        </div>
        <div style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-top: 8px;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, description: str = ""):
    """
    Renders a section header with optional description.
    
    Parameters:
    -----------
    title : str
        The section title
    description : str, optional
        Optional description text below the title
    """
    desc_html = f'<p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.3rem; margin-bottom: 0;">{description}</p>' if description else ""
    
    html = f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent-color);">
            {title}
        </h2>
        {desc_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_info_card(title: str, content: str, icon: str = "ℹ️"):
    """
    Renders an information card with icon and content.
    
    Parameters:
    -----------
    title : str
        Card title
    content : str
        Card content (can include HTML)
    icon : str, optional
        Emoji or icon to display
    """
    html = f"""
    <div class="custom-card" style="border-left: 4px solid var(--primary-color); background: rgba(46, 134, 171, 0.08);">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
            <span style="font-size: 1.3rem;">{icon}</span>
            <div style="color: #ffffff; font-weight: 700; font-size: 1rem;">{title}</div>
        </div>
        <div style="color: var(--text-primary); font-size: 0.95rem; line-height: 1.5;">
            {content}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stat_row(*stats):
    """
    Renders a row of statistics in a grid layout.
    
    Parameters:
    -----------
    *stats : tuple
        Variable number of tuples: (label, value, color)
        Example: render_stat_row(("R²", "0.89", "green"), ("RMSE", "1.23", "blue"))
    """
    cols = st.columns(len(stats))
    for col, (label, value, color) in zip(cols, stats):
        with col:
            render_custom_metric(label, value, color=color)


# Plotly chart styling
def style_plotly_chart(fig):
    """
    Apply glassmorphism styling to Plotly figures.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to style
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Styled figure
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.01)',
        font=dict(
            family="'Inter', sans-serif",
            color="#E0E0E0",
            size=12
        ),
        title_font_size=18,
        title_font_color="#ffffff",
        margin=dict(t=40, l=10, r=10, b=10),
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            linecolor='rgba(255,255,255,0.1)',
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            linecolor='rgba(255,255,255,0.1)',
        ),
    )
    
    # Style traces
    for trace in fig.data:
        if trace.name:
            trace.name = trace.name
    
    return fig
"""
Dashboard & Analytics Module
=============================
Comprehensive analytics dashboard with multiple views and insights.

Features:
- Overview dashboard with KPIs
- Performance analytics
- Technique comparison
- Model ranking
- Advanced filtering and search
- Export-ready visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure Plotly for Glassmorphism theme
pio.templates.default = "plotly_dark"


class DashboardKPIs:
    """Key Performance Indicators for dashboard."""
    
    @staticmethod
    def calculate_kpis(results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key performance indicators.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        Dict[str, float]
            KPI dictionary
        """
        return {
            'total_models_trained': len(results_df),
            'best_r2': float(results_df['Test_R²'].max()),
            'average_r2': float(results_df['Test_R²'].mean()),
            'worst_r2': float(results_df['Test_R²'].min()),
            'models_above_threshold': len(results_df[results_df['Test_R²'] > 0.7]),
            'success_rate': float(len(results_df[results_df['Test_R²'] > 0.5]) / len(results_df) * 100),
            'average_rmse': float(results_df['Test_RMSE'].mean()) if 'Test_RMSE' in results_df.columns else 0,
            'number_of_techniques': len(results_df['Technique'].unique()),
            'number_of_models': len(results_df['Model'].unique()),
        }
    
    
    @staticmethod
    def render_kpi_cards(kpis: Dict[str, float]):
        """Render KPI cards in Streamlit."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Total Models",
                int(kpis['total_models_trained'])
            )
        
        with col2:
            st.metric(
                "🏆 Best R²",
                f"{kpis['best_r2']:.4f}",
                delta=f"Avg: {kpis['average_r2']:.4f}",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "✅ Success Rate",
                f"{kpis['success_rate']:.1f}%",
                delta=f"{int(kpis['models_above_threshold'])} > 0.7"
            )
        
        with col4:
            st.metric(
                "🔄 Techniques",
                int(kpis['number_of_techniques']),
                delta=f"{int(kpis['number_of_models'])} algorithms"
            )


def style_plotly_figure(fig: go.Figure) -> go.Figure:
    """
    Apply Glassmorphism styling to Plotly figures.
    
    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to style
        
    Returns
    -------
    go.Figure
        Styled figure
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.01)',
        font=dict(
            family="'Inter', sans-serif",
            color="#E0E0E0",
            size=12
        ),
        title_font_size=16,
        title_font_color="#ffffff",
        margin=dict(t=40, l=10, r=10, b=10),
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            linecolor='rgba(255,255,255,0.1)',
            showline=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            linecolor='rgba(255,255,255,0.1)',
            showline=True,
        ),
    )
    return fig


class PerformanceAnalytics:
    """Performance analytics visualizations."""
    
    @staticmethod
    def create_performance_distribution(results_df: pd.DataFrame) -> go.Figure:
        """Create distribution chart of R² scores."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=results_df['Test_R²'],
            nbinsx=20,
            name='R² Distribution',
            marker_color='rgba(46, 134, 171, 0.7)',
            hovertemplate='<b>R² Range</b>: %{x:.4f}<br><b>Count</b>: %{y}<extra></extra>'
        ))
        
        fig.add_vline(
            x=results_df['Test_R²'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {results_df['Test_R²'].mean():.4f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title="📊 R² Score Distribution",
            xaxis_title="R² Score",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        return style_plotly_figure(fig)
    
    
    @staticmethod
    def create_model_ranking_chart(results_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Create model ranking chart."""
        top_models = results_df.nlargest(top_n, 'Test_R²')
        top_models['Label'] = top_models['Model'] + ' (' + top_models['Technique'] + ')'
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_models['Label'],
            x=top_models['Test_R²'],
            orientation='h',
            marker=dict(
                color=top_models['Test_R²'],
                colorscale='Viridis',
                showscale=True
            ),
            text=top_models['Test_R²'].round(4),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🏆 Top Performing Models",
            xaxis_title="R² Score",
            height=400,
            showlegend=False
        )
        
        return style_plotly_figure(fig)
    
    
    @staticmethod
    def create_technique_comparison(results_df: pd.DataFrame) -> go.Figure:
        """Create technique performance comparison."""
        technique_stats = results_df.groupby('Technique').agg({
            'Test_R²': ['mean', 'max', 'min', 'std']
        }).round(4)
        
        technique_stats.columns = ['Mean', 'Max', 'Min', 'Std']
        technique_stats = technique_stats.reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=technique_stats['Technique'],
            y=technique_stats['Mean'],
            name='Mean R²',
            marker_color='rgba(46, 134, 171, 0.7)',
            error_y=dict(type='data', array=technique_stats['Std'])
        ))
        
        fig.add_trace(go.Bar(
            x=technique_stats['Technique'],
            y=technique_stats['Max'],
            name='Max R²',
            marker_color='rgba(162, 35, 114, 0.5)'
        ))
        
        fig.update_layout(
            title="📈 Technique Comparison",
            xaxis_title="Technique",
            yaxis_title="R² Score",
            barmode='group',
            height=400
        )
        
        return style_plotly_figure(fig)
    
    
    @staticmethod
    def create_metric_scatter(results_df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of metrics."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_df['Test_R²'],
            y=results_df['Test_RMSE'] if 'Test_RMSE' in results_df.columns else results_df['Test_MAE'],
            mode='markers',
            marker=dict(
                size=10,
                color=results_df['Test_R²'],
                colorscale='Viridis',
                showscale=True,
                line=dict(width=2, color='white')
            ),
            text=results_df['Model'] + ' (' + results_df['Technique'] + ')',
            hovertemplate='<b>%{text}</b><br>R²: %{x:.4f}<br>RMSE: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🔍 R² vs RMSE",
            xaxis_title="R² Score",
            yaxis_title="RMSE",
            height=400
        )
        
        return style_plotly_figure(fig)


class DashboardFilters:
    """Interactive filtering utilities for dashboard."""
    
    @staticmethod
    def render_filter_panel(results_df: pd.DataFrame, key_prefix: str = "") -> Tuple[pd.DataFrame, Dict]:
        """
        Render filter panel and return filtered data.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Original data
        key_prefix : str
            Unique prefix for widget keys to avoid collisions
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Filtered data and filter params
        """
        st.markdown("### 🔍 Filter Results")
        
        filters = {}
        col1, col2, col3 = st.columns(3)
        
        # Model filter
        with col1:
            all_models = results_df['Model'].unique().tolist()
            selected_models = st.multiselect(
                "Filter by Model",
                all_models,
                default=all_models[:2] if len(all_models) > 2 else all_models,
                key=f"{key_prefix}_models"
            )
            filters['models'] = selected_models
        
        # Technique filter
        with col2:
            all_techniques = results_df['Technique'].unique().tolist()
            selected_techniques = st.multiselect(
                "Filter by Technique",
                all_techniques,
                default=all_techniques[:1] if len(all_techniques) > 1 else all_techniques,
                key=f"{key_prefix}_techniques"
            )
            filters['techniques'] = selected_techniques
        
        # R² threshold
        with col3:
            r2_threshold = st.slider(
                "Minimum R² Score",
                0.0, 1.0, 0.0,
                step=0.05,
                key=f"{key_prefix}_r2_threshold"
            )
            filters['r2_threshold'] = r2_threshold
        
        # Apply filters
        filtered_data = results_df[
            (results_df['Model'].isin(selected_models)) &
            (results_df['Technique'].isin(selected_techniques)) &
            (results_df['Test_R²'] >= r2_threshold)
        ]
        
        st.info(f"📊 Showing {len(filtered_data)} of {len(results_df)} results")
        
        return filtered_data, filters


class ComprehensiveDashboard:
    """Complete dashboard implementation."""
    
    @staticmethod
    def render_overview_dashboard(results_df: pd.DataFrame, paradigm: str = ""):
        """Render overview dashboard."""
        st.markdown(f"## 📊 {paradigm} Training Results Dashboard")
        
        # KPIs
        kpis = DashboardKPIs.calculate_kpis(results_df)
        DashboardKPIs.render_kpi_cards(kpis)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = PerformanceAnalytics.create_performance_distribution(results_df)
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            fig2 = PerformanceAnalytics.create_technique_comparison(results_df)
            st.plotly_chart(fig2, width='stretch')
        
        # Rankings
        st.markdown("---")
        st.markdown("### 🏆 Model Rankings")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig3 = PerformanceAnalytics.create_model_ranking_chart(results_df)
            st.plotly_chart(fig3, width='stretch')
        
        with col2:
            # Summary table
            top_5 = results_df.nlargest(5, 'Test_R²')[['Model', 'Technique', 'Test_R²']]
            st.markdown("**Top 5 Models**")
            st.dataframe(
                top_5.reset_index(drop=True),
                width='stretch',
                hide_index=True
            )
    
    
    @staticmethod
    def render_detailed_analytics(results_df: pd.DataFrame):
        """Render detailed analytics view."""
        st.markdown("## 📈 Detailed Analytics")
        
        # Filter panel
        filtered_data, filters = DashboardFilters.render_filter_panel(results_df)
        
        st.markdown("---")
        
        # Metrics table
        st.markdown("### 📋 Detailed Results")
        
        display_cols = ['Model', 'Technique', 'Test_R²', 'Test_RMSE', 'Test_MAE']
        display_cols = [col for col in display_cols if col in filtered_data.columns]
        
        st.dataframe(
            filtered_data[display_cols].sort_values('Test_R²', ascending=False),
            width='stretch'
        )
        
        st.markdown("---")
        
        # Advanced visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = PerformanceAnalytics.create_metric_scatter(filtered_data)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Statistics summary
            st.markdown("### 📊 Statistics Summary")
            
            stats_data = {
                'Metric': [
                    'Total Results',
                    'Best R²',
                    'Mean R²',
                    'Std Dev',
                    'Models Above 0.7',
                    'Success Rate'
                ],
                'Value': [
                    len(filtered_data),
                    f"{filtered_data['Test_R²'].max():.4f}",
                    f"{filtered_data['Test_R²'].mean():.4f}",
                    f"{filtered_data['Test_R²'].std():.4f}",
                    len(filtered_data[filtered_data['Test_R²'] > 0.7]),
                    f"{len(filtered_data[filtered_data['Test_R²'] > 0.5]) / len(filtered_data) * 100:.1f}%"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width='stretch', hide_index=True)
"""
Data Analytics Module
=====================
Comprehensive analytics for spectral soil data including:
- Dataset statistics and profiling
- Feature engineering and correlation analysis
- Distribution analysis
- Outlier detection
- Data quality assessment
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import skew, kurtosis
import logging

logger = logging.getLogger(__name__)


class DataProfiler:
    """Comprehensive data profiling and statistics."""
    
    @staticmethod
    def get_basic_statistics(data: pd.DataFrame) -> dict:
        """
        Calculate basic statistics for the dataset.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
            
        Returns
        -------
        dict
            Dictionary with basic statistics
        """
        return {
            'samples': data.shape[0],
            'features': data.shape[1],
            'memory_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'missing_values': data.isnull().sum().sum(),
            'missing_percent': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            'dtypes': data.dtypes.value_counts().to_dict(),
            'numeric_features': data.select_dtypes(include=[np.number]).shape[1],
            'categorical_features': data.select_dtypes(include=['object']).shape[1]
        }
    
    @staticmethod
    def get_column_statistics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed statistics for each column.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
            
        Returns
        -------
        pd.DataFrame
            Statistics for each column
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        stats_dict = {}
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            stats_dict[col] = {
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Std Dev': col_data.std(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Q1': col_data.quantile(0.25),
                'Q3': col_data.quantile(0.75),
                'IQR': col_data.quantile(0.75) - col_data.quantile(0.25),
                'Skewness': skew(col_data),
                'Kurtosis': kurtosis(col_data),
                'Missing': data[col].isnull().sum()
            }
        
        return pd.DataFrame(stats_dict).T
    
    @staticmethod
    def display_data_profile(data: pd.DataFrame):
        """Display comprehensive data profile in Streamlit."""
        st.markdown("### 📊 Dataset Profile")
        
        # Basic statistics
        basic_stats = DataProfiler.get_basic_statistics(data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Samples", f"{basic_stats['samples']:,}")
        with col2:
            st.metric("🎯 Features", f"{basic_stats['features']}")
        with col3:
            st.metric("💾 Memory", f"{basic_stats['memory_mb']:.2f} MB")
        with col4:
            st.metric("⚠️ Missing", f"{basic_stats['missing_percent']:.2f}%")
        
        st.markdown("---")
        
        # Detailed column statistics
        st.markdown("### 📈 Feature Statistics")
        col_stats = DataProfiler.get_column_statistics(data)
        st.dataframe(col_stats.round(4), width='stretch')


class FeatureAnalytics:
    """Feature engineering and correlation analysis."""
    
    @staticmethod
    def calculate_correlations(data: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        target : str, optional
            Target column name
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        numeric_data = data.select_dtypes(include=[np.number])
        return numeric_data.corr()
    
    @staticmethod
    def get_feature_target_correlation(data: pd.DataFrame, target: str) -> pd.Series:
        """
        Get correlation of features with target.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        target : str
            Target column name
            
        Returns
        -------
        pd.Series
            Correlation of each feature with target
        """
        numeric_data = data.select_dtypes(include=[np.number])
        if target in numeric_data.columns:
            target_corr = numeric_data.corr()[target].drop(target).sort_values(ascending=False)
            return target_corr
        return pd.Series()
    
    @staticmethod
    def display_correlation_analysis(data: pd.DataFrame, target: str = None):
        """Display correlation analysis in Streamlit."""
        st.markdown("### 🔗 Feature Correlations")
        
        if target and target in data.columns:
            # Target correlation
            st.markdown("#### Target Correlation")
            target_corr = FeatureAnalytics.get_feature_target_correlation(data, target)
            
            # Bar plot
            fig = px.bar(
                x=target_corr.values,
                y=target_corr.index,
                orientation='h',
                title=f"Feature Correlation with {target}",
                labels={'x': 'Correlation', 'y': 'Feature'},
                color=target_corr.values,
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch', key="tgt_corr_001")
            
            st.markdown("---")
        
        # Full correlation heatmap
        st.markdown("#### Correlation Matrix")
        corr_matrix = FeatureAnalytics.calculate_correlations(data)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            width=600
        )
        st.plotly_chart(fig, width='stretch', key="corr_heatmap_002")


class DistributionAnalytics:
    """Distribution and outlier analysis."""
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> dict:
        """
        Detect outliers using IQR or Z-score method.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        method : str
            'iqr' or 'zscore'
        threshold : float
            Threshold for detection
            
        Returns
        -------
        dict
            Dictionary with outlier information
        """
        numeric_data = data.select_dtypes(include=[np.number])
        outliers = {}
        
        if method == 'iqr':
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(numeric_data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        elif method == 'zscore':
            for col in numeric_data.columns:
                z_scores = np.abs(stats.zscore(numeric_data[col].dropna()))
                outlier_mask = z_scores > threshold
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(numeric_data)) * 100
                }
        
        return outliers
    
    @staticmethod
    def display_distribution_analysis(data: pd.DataFrame, target: str = None):
        """Display distribution analysis in Streamlit."""
        st.markdown("### 📉 Distribution Analysis")
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Target Distribution")
            if target and target in numeric_data.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=numeric_data[target],
                    nbinsx=30,
                    name='Distribution',
                    marker_color='rgba(46, 134, 171, 0.7)'
                ))
                fig.add_vline(
                    x=numeric_data[target].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {numeric_data[target].mean():.2f}"
                )
                fig.update_layout(
                    title=f"{target} Distribution",
                    xaxis_title=target,
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, width='stretch', key="tgt_dist_003")
        
        with col2:
            st.markdown("#### Outlier Detection")
            outliers = DistributionAnalytics.detect_outliers(data, method='iqr')
            
            outlier_counts = {col: v['count'] for col, v in outliers.items()}
            outlier_df = pd.Series(outlier_counts).sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=outlier_df.values,
                y=outlier_df.index,
                orientation='h',
                title="Top 10 Features with Outliers",
                labels={'x': 'Outlier Count', 'y': 'Feature'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch', key="outlier_det_004")
        
        st.markdown("---")
        
        # Box plots for top features
        st.markdown("#### Box Plots - Top Features")
        top_features = numeric_data.std().nlargest(6).index.tolist()
        
        if top_features:
            fig = go.Figure()
            for feature in top_features:
                fig.add_trace(go.Box(y=numeric_data[feature], name=feature))
            
            fig.update_layout(
                title="Box Plots - Top 6 Features by Variance",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, width='stretch', key="boxplot_005")


class DataQualityAnalytics:
    """Data quality assessment."""
    
    @staticmethod
    def assess_quality(data: pd.DataFrame) -> dict:
        """
        Assess overall data quality.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
            
        Returns
        -------
        dict
            Quality metrics
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        completeness = (1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))) * 100
        
        # Check for duplicates
        duplicate_rows = data.duplicated().sum()
        duplicate_percent = (duplicate_rows / len(data)) * 100
        
        # Check variance
        low_variance_cols = []
        for col in numeric_data.columns:
            if numeric_data[col].std() < 0.01:
                low_variance_cols.append(col)
        
        # Check normality (Shapiro-Wilk test)
        normality_scores = {}
        for col in numeric_data.columns:
            if len(numeric_data[col].dropna()) > 5000:  # Shapiro-Wilk limited to 5000 samples
                sample = numeric_data[col].dropna().sample(n=5000, random_state=42)
            else:
                sample = numeric_data[col].dropna()
            
            if len(sample) > 3:
                _, p_value = stats.shapiro(sample)
                normality_scores[col] = p_value
        
        return {
            'completeness': completeness,
            'duplicate_rows': duplicate_rows,
            'duplicate_percent': duplicate_percent,
            'low_variance_cols': low_variance_cols,
            'normality_scores': normality_scores
        }
    
    @staticmethod
    def display_quality_assessment(data: pd.DataFrame):
        """Display data quality assessment in Streamlit."""
        st.markdown("### ✅ Data Quality Assessment")
        
        quality = DataQualityAnalytics.assess_quality(data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completeness_pct = quality['completeness']
            color = "🟢" if completeness_pct > 95 else "🟡" if completeness_pct > 80 else "🔴"
            st.metric(f"{color} Completeness", f"{completeness_pct:.1f}%")
        
        with col2:
            dup_pct = quality['duplicate_percent']
            color = "🟢" if dup_pct < 1 else "🟡" if dup_pct < 5 else "🔴"
            st.metric(f"{color} Duplicates", f"{quality['duplicate_rows']} ({dup_pct:.2f}%)")
        
        with col3:
            low_var = len(quality['low_variance_cols'])
            color = "🟢" if low_var == 0 else "🟡" if low_var < 3 else "🔴"
            st.metric(f"{color} Low Variance", f"{low_var} cols")
        
        with col4:
            normal_count = sum(1 for p in quality['normality_scores'].values() if p > 0.05)
            st.metric("📊 Normal Distribution", f"{normal_count}/{len(quality['normality_scores'])}")
        
        st.markdown("---")
        
        if quality['low_variance_cols']:
            st.warning(f"⚠️ Low variance detected in: {', '.join(quality['low_variance_cols'][:5])}")


class DataAnalyticsUI:
    """Unified UI for all data analytics."""
    
    @staticmethod
    def render_data_analytics(data: pd.DataFrame, target: str = None):
        """
        Render comprehensive data analytics interface.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        target : str, optional
            Target column name
        """
        st.markdown("## 📊 Data Analytics & Profiling")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Profile",
            "🔗 Correlations",
            "📉 Distributions",
            "✅ Quality"
        ])
        
        with tab1:
            DataProfiler.display_data_profile(data)
        
        with tab2:
            FeatureAnalytics.display_correlation_analysis(data, target)
        
        with tab3:
            DistributionAnalytics.display_distribution_analysis(data, target)
        
        with tab4:
            DataQualityAnalytics.display_quality_assessment(data)
"""
App Enhancement Integration
============================
High-level functions for integrating all new UI and analytics features into the main app.

This module provides easy-to-use wrapper functions for incorporating:
- Modern UI themes
- Results export functionality
- AI-powered reports
- Model analysis and statistics
- Conversational AI interface
- Comprehensive dashboards
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from services import ContextBuilder

logger = logging.getLogger(__name__)


def setup_enhanced_ui(theme: str = "scientific"):
    """
    Initialize enhanced UI with modern CSS theme.
    
    Parameters
    ----------
    theme : str
        Theme name: 'scientific', 'dark', 'professional', 'light'
    """
    try:
        apply_modern_style()
    except Exception as e:
        logger.warning(f"Could not initialize enhanced UI: {e}")


def render_training_results_section(results_df: pd.DataFrame,
                                   paradigm: str = "Standard",
                                   include_exports: bool = True,
                                   include_reports: bool = True,
                                   include_analytics: bool = True,
                                   is_individual_paradigm: bool = True):
    """
    Render comprehensive training results section with all enhancements.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Training results
    paradigm : str
        Training paradigm (Standard, Tuned, etc.)
    include_exports : bool
        Include export functionality
    include_reports : bool
        Include AI reports
    include_analytics : bool
        Include analytics dashboard
    is_individual_paradigm : bool
        If False (e.g., when part of comparison), skip AI tab (will be shown in comparison)
    """
    import streamlit as st
    
    st.markdown(f"## {paradigm} Training Results")
    
    # Determine which tabs to create based on options
    tab_names = ["📊 Overview", "📈 Analytics", "📋 Model Details"]
    # Only include AI Insights tab if this is an individual paradigm AND reports are enabled
    include_ai_tab = include_reports and is_individual_paradigm
    if include_ai_tab:
        tab_names.append("💬 AI Insights")
    if include_exports:
        tab_names.append("📥 Export")
    
    tabs = st.tabs(tab_names)
    tab_idx = 0
    
    # Tab 1: Overview
    with tabs[tab_idx]:
        render_overview_tab(results_df, paradigm)
    tab_idx += 1
    
    # Tab 2: Analytics
    if include_analytics:
        with tabs[tab_idx]:
            render_analytics_tab(results_df, paradigm)
        tab_idx += 1
    
    # Tab 3: Model Details
    with tabs[tab_idx]:
        render_model_details_tab(results_df, paradigm)
    tab_idx += 1
    
    # Tab 4: AI Insights (if enabled and this is an individual paradigm)
    if include_ai_tab:
        with tabs[tab_idx]:
            render_ai_insights_tab(results_df, paradigm)
        tab_idx += 1
    
    # Tab 5: Export (if enabled)
    if include_exports:
        with tabs[tab_idx]:
            render_export_tab(results_df, paradigm)


def render_overview_tab(results_df: pd.DataFrame, paradigm: str):
    """Render overview tab with KPIs and summary."""
    try:
        
        # Display clear performance metrics
        st.markdown("### 📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_r2 = float(results_df['Test_R²'].max())
            st.metric("🏆 Best R²", f"{best_r2:.4f}")
        
        with col2:
            mean_r2 = float(results_df['Test_R²'].mean())
            st.metric("📈 Mean R²", f"{mean_r2:.4f}", delta=f"{(best_r2 - mean_r2):.4f} gap")
        
        with col3:
            best_rmse = float(results_df['Test_RMSE'].min()) if 'Test_RMSE' in results_df.columns else 0
            st.metric("✅ Best RMSE", f"{best_rmse:.6f}")
        
        with col4:
            mean_rmse = float(results_df['Test_RMSE'].mean()) if 'Test_RMSE' in results_df.columns else 0
            st.metric("📊 Mean RMSE", f"{mean_rmse:.6f}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = PerformanceAnalytics.create_performance_distribution(results_df)
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            fig2 = PerformanceAnalytics.create_technique_comparison(results_df)
            st.plotly_chart(fig2, width='stretch')
        
        # Top models
        st.markdown("### 🏆 Top Performing Models")
        top_5 = results_df.nlargest(5, 'Test_R²')[['Model', 'Technique', 'Test_R²']]
        st.dataframe(
            top_5.reset_index(drop=True),
            width='stretch',
            hide_index=True
        )
        
        # Additional metrics overview
        st.markdown("---")
        st.markdown("### 📊 Additional Metrics Summary")
        
        # Calculate additional stats
        if 'Test_RMSE' in results_df.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best RMSE", f"{results_df['Test_RMSE'].min():.6f}")
            
            with col2:
                st.metric("Mean RMSE", f"{results_df['Test_RMSE'].mean():.6f}")
            
            with col3:
                st.metric("Worst RMSE", f"{results_df['Test_RMSE'].max():.6f}")
        
        if 'Test_MAE' in results_df.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best MAE", f"{results_df['Test_MAE'].min():.6f}")
            
            with col2:
                st.metric("Mean MAE", f"{results_df['Test_MAE'].mean():.6f}")
            
            with col3:
                st.metric("Worst MAE", f"{results_df['Test_MAE'].max():.6f}")
        
        if 'RPD' in results_df.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best RPD", f"{results_df['RPD'].max():.4f}")
            
            with col2:
                st.metric("Mean RPD", f"{results_df['RPD'].mean():.4f}")
            
            with col3:
                st.metric("Quality", "Excellent" if results_df['RPD'].mean() > 3 else "Good" if results_df['RPD'].mean() > 2 else "Fair")
    
    except Exception as e:
        st.error(f"Error rendering overview: {e}")
        logger.error(f"Overview rendering error: {e}", exc_info=True)


def render_analytics_tab(results_df: pd.DataFrame, paradigm: str = ""):
    """Render analytics tab with filtering and detailed charts."""
    try:
        
        # Create sub-tabs for different analysis types
        subtab1, subtab2, subtab3 = st.tabs([
            "🔍 Basic Filters",
            "📈 Comprehensive Analysis",
            "📊 Advanced Metrics"
        ])
        
        with subtab1:
            # Filter panel - use paradigm as key prefix to avoid collisions
            key_prefix = f"analytics_{paradigm.lower()}" if paradigm else "analytics"
            filtered_data, filters = DashboardFilters.render_filter_panel(results_df, key_prefix=key_prefix)
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = PerformanceAnalytics.create_model_ranking_chart(filtered_data)
                st.plotly_chart(fig1, width='stretch')
            
            with col2:
                fig2 = PerformanceAnalytics.create_metric_scatter(filtered_data)
                st.plotly_chart(fig2, width='stretch')
            
            # Detailed table
            st.markdown("### 📋 Filtered Results")
            display_cols = ['Model', 'Technique', 'Test_R²', 'Test_RMSE', 'Test_MAE']
            display_cols = [col for col in display_cols if col in filtered_data.columns]
            
            st.dataframe(
                filtered_data[display_cols].sort_values('Test_R²', ascending=False),
                width='stretch'
            )
        
        with subtab2:
            # Comprehensive metrics analysis
            analysis_key_prefix = f"analytics_comprehensive_{paradigm.lower()}" if paradigm else "analytics_comprehensive"
            MetricsAnalyzer.display_comprehensive_analysis(results_df, key_prefix=analysis_key_prefix)
        
        with subtab3:
            # Metric-specific filtering and analysis
            filter_key_prefix = f"analytics_filter_{paradigm.lower()}" if paradigm else "analytics_filter"
            MetricsAnalyzer.display_metric_filter_analysis(results_df, key_prefix=filter_key_prefix)
    
    except Exception as e:
        st.error(f"Error rendering analytics: {e}")
        logger.error(f"Analytics rendering error: {e}", exc_info=True)


def render_model_details_tab(results_df: pd.DataFrame, paradigm: str = ""):
    """Render model details and statistics tab."""
    try:
        from model_analyzer import ModelAnalyzer, PerformanceComparator
        
        
        # Create sub-tabs
        subtab1, subtab2 = st.tabs(["Basic Comparison", "Detailed Metrics"])
        
        with subtab1:
            # Model selection
            col1, col2 = st.columns(2)
            
            # Generate unique keys based on paradigm
            key_prefix = f"modeldetails_{paradigm.lower()}" if paradigm else "modeldetails"
            
            with col1:
                selected_model = st.selectbox(
                    "Select Model for Detailed Analysis",
                    results_df['Model'].unique(),
                    key=f"{key_prefix}_model_select"
                )
            
            with col2:
                selected_technique = st.selectbox(
                    "Select Technique",
                    results_df['Technique'].unique(),
                    key=f"{key_prefix}_technique_select"
                )
            
            # Get statistics
            stats = ModelAnalyzer.get_combination_statistics(results_df, selected_model, selected_technique)
            
            if 'error' not in stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{stats['metrics']['test_r2']:.4f}")
                with col2:
                    st.metric("RMSE", f"{stats['metrics']['test_rmse']:.4f}")
                with col3:
                    st.metric("MAE", f"{stats['metrics']['test_mae']:.4f}")
                with col4:
                    st.metric("Quality", stats['quality_assessment'])
            
            st.markdown("---")
            
            # Comparison matrices
            st.markdown("### 📊 Performance Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Models Comparison**")
                model_comparison = ModelAnalyzer.compare_models(
                    results_df,
                    results_df['Model'].unique()[:5]
                )
                st.dataframe(model_comparison, width='stretch')
            
            with col2:
                st.markdown("**Techniques Comparison**")
                tech_comparison = ModelAnalyzer.compare_techniques(results_df)
                st.dataframe(tech_comparison, width='stretch')
            
            # Ranking
            st.markdown("### 🏅 Overall Ranking")
            ranking = PerformanceComparator.calculate_ranking(results_df)
            st.dataframe(ranking, width='stretch', hide_index=True)
        
        with subtab2:
            # Detailed metrics analysis
            st.markdown("### 📈 Detailed Metrics Analysis")
            details_key_prefix = f"modeldetails_comprehensive_{paradigm.lower()}" if paradigm else "modeldetails_comprehensive"
            MetricsAnalyzer.display_comprehensive_analysis(results_df, key_prefix=details_key_prefix)
    
    except Exception as e:
        st.error(f"Error rendering model details: {e}")
        logger.error(f"Model details rendering error: {e}", exc_info=True)


def render_ai_insights_tab(results_df: pd.DataFrame, paradigm: str):
    """Render AI insights and conversational interface tab."""
    try:
        from services import ReportGenerator, StreamlitChatUI, ChatInterface
        
        # AI Report Generation
        st.markdown("### 📄 AI-Generated Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate unique key based on paradigm
            button_key = f"generate_report_{paradigm.lower()}" if paradigm else "generate_report"
            if st.button("📝 Generate Report", key=button_key):
                try:
                    with st.spinner("Generating AI report..."):
                        # Get raw data and target from session state if available
                        raw_data = st.session_state.get('raw_data', None)
                        target_col = st.session_state.get('target_col', None)
                        
                        gen = ReportGenerator(
                            raw_data=raw_data,
                            target_col=target_col
                        )
                        if not gen.ai_available:
                            st.warning("⚠️ AI features are not available. Please install google-generativeai or openai and set API keys.")
                            st.info("To enable AI:\n1. pip install google-generativeai openai\n2. Set GOOGLE_API_KEY or OPENAI_API_KEY environment variables")
                        else:
                            report = gen.generate_training_report(results_df, paradigm, include_ai_insights=True)
                            
                            # Display report
                            report_md = gen.format_report_as_markdown(report)
                            st.markdown(report_md)
                            
                            # Download report
                            st.download_button(
                                "📥 Download Report as Markdown",
                                report_md,
                                f"report_{paradigm.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                                "text/markdown",
                                key=f"download_report_{paradigm.lower()}_{id(report_md)}"
                            )
                            
                            st.success("✅ Report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        with col2:
            st.info("💡 Reports include:\n- Performance summary\n- Top models\n- AI insights\n- Recommendations")
        
        st.markdown("---")
        
        # Conversational AI
        st.markdown("### 💬 Ask AI Questions")
        
        # Prepare comprehensive context with full analytics
        try:
            # Get data context if available from session state
            data_context = st.session_state.get('data_analytics_context', '')
            
            # Build comprehensive training context with explicit paradigm
            training_context = ContextBuilder.build_training_context(
                results_df, 
                st.session_state.get('raw_data', pd.DataFrame()),
                st.session_state.get('target_col', 'target'),
                paradigm=paradigm,
                data_analytics_context=data_context
            )
            
            # Extract hyperparameters early so we can add them to context string
            best_idx = results_df['Test_R²'].idxmax()
            best_hyperparams_dict = {}
            
            # Try to get hyperparameters from Hyperparameters column first (most reliable)
            if 'Hyperparameters' in results_df.columns:
                try:
                    hyperparams_value = results_df.loc[best_idx, 'Hyperparameters']
                    if isinstance(hyperparams_value, dict) and hyperparams_value:  # Check if dict is not empty
                        best_hyperparams_dict = hyperparams_value
                except Exception as e:
                    logger.debug(f"Could not extract hyperparameters from column: {e}")
            
            # Fallback to Model_Object if Hyperparameters column not available or empty
            if not best_hyperparams_dict and 'Model_Object' in results_df.columns:
                try:
                    from model_analyzer import ParameterInspector
                    best_model_obj = results_df.loc[best_idx, 'Model_Object']
                    best_hyperparams_dict = ParameterInspector.get_hyperparameters(best_model_obj)
                except Exception as e:
                    logger.debug(f"Could not extract hyperparameters: {e}")
            
            # Enhance training context with paradigm and hyperparameters
            if training_context and best_hyperparams_dict:
                hyperparams_str = "\nKEY HYPERPARAMETERS FOR BEST MODEL:\n"
                for k, v in list(best_hyperparams_dict.items())[:8]:
                    hyperparams_str += f"  • {k}: {v}\n"
                
                # Insert paradigm info early in the context
                paradigm_section = f"\nTRAINING PARADIGM: {paradigm}\n"
                if paradigm.lower() == 'tuned':
                    paradigm_section += "  • Hyperparameter tuning applied using cross-validation\n"
                elif paradigm.lower() == 'standard':
                    paradigm_section += "  • Standard training with default hyperparameters\n"
                
                training_context = training_context.replace(
                    "TRAINING RESULTS CONTEXT",
                    f"TRAINING RESULTS CONTEXT\n{paradigm_section}",
                    1
                )
                # Add hyperparameters section before model results
                training_context = training_context.replace(
                    "ALL MODEL RESULTS",
                    f"{hyperparams_str}\nALL MODEL RESULTS",
                    1
                )
        except Exception as e:
            logger.warning(f"Could not build detailed context: {e}")
            training_context = None
        
        # Prepare context dictionary for UI
        best_model_val = str(results_df.loc[best_idx, 'Model']) if pd.notna(results_df.loc[best_idx, 'Model']) else 'Unknown'
        best_technique_val = str(results_df.loc[best_idx, 'Technique']) if pd.notna(results_df.loc[best_idx, 'Technique']) else 'Unknown'
        
        # Format hyperparameters for context (convert to string dict if not already done)
        best_hyperparams = {k: str(v) for k, v in list(best_hyperparams_dict.items())[:10]} if best_hyperparams_dict else {}
        
        context = {
            'summary': {
                'total_models': len(results_df),
                'best_r2': float(results_df['Test_R²'].max()),
                'mean_r2': float(results_df['Test_R²'].mean()),
                'best_model': best_model_val,
                'best_technique': best_technique_val,
                'paradigm': paradigm,
                'hyperparameters': best_hyperparams,
            },
            'top_models': results_df.nlargest(5, 'Test_R²')[['Model', 'Technique', 'Test_R²']].to_dict('records'),
            'all_results': results_df[['Model', 'Technique', 'Test_R²'] + [col for col in results_df.columns if 'RMSE' in col or 'MAE' in col]].head(20).to_dict('records'),
            'statistics': {
                'by_technique': {
                    tech: {
                        'mean_r2': float(results_df[results_df['Technique'] == tech]['Test_R²'].mean()),
                        'count': len(results_df[results_df['Technique'] == tech])
                    }
                    for tech in results_df['Technique'].unique() if pd.notna(tech)
                } if 'Technique' in results_df.columns else {}
            },
            'training_context_str': training_context,  # Full context string for AI
        }
        
        # Initialize chat
        if 'chat_interface' not in st.session_state or st.session_state.get('chat_interface_provider') != st.session_state.ai_provider:
            st.session_state.chat_interface = ChatInterface(ai_provider=st.session_state.ai_provider)
            st.session_state.chat_interface_provider = st.session_state.ai_provider
        
        chat = st.session_state.chat_interface
        
        if not chat.ai_available:
            import os
            if st.session_state.ai_provider == 'gemini':
                st.warning("⚠️ Gemini not available. Please set GOOGLE_API_KEY environment variable.")
            else:
                st.warning("⚠️ ChatGPT not available. Please set OPENAI_API_KEY environment variable.")
        else:
            # Render chat UI with paradigm-specific key to avoid duplicate key errors
            paradigm_key = paradigm.lower().replace(' ', '_') if paradigm else 'results'
            chat_key_prefix = f"training_{paradigm_key}_chat"
            StreamlitChatUI.render_chat_interface(chat, context, chat_key_prefix)
    
    except Exception as e:
        st.warning(f"AI features not available: {str(e)}")
        st.info("To enable AI features:\n1. Install: pip install google-generativeai openai\n2. Set API keys: GOOGLE_API_KEY or OPENAI_API_KEY")


def render_export_tab(results_df: pd.DataFrame, paradigm: str):
    """Render export options tab."""
    try:
        from system import StreamlitExporter
        
        st.markdown("### 📥 Export Results")
        
        st.markdown("""
        Export your training results in multiple formats:
        - **CSV**: Simple spreadsheet format for analysis
        - **Excel**: Formatted spreadsheet with multiple sheets
        - **JSON**: Structured format for integration
        """)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 CSV Export")
            StreamlitExporter.get_csv_download_button(
                results_df,
                f"results_{paradigm.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "📥 Download CSV"
            )
        
        with col2:
            st.markdown("#### 📈 Excel Export")
            export_data = {
                'Results': results_df,
                'Summary': pd.DataFrame([{
                    'Total Models': len(results_df),
                    'Best R²': results_df['Test_R²'].max(),
                    'Mean R²': results_df['Test_R²'].mean(),
                }])
            }
            StreamlitExporter.get_excel_download_button(
                export_data,
                f"results_{paradigm.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "📥 Download Excel"
            )
        
        with col3:
            st.markdown("#### 📄 JSON Export")
            export_json = {
                'paradigm': paradigm,
                'results': results_df.to_dict(orient='records'),
                'summary': {
                    'total_models': len(results_df),
                    'best_r2': float(results_df['Test_R²'].max()),
                    'mean_r2': float(results_df['Test_R²'].mean()),
                }
            }
            StreamlitExporter.get_json_download_button(
                export_json,
                f"results_{paradigm.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                "📥 Download JSON"
            )
    
    except Exception as e:
        st.error(f"Error rendering export: {e}")


def render_prediction_model_inspection(model, model_metadata: Dict):
    """
    Render model parameters inspection in prediction mode.
    
    Parameters
    ----------
    model : object
        Trained model instance
    model_metadata : Dict
        Model metadata
    """
    try:
        from model_analyzer import ParameterInspector
        
        st.markdown("### 🔧 Model Parameters Inspection")
        
        # Ensure metadata is a dict
        if not isinstance(model_metadata, dict):
            model_metadata = {}
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algo = model_metadata.get('algorithm', 'Unknown')
            st.metric("Algorithm", algo if algo else 'Unknown')
        with col2:
            tech = model_metadata.get('technique', 'Unknown')
            st.metric("Technique", tech if tech else 'Unknown')
        with col3:
            r2_val = model_metadata.get('test_r2', None)
            if r2_val is not None:
                try:
                    r2_float = float(r2_val)
                    if r2_float > 0 and r2_float <= 1:
                        st.metric("Test R²", f"{r2_float:.4f}")
                    else:
                        st.metric("Test R²", "Not available")
                except:
                    st.metric("Test R²", "Not available")
            else:
                st.metric("Test R²", "Not available")
        
        st.markdown("---")
        
        # Hyperparameters
        st.markdown("#### ⚙️ Hyperparameters")
        try:
            hyperparams = ParameterInspector.get_hyperparameters(model)
            
            if hyperparams:
                formatted_params = ParameterInspector.format_parameters_for_display(hyperparams)
                
                params_df = pd.DataFrame([formatted_params]).T
                params_df.columns = ['Value']
                st.dataframe(params_df, width='stretch')
            else:
                st.info("No hyperparameters available for this model type.")
        except Exception as e:
            st.warning(f"Could not extract hyperparameters: {e}")
        
        # Model parameters
        st.markdown("#### 📊 Model Parameters")
        try:
            params = ParameterInspector.extract_parameters(model)
            
            if params:
                for param_name, param_value in params.items():
                    with st.expander(f"View {param_name}"):
                        if isinstance(param_value, dict):
                            st.json(param_value)
                        elif hasattr(param_value, 'shape'):
                            st.write(f"Shape: {param_value.shape}")
                            st.write(f"Type: {type(param_value).__name__}")
                            if hasattr(param_value, '__len__') and len(param_value) <= 20:
                                st.write(param_value)
                        else:
                            st.write(param_value)
            else:
                st.info("No parameters available for this model type.")
        except Exception as e:
            st.warning(f"Could not display model parameters: {e}")
    
    except Exception as e:
        st.warning(f"Model inspection features not available: {str(e)}")
        st.info("Advanced model inspection requires additional libraries.")


def _build_algorithm_performance_section(results_df: pd.DataFrame) -> str:
    """Build performance by algorithm section showing model consistency."""
    lines = []
    if 'Model' in results_df.columns:
        models = results_df['Model'].unique()
        for model in sorted(models):
            model_data = results_df[results_df['Model'] == model]
            lines.append(f"  • {model}:")
            lines.append(f"    - Techniques tested: {len(model_data)}")
            lines.append(f"    - Best R²: {model_data['Test_R²'].max():.6f}")
            lines.append(f"    - Mean R²: {model_data['Test_R²'].mean():.6f}")
            lines.append(f"    - Std Dev (consistency metric): {model_data['Test_R²'].std():.6f}")
            lines.append(f"    - Performance by technique:")
            for technique in sorted(results_df['Technique'].unique()):
                tech_model_data = model_data[model_data['Technique'] == technique]
                if len(tech_model_data) > 0:
                    r2_val = tech_model_data['Test_R²'].values[0]
                    lines.append(f"      • {technique}: R²={r2_val:.6f}")
    return "\n".join(lines)


def _build_all_models_section(results_df: pd.DataFrame, paradigm_name: str) -> str:
    """Build section showing all models ranked by R²."""
    lines = []
    for rank, (idx, row) in enumerate(results_df.nlargest(len(results_df), 'Test_R²').iterrows(), 1):
        model_line = f"  {rank:2d}. [{paradigm_name}] {row['Model']} + {row['Technique']} → R²={row['Test_R²']:.6f}"
        if 'Test_RMSE' in results_df.columns and pd.notna(row['Test_RMSE']):
            model_line += f" | RMSE={row['Test_RMSE']:.6f}"
        if 'Test_MAE' in results_df.columns and pd.notna(row['Test_MAE']):
            model_line += f" | MAE={row['Test_MAE']:.6f}"
        lines.append(model_line)
    return "\n".join(lines)


def render_comparison_mode(standard_results: pd.DataFrame,
                          tuned_results: pd.DataFrame):
    """
    Render comparison mode for Standard vs Tuned models.
    Includes AI insights only in this comparison section (not duplicated in individual tabs).
    
    Parameters
    ----------
    standard_results : pd.DataFrame
        Standard training results
    tuned_results : pd.DataFrame
        Tuned training results
    """
    try:
        from services import ReportGenerator, StreamlitChatUI, ChatInterface
        
        st.markdown("## 📊 Standard vs Tuned Comparison")
        
        # Validate data
        if standard_results is None or len(standard_results) == 0:
            st.error("Standard training results are empty")
            return
        
        if tuned_results is None or len(tuned_results) == 0:
            st.error("Tuned training results are empty")
            return
        
        # Generate comparison report
        try:
            # Get raw data and target from session state if available
            raw_data = st.session_state.get('raw_data', None)
            target_col = st.session_state.get('target_col', None)
            
            gen = ReportGenerator(
                raw_data=raw_data,
                target_col=target_col
            )
            comparison = gen.generate_comparison_report(standard_results, tuned_results)
        except Exception as e:
            st.error(f"Error generating comparison report: {e}")
            logger.error(f"Comparison report error: {e}", exc_info=True)
            return
        
        # Create tabs for comparison: Charts, Analysis, and AI Insights
        comp_tab1, comp_tab2, comp_tab3 = st.tabs(["📊 Comparison Charts", "📈 Detailed Analysis", "💬 AI Comparison Insights"])
        
        with comp_tab1:
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                std_best = comparison['standard_summary'].get('best_r2', 0)
                st.metric(
                    "Standard Best R²",
                    f"{std_best:.4f}" if std_best else "N/A"
                )
            
            with col2:
                tuned_best = comparison['tuned_summary'].get('best_r2', 0)
                st.metric(
                    "Tuned Best R²",
                    f"{tuned_best:.4f}" if tuned_best else "N/A"
                )
            
            with col3:
                improvement = comparison['improvements'].get('improvement_percent', 0)
                # For delta color coding: positive = green, negative/zero = red
                delta_value = f"{improvement:.2f}%" if improvement != 0 else "-0.00%"
                st.metric(
                    "Improvement",
                    f"{improvement:.2f}%",
                    delta=delta_value
                )
            
            with col4:
                best = comparison['best_overall']
                best_model_str = f"{best.get('paradigm', 'Unknown')} - {best.get('model', 'Unknown')}"
                st.metric(
                    "Overall Best",
                    best_model_str
                )
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    fig1 = PerformanceAnalytics.create_performance_distribution(standard_results)
                    fig1.update_layout(title="Standard Distribution")
                    st.plotly_chart(fig1, width='stretch')
                except Exception as e:
                    st.warning(f"Could not create standard distribution chart: {e}")
            
            with col2:
                try:
                    fig2 = PerformanceAnalytics.create_performance_distribution(tuned_results)
                    fig2.update_layout(title="Tuned Distribution")
                    st.plotly_chart(fig2, width='stretch')
                except Exception as e:
                    st.warning(f"Could not create tuned distribution chart: {e}")
        
        with comp_tab2:
            # Detailed comparison analysis
            st.markdown("### Detailed Metrics Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Standard Results Summary**")
                std_summary = pd.DataFrame([
                    {
                        'Metric': 'Best R²',
                        'Value': f"{comparison['standard_summary'].get('best_r2', 0):.4f}"
                    },
                    {
                        'Metric': 'Mean R²',
                        'Value': f"{comparison['standard_summary'].get('mean_r2', 0):.4f}"
                    },
                    {
                        'Metric': 'Models Trained',
                        'Value': str(len(standard_results))
                    }
                ])
                st.dataframe(std_summary, width='stretch', hide_index=True)
            
            with col2:
                st.markdown("**Tuned Results Summary**")
                tuned_summary = pd.DataFrame([
                    {
                        'Metric': 'Best R²',
                        'Value': f"{comparison['tuned_summary'].get('best_r2', 0):.4f}"
                    },
                    {
                        'Metric': 'Mean R²',
                        'Value': f"{comparison['tuned_summary'].get('mean_r2', 0):.4f}"
                    },
                    {
                        'Metric': 'Models Trained',
                        'Value': str(len(tuned_results))
                    }
                ])
                st.dataframe(tuned_summary, width='stretch', hide_index=True)
            
            st.markdown("---")
            st.markdown("**Improvements Summary**")
            if 'improvements' in comparison:
                improvements = pd.DataFrame([comparison['improvements']])
                st.dataframe(improvements, width='stretch', hide_index=True)
        
        with comp_tab3:
            # AI Insights for comparison (only shown here when in Both mode)
            st.markdown("### 🤖 AI-Powered Comparison Analysis")
            
            # Report generation for comparison
            st.markdown("#### 📄 AI-Generated Comparison Report")
            
            col_report1, col_report2 = st.columns(2)
            
            with col_report1:
                button_key = f"generate_comparison_report_{id(standard_results)}"
                if st.button("📝 Generate Comparison Report", key=button_key):
                    try:
                        with st.spinner("Generating comparison report..."):
                            from services import ReportGenerator
                            
                            # Get raw data and target from session state if available
                            raw_data = st.session_state.get('raw_data', None)
                            target_col = st.session_state.get('target_col', None)
                            
                            gen = ReportGenerator(
                                raw_data=raw_data,
                                target_col=target_col
                            )
                            if not gen.ai_available:
                                st.warning("⚠️ AI features are not available. Please install google-generativeai or openai and set API keys.")
                                st.info("To enable AI:\n1. pip install google-generativeai openai\n2. Set GOOGLE_API_KEY or OPENAI_API_KEY environment variables")
                            else:
                                # Generate comparison report
                                report = gen.generate_comparison_report(standard_results, tuned_results)
                                report_md = gen.format_report_as_markdown(report)
                                
                                # Store report in session state for display
                                st.session_state.comparison_report = report_md
                                
                                # Display report
                                st.markdown(report_md)
                                
                                # Download report
                                st.download_button(
                                    "📥 Download Comparison Report as Markdown",
                                    report_md,
                                    f"comparison_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    "text/markdown",
                                    key=f"download_comparison_report_{id(report_md)}"
                                )
                                
                                st.success("✅ Report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating comparison report: {str(e)}")
            
            # Display stored report if available (on subsequent views)
            if st.session_state.get('comparison_report', None):
                report_md = st.session_state.comparison_report
                st.markdown(report_md)
                
                # Download button
                st.download_button(
                    "📥 Download Comparison Report as Markdown",
                    report_md,
                    f"comparison_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    key=f"download_comparison_report_{id(report_md)}_display"
                )
            
            with col_report2:
                st.info("💡 Comparison reports include:\n- Standard vs Tuned analysis\n- Performance improvements\n- Model recommendations\n- Technique comparisons")
            
            st.markdown("---")
            
            try:
                # Determine which paradigm performed best
                standard_best_r2 = comparison['standard_summary'].get('best_r2', 0)
                tuned_best_r2 = comparison['tuned_summary'].get('best_r2', 0)
                best_overall = comparison['best_overall']
                
                # Get the best model details from both paradigms
                std_best_idx = standard_results['Test_R²'].idxmax()
                tuned_best_idx = tuned_results['Test_R²'].idxmax()
                
                std_best_model = str(standard_results.loc[std_best_idx, 'Model']) if pd.notna(standard_results.loc[std_best_idx, 'Model']) else 'Unknown'
                std_best_technique = str(standard_results.loc[std_best_idx, 'Technique']) if pd.notna(standard_results.loc[std_best_idx, 'Technique']) else 'Unknown'
                
                tuned_best_model = str(tuned_results.loc[tuned_best_idx, 'Model']) if pd.notna(tuned_results.loc[tuned_best_idx, 'Model']) else 'Unknown'
                tuned_best_technique = str(tuned_results.loc[tuned_best_idx, 'Technique']) if pd.notna(tuned_results.loc[tuned_best_idx, 'Technique']) else 'Unknown'
                
                # Determine overall winner
                if best_overall.get('paradigm') == 'Standard':
                    best_model_name = std_best_model
                    best_technique_name = std_best_technique
                    best_r2_val = standard_best_r2
                else:
                    best_model_name = tuned_best_model
                    best_technique_name = tuned_best_technique
                    best_r2_val = tuned_best_r2
                
                # Extract hyperparameters from best models using ParameterInspector
                std_hyperparams = {}
                tuned_hyperparams = {}
                
                # Try Standard hyperparameters from Hyperparameters column first
                if 'Hyperparameters' in standard_results.columns and pd.notna(standard_results.loc[std_best_idx, 'Hyperparameters']):
                    hyperparams_val = standard_results.loc[std_best_idx, 'Hyperparameters']
                    if isinstance(hyperparams_val, dict) and hyperparams_val:
                        std_hyperparams = {k: str(v) for k, v in list(hyperparams_val.items())[:8]}
                
                # Fallback for Standard: use Model_Object with ParameterInspector
                if not std_hyperparams and 'Model_Object' in standard_results.columns:
                    try:
                        from model_analyzer import ParameterInspector
                        std_model_obj = standard_results.loc[std_best_idx, 'Model_Object']
                        std_hyperparams = {k: str(v) for k, v in list(ParameterInspector.get_hyperparameters(std_model_obj).items())[:8]}
                    except Exception as e:
                        logger.debug(f"Could not extract standard hyperparameters: {e}")
                
                # Try Tuned hyperparameters from Hyperparameters column first
                if 'Hyperparameters' in tuned_results.columns and pd.notna(tuned_results.loc[tuned_best_idx, 'Hyperparameters']):
                    hyperparams_val = tuned_results.loc[tuned_best_idx, 'Hyperparameters']
                    if isinstance(hyperparams_val, dict) and hyperparams_val:
                        tuned_hyperparams = {k: str(v) for k, v in list(hyperparams_val.items())[:8]}
                
                # Fallback for Tuned: use Model_Object with ParameterInspector
                if not tuned_hyperparams and 'Model_Object' in tuned_results.columns:
                    try:
                        from model_analyzer import ParameterInspector
                        tuned_model_obj = tuned_results.loc[tuned_best_idx, 'Model_Object']
                        tuned_hyperparams = {k: str(v) for k, v in list(ParameterInspector.get_hyperparameters(tuned_model_obj).items())[:8]}
                    except Exception as e:
                        logger.debug(f"Could not extract tuned hyperparameters: {e}")
                
                # Prepare comprehensive context for comparison with proper structure
                combined_context = {
                    'summary': {
                        'total_models': len(standard_results) + len(tuned_results),
                        'best_model': best_model_name,
                        'best_technique': best_technique_name,
                        'best_r2': best_r2_val,
                        'mean_r2': (standard_results['Test_R²'].mean() + tuned_results['Test_R²'].mean()) / 2,
                        'paradigm': best_overall.get('paradigm', 'Unknown'),
                        'hyperparameters': std_hyperparams if best_overall.get('paradigm') == 'Standard' else tuned_hyperparams,
                        'standard_best_r2': standard_best_r2,
                        'tuned_best_r2': tuned_best_r2,
                        'improvement': comparison['improvements'].get('improvement_percent', 0),
                    },
                    'comparison': comparison,
                    'top_models_standard': standard_results.nlargest(3, 'Test_R²')[['Model', 'Technique', 'Test_R²']].to_dict('records'),
                    'top_models_tuned': tuned_results.nlargest(3, 'Test_R²')[['Model', 'Technique', 'Test_R²']].to_dict('records'),
                    'training_context_str': f"""
COMPARISON: Standard vs Tuned Training Paradigms

Standard Training Results:
- Paradigm: Standard (No Hyperparameter Tuning)
- Best Model: {std_best_model}
- Best Technique: {std_best_technique}
- Best R² Score: {standard_best_r2:.4f}
- Mean R² Score: {standard_results['Test_R²'].mean():.4f}
- Total Models Trained: {len(standard_results)}
{f"- Key Hyperparameters: {', '.join([f'{k}={v}' for k, v in list(std_hyperparams.items())[:5]])}" if std_hyperparams else ""}

Tuned Training Results:
- Paradigm: Tuned (With Hyperparameter Optimization)
- Best Model: {tuned_best_model}
- Best Technique: {tuned_best_technique}
- Best R² Score: {tuned_best_r2:.4f}
- Mean R² Score: {tuned_results['Test_R²'].mean():.4f}
- Total Models Trained: {len(tuned_results)}
{f"- Key Hyperparameters: {', '.join([f'{k}={v}' for k, v in list(tuned_hyperparams.items())[:5]])}" if tuned_hyperparams else ""}

Comparison Summary:
- Winner: {best_overall.get('paradigm', 'Unknown')}
- Best Model Overall: {best_model_name}
- Best Technique Overall: {best_technique_name}
- Winner R² Score: {best_r2_val:.4f}
- Improvement: {comparison['improvements'].get('improvement_percent', 0):.2f}%

{'='*80}
PERFORMANCE BY ALGORITHM - STANDARD PARADIGM (Model Consistency Across Techniques)
{'='*80}
Use this section to assess which model is most consistent across different preprocessing techniques.
Lower Std Dev = higher consistency; Higher Std Dev = more variable performance across techniques.

{_build_algorithm_performance_section(standard_results)}

{'='*80}
PERFORMANCE BY ALGORITHM - TUNED PARADIGM (Model Consistency Across Techniques)
{'='*80}
Use this section to assess which model is most consistent across different preprocessing techniques.
Lower Std Dev = higher consistency; Higher Std Dev = more variable performance across techniques.

{_build_algorithm_performance_section(tuned_results)}

{'='*80}
ALL MODELS FROM BOTH PARADIGMS (Ranked by R²)
{'='*80}

Standard Paradigm Models:
{_build_all_models_section(standard_results, 'Standard')}

Tuned Paradigm Models:
{_build_all_models_section(tuned_results, 'Tuned')}
"""
                }
                
                # Initialize chat
                if 'chat_interface' not in st.session_state:
                    st.session_state.chat_interface = ChatInterface()
                
                chat = st.session_state.chat_interface
                
                if not chat.ai_available:
                    st.warning("⚠️ AI features are not available. Please install google-generativeai or openai and set API keys.")
                else:
                    # Render chat UI with comparison-specific key
                    StreamlitChatUI.render_chat_interface(chat, combined_context, "comparison_paradigm_chat")
            
            except Exception as e:
                st.warning(f"AI comparison features not available: {str(e)}")
                st.info("To enable AI features:\n1. Install: pip install google-generativeai openai\n2. Set API keys: GOOGLE_API_KEY or OPENAI_API_KEY")
    
    except Exception as e:
        st.error(f"Error rendering comparison: {e}")
        logger.error(f"Comparison rendering error: {e}", exc_info=True)
"""
Enhanced Model Inspection Module
================================
Provides detailed model information, performance metrics, and comprehensive analysis.

Features:
- Complete model metadata display
- Detailed hyperparameter inspection
- Multi-metric analysis with filters
- Performance comparison visualizations
- RPD and other advanced metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import logging

logger = logging.getLogger(__name__)


class DetailedModelInspection:
    """Comprehensive model inspection and display."""
    
    @staticmethod
    def display_model_overview(model_metadata: Dict, model=None):
        """
        Display complete model overview.
        
        Parameters
        ----------
        model_metadata : Dict
            Model metadata containing algorithm, technique, scores, etc.
        model : object
            The actual model object
        """
        st.markdown("### 📋 Model Overview")
        
        # Create three columns for basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algo = model_metadata.get('algorithm', 'Unknown')
            st.metric("Algorithm", algo if algo else 'Unknown')
        
        with col2:
            tech = model_metadata.get('technique', 'Unknown')
            st.metric("Preprocessing Technique", tech if tech else 'Unknown')
        
        with col3:
            training_type = model_metadata.get('training_type', 'Unknown')
            st.metric("Training Type", training_type if training_type else 'Unknown')
        
        st.markdown("---")
        
        # Model creation details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            timestamp = model_metadata.get('timestamp', 'N/A')
            st.metric("Created", timestamp if timestamp != 'N/A' else 'Unknown')
        
        with col2:
            n_features = model_metadata.get('n_features', 'N/A')
            if n_features != 'N/A':
                st.metric("Input Features", int(n_features))
            else:
                st.metric("Input Features", 'N/A')
        
        with col3:
            model_source = model_metadata.get('source', 'Unknown')
            st.metric("Source", model_source if model_source else 'Unknown')
    
    
    @staticmethod
    def display_performance_metrics(model_metadata: Dict):
        """
        Display comprehensive performance metrics.
        
        Parameters
        ----------
        model_metadata : Dict
            Model metadata with performance metrics
        """
        st.markdown("### 📊 Performance Metrics")
        
        # Extract metrics
        test_r2 = float(model_metadata.get('test_r2', 0)) if model_metadata.get('test_r2') else 0
        train_r2 = float(model_metadata.get('train_r2', 0)) if model_metadata.get('train_r2') else 0
        test_rmse = float(model_metadata.get('test_rmse', 0)) if model_metadata.get('test_rmse') else 0
        train_rmse = float(model_metadata.get('train_rmse', 0)) if model_metadata.get('train_rmse') else 0
        test_mae = float(model_metadata.get('test_mae', 0)) if model_metadata.get('test_mae') else 0
        train_mae = float(model_metadata.get('train_mae', 0)) if model_metadata.get('train_mae') else 0
        rpd = float(model_metadata.get('rpd', 0)) if model_metadata.get('rpd') else 0
        
        # Create metric cards in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Test R²",
                f"{test_r2:.4f}",
                delta=f"Train: {train_r2:.4f}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Test RMSE",
                f"{test_rmse:.4f}",
                delta=f"Train: {train_rmse:.4f}",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Test MAE",
                f"{test_mae:.4f}",
                delta=f"Train: {train_mae:.4f}",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "RPD",
                f"{rpd:.4f}",
                delta="Ratio of Prediction to Deviation"
            )
        
        st.markdown("---")
        
        # Detailed metrics table
        st.markdown("#### 📈 Detailed Metrics Breakdown")
        
        metrics_data = {
            'Metric': [
                'R² Score (Coefficient of Determination)',
                'RMSE (Root Mean Square Error)',
                'MAE (Mean Absolute Error)',
                'RPD (Ratio of Prediction to Deviation)'
            ],
            'Train': [
                f"{train_r2:.6f}",
                f"{train_rmse:.6f}",
                f"{train_mae:.6f}",
                "N/A (Test only)"
            ],
            'Test': [
                f"{test_r2:.6f}",
                f"{test_rmse:.6f}",
                f"{test_mae:.6f}",
                f"{rpd:.6f}"
            ],
            'Interpretation': [
                f"{'Excellent' if test_r2 > 0.9 else 'Good' if test_r2 > 0.7 else 'Fair' if test_r2 > 0.5 else 'Poor'}",
                "Lower is better",
                "Lower is better",
                f"{'Excellent' if rpd > 3 else 'Good' if rpd > 2 else 'Fair' if rpd > 1.5 else 'Poor'}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        
        # Interpretation guide
        with st.expander("📖 Metric Interpretations"):
            st.markdown("""
            **R² Score:**
            - > 0.9: Excellent predictive power
            - 0.7-0.9: Good fit
            - 0.5-0.7: Fair fit
            - < 0.5: Poor fit
            
            **RMSE (Root Mean Square Error):**
            - Penalizes larger errors more heavily
            - In same units as target variable
            - Lower values indicate better performance
            
            **MAE (Mean Absolute Error):**
            - Average absolute deviation
            - More interpretable than RMSE
            - Less sensitive to outliers than RMSE
            
            **RPD (Ratio of Prediction to Deviation):**
            - SD(actual) / RMSE
            - > 3.0: Excellent
            - 2.0-3.0: Good
            - 1.5-2.0: Fair
            - < 1.5: Poor
            """)
    
    
    @staticmethod
    def display_hyperparameters(model, model_metadata: Dict):
        """
        Display detailed hyperparameters.
        
        Parameters
        ----------
        model : object
            Trained model
        model_metadata : Dict
            Model metadata
        """
        st.markdown("### ⚙️ Hyperparameters")
        
        try:
            from model_analyzer import ParameterInspector
            
            # First, try to get hyperparameters from metadata (stored during training)
            hyperparams_from_metadata = model_metadata.get('hyperparameters', {})
            
            # Try to extract hyperparameters from the model
            hyperparams_from_model = ParameterInspector.get_hyperparameters(model)
            
            # Combine both sources (model takes precedence)
            hyperparams = {**hyperparams_from_metadata, **hyperparams_from_model}
            
            if hyperparams and len(hyperparams) > 0:
                # Format for display
                formatted_params = ParameterInspector.format_parameters_for_display(hyperparams)
                
                # Create a clean dataframe
                params_data = []
                for key, value in formatted_params.items():
                    original_value = hyperparams.get(key)
                    params_data.append({
                        'Parameter': key,
                        'Value': value,
                        'Type': type(original_value).__name__ if original_value else 'Unknown'
                    })
                
                params_df = pd.DataFrame(params_data)
                st.dataframe(params_df, width='stretch', hide_index=True)
                
                # Show raw hyperparameters in JSON format
                with st.expander("View Raw Hyperparameters (JSON)"):
                    st.json(hyperparams)
                
                # Show source info
                source_info = []
                if hyperparams_from_metadata:
                    source_info.append("✓ Loaded from metadata")
                if hyperparams_from_model:
                    source_info.append("✓ Extracted from model")
                
                if source_info:
                    st.caption(f"Sources: {', '.join(source_info)}")
            else:
                # Model may not support get_params() or hyperparams couldn't be extracted
                st.info("""
                ℹ️ **Model Information:**
                - This model type may not expose its hyperparameters via `get_params()`
                - The model has been trained and can still be used for predictions
                - Check the model's documentation for specific hyperparameter details
                """)
                
                # Try to show some basic model info
                if hasattr(model, '__class__'):
                    st.json({
                        'Model Type': str(model.__class__),
                        'Module': model.__class__.__module__,
                        'Name': model.__class__.__name__
                    })
        
        except Exception as e:
            logger.warning(f"Could not extract hyperparameters: {e}")
            st.warning(f"""
            ⚠️ **Could not extract hyperparameters**: {str(e)}
            
            **Solution:** 
            - Ensure the model was properly trained
            - Check that all required packages are installed
            - The model may still be used for predictions
            """)
            
            # Show whatever information we can
            if model_metadata:
                st.json({
                    "Available Metadata": model_metadata
                })


class MetricsAnalyzer:
    """Comprehensive metrics analysis and visualization."""
    
    @staticmethod
    def create_metrics_comparison_chart(results_df: pd.DataFrame) -> go.Figure:
        """
        Create comparison chart for multiple metrics.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        metrics = ['Test_R²', 'Test_RMSE', 'Test_MAE']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            return go.Figure()
        
        fig = go.Figure()
        
        for metric in available_metrics:
            fig.add_trace(go.Box(
                y=results_df[metric],
                name=metric.replace('_', ' '),
                boxmean='sd'
            ))
        
        fig.update_layout(
            title="Metrics Distribution Across All Models",
            yaxis_title="Value",
            boxmode='group',
            height=500
        )
        
        return fig
    
    
    @staticmethod
    def create_model_metrics_heatmap(results_df: pd.DataFrame) -> go.Figure:
        """
        Create heatmap of metrics by model and technique.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        metrics_to_use = ['Test_R²', 'Test_RMSE', 'Test_MAE']
        available_metrics = [m for m in metrics_to_use if m in results_df.columns]
        
        if not available_metrics:
            return go.Figure()
        
        # Create pivot table
        pivot_data = results_df.pivot_table(
            index='Model',
            columns='Technique',
            values='Test_R²',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            text=pivot_data.values,
            texttemplate='.3f',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="R² Score Heatmap: Model vs Technique",
            xaxis_title="Technique",
            yaxis_title="Model",
            height=500
        )
        
        return fig
    
    
    @staticmethod
    def create_scatter_matrix(results_df: pd.DataFrame) -> go.Figure:
        """
        Create scatter plot matrix for metrics.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        metrics = ['Test_R²', 'Test_RMSE', 'Test_MAE']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if len(available_metrics) < 2:
            return go.Figure()
        
        fig = px.scatter_matrix(
            results_df,
            dimensions=available_metrics,
            color='Model',
            hover_data=['Technique'],
            title="Metrics Correlation Matrix"
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        
        return fig
    
    
    @staticmethod
    def display_metric_filter_analysis(results_df: pd.DataFrame, key_prefix: str = "analytics"):
        """
        Display metrics with interactive filtering slider.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
        key_prefix : str
            Prefix for widget keys to ensure uniqueness
        """
        st.markdown("### 🔍 Metric-Based Performance Analysis")
        
        # Get available metrics
        available_metrics = [col for col in ['Test_R²', 'Test_RMSE', 'Test_MAE', 'RPD'] 
                            if col in results_df.columns]
        
        if not available_metrics:
            st.warning("No metrics available for analysis")
            return
        
        # Select metric to filter by
        col1, col2 = st.columns(2)
        
        with col1:
            selected_metric = st.selectbox(
                "Select metric to isolate",
                available_metrics,
                key=f"{key_prefix}_metric_filter_select"
            )
        
        # Get metric range
        metric_min = float(results_df[selected_metric].min())
        metric_max = float(results_df[selected_metric].max())
        metric_mean = float(results_df[selected_metric].mean())
        metric_std = float(results_df[selected_metric].std())
        
        with col2:
            st.metric(
                f"{selected_metric} Statistics",
                f"{metric_mean:.4f}",
                delta=f"Std: {metric_std:.4f}"
            )
        
        st.markdown("---")
        
        # Create slider based on metric type
        if selected_metric == 'Test_R²':
            lower_bound, upper_bound = st.slider(
                f"Filter {selected_metric} Range",
                min_value=metric_min,
                max_value=metric_max,
                value=(metric_min, metric_max),
                step=0.01,
                key=f"{key_prefix}_r2_range_slider"
            )
        elif selected_metric in ['Test_RMSE', 'Test_MAE']:
            lower_bound, upper_bound = st.slider(
                f"Filter {selected_metric} Range (Lower is Better)",
                min_value=metric_min,
                max_value=metric_max,
                value=(metric_min, metric_max),
                step=0.01,
                key=f"{key_prefix}_rmse_mae_range_slider"
            )
        else:
            lower_bound, upper_bound = st.slider(
                f"Filter {selected_metric} Range",
                min_value=metric_min,
                max_value=metric_max,
                value=(metric_min, metric_max),
                step=0.01,
                key=f"{key_prefix}_other_range_slider"
            )
        
        # Filter data
        filtered_df = results_df[
            (results_df[selected_metric] >= lower_bound) &
            (results_df[selected_metric] <= upper_bound)
        ]
        
        # Show statistics
        st.markdown(f"#### Filtered Results ({len(filtered_df)} of {len(results_df)} models)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Count", len(filtered_df))
        
        with col2:
            st.metric(
                "Mean",
                f"{filtered_df[selected_metric].mean():.4f}"
            )
        
        with col3:
            st.metric(
                "Std Dev",
                f"{filtered_df[selected_metric].std():.4f}"
            )
        
        with col4:
            st.metric(
                "Range",
                f"{filtered_df[selected_metric].min():.4f} - {filtered_df[selected_metric].max():.4f}"
            )
        
        # Display table
        st.markdown("#### Filtered Models")
        display_cols = ['Model', 'Technique', 'Test_R²', 'Test_RMSE', 'Test_MAE']
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        # Ensure selected_metric is in display columns for sorting
        if selected_metric not in display_cols and selected_metric in filtered_df.columns:
            display_cols.append(selected_metric)
        
        # Sort by selected_metric only if it's in the filtered_df AND in display_cols
        if selected_metric in filtered_df.columns and selected_metric in display_cols:
            sorted_df = filtered_df[display_cols].sort_values(selected_metric, ascending=False)
        else:
            sorted_df = filtered_df[display_cols]
        
        st.dataframe(sorted_df, width='stretch')
        
        # Visualization
        st.markdown("#### Visualization")
        
        try:
            # Use Test_R² for size only if it's positive
            size_col = None
            if 'Test_R²' in filtered_df.columns:
                r2_values = filtered_df['Test_R²']
                if (r2_values > 0).all():
                    size_col = 'Test_R²'
            
            fig = px.scatter(
                filtered_df,
                x='Model',
                y=selected_metric,
                color='Technique',
                size=size_col,
                title=f"Models Filtered by {selected_metric}",
                labels={selected_metric: f"{selected_metric} Value"}
            )
            
            st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_metric_scatter_{selected_metric}")
        except Exception as e:
            st.warning(f"Could not create scatter plot: {e}")
    
    
    @staticmethod
    def display_comprehensive_analysis(results_df: pd.DataFrame, key_prefix: str = "analytics"):
        """
        Display comprehensive multi-metric analysis.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
        key_prefix : str
            Prefix for widget keys to ensure uniqueness
        """
        st.markdown("### 📈 Comprehensive Metrics Analysis")
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribution Analysis")
            try:
                fig = MetricsAnalyzer.create_metrics_comparison_chart(results_df)
                st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_metrics_distribution_chart")
            except Exception as e:
                st.warning(f"Could not create distribution chart: {e}")
        
        with col2:
            st.markdown("#### Model-Technique Heatmap")
            try:
                fig = MetricsAnalyzer.create_model_metrics_heatmap(results_df)
                st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_model_technique_heatmap")
            except Exception as e:
                st.warning(f"Could not create heatmap: {e}")
        
        # Correlation matrix
        st.markdown("#### Metrics Correlation")
        try:
            fig = MetricsAnalyzer.create_scatter_matrix(results_df)
            st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_metrics_correlation_matrix")
        except Exception as e:
            st.warning(f"Could not create correlation matrix: {e}")
        
        # Statistical summary
        st.markdown("#### Statistical Summary")
        metrics = ['Test_R²', 'Test_RMSE', 'Test_MAE']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if available_metrics:
            summary_stats = results_df[available_metrics].describe()
            st.dataframe(summary_stats, width='stretch')


class PredictionModeInspection:
    """Comprehensive inspection for prediction mode."""
    
    @staticmethod
    def display_full_model_info(model, model_metadata: Dict):
        """
        Display complete model information for prediction mode.
        
        Parameters
        ----------
        model : object
            Trained model
        model_metadata : Dict
            Model metadata
        """
        st.markdown("# 🔍 Complete Model Inspection")
        
        # Create tabs for organization
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Overview",
            "📊 Performance",
            "⚙️ Hyperparameters",
            "📈 Detailed Analysis"
        ])
        
        with tab1:
            DetailedModelInspection.display_model_overview(model_metadata, model)
        
        with tab2:
            DetailedModelInspection.display_performance_metrics(model_metadata)
        
        with tab3:
            DetailedModelInspection.display_hyperparameters(model, model_metadata)
        
        with tab4:
            st.markdown("### 📈 Model Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Type")
                st.write(f"```python\n{type(model).__name__}\n```")
            
            with col2:
                st.markdown("#### Model Configuration")
                try:
                    if hasattr(model, 'get_params'):
                        params = model.get_params()
                        st.write(f"**Total Parameters:** {len(params)}")
                except:
                    st.write("Unable to retrieve parameter count")
            
            # Model capabilities
            st.markdown("#### Model Capabilities")
            capabilities = {
                'Has predict': hasattr(model, 'predict'),
                'Has predict_proba': hasattr(model, 'predict_proba'),
                'Has score': hasattr(model, 'score'),
                'Has feature_importances': hasattr(model, 'feature_importances_'),
                'Has coef_': hasattr(model, 'coef_'),
            }
            
            caps_df = pd.DataFrame([
                {'Capability': k.replace('_', ' ').title(), 'Available': v}
                for k, v in capabilities.items()
            ])
            
            st.dataframe(caps_df, width='stretch', hide_index=True)


class TrainingModeAnalysis:
    """Comprehensive analysis for training mode results."""
    
    @staticmethod
    def display_detailed_metrics_dashboard(results_df: pd.DataFrame, paradigm: str = "", key_prefix: str = "training"):
        """
        Display detailed metrics dashboard for training results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Training results
        paradigm : str
            Training paradigm name
        key_prefix : str
            Prefix for widget keys to ensure uniqueness
        """
        st.markdown(f"### 📊 Detailed Metrics Dashboard - {paradigm}")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Analysis",
            "🔍 Filters",
            "📉 Comparisons",
            "📋 Raw Data"
        ])
        
        with tab1:
            MetricsAnalyzer.display_comprehensive_analysis(results_df, key_prefix=f"{key_prefix}_analysis")
        
        with tab2:
            MetricsAnalyzer.display_metric_filter_analysis(results_df, key_prefix=f"{key_prefix}_filter")
        
        with tab3:
            st.markdown("#### Model Performance Comparison")
            
            # Get all unique models
            models = results_df['Model'].unique()
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_models = st.multiselect(
                    "Select models to compare",
                    models,
                    default=list(models[:3]) if len(models) > 3 else list(models),
                    key=f"{key_prefix}_model_comparison_multiselect"
                )
            
            with col2:
                metric_to_compare = st.selectbox(
                    "Metric to compare",
                    ['Test_R²', 'Test_RMSE', 'Test_MAE', 'RPD'],
                    key=f"{key_prefix}_metric_to_compare_select"
                )
                if metric_to_compare not in results_df.columns:
                    metric_to_compare = 'Test_R²'
            
            # Filter and visualize
            if selected_models:
                comparison_df = results_df[results_df['Model'].isin(selected_models)]
                
                fig = px.box(
                    comparison_df,
                    x='Model',
                    y=metric_to_compare,
                    color='Technique',
                    title=f"{metric_to_compare} Distribution by Model and Technique",
                    points="all"
                )
                
                st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_model_metric_comparison_box")
        
        with tab4:
            st.markdown("#### Full Results Table")
            
            display_cols = [col for col in ['Model', 'Technique', 'Test_R²', 'Train_R²', 
                                           'Test_RMSE', 'Train_RMSE', 'Test_MAE', 'Train_MAE', 'RPD']
                           if col in results_df.columns]
            
            st.dataframe(
                results_df[display_cols].sort_values('Test_R²', ascending=False),
                width='stretch'
            )
