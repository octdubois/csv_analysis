"""
Chart generation functions for CSV Insight app.
All functions return Plotly figures for safe, offline charting.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def line_ts(df: pd.DataFrame, x: str, y: Union[str, List[str]], 
            resample: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """Create time series line plot."""
    if x not in df.columns:
        raise ValueError(f"X column '{x}' not found")
    
    if isinstance(y, str):
        y = [y]
    
    missing_cols = [col for col in y if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Y columns not found: {missing_cols}")
    
    # Ensure datetime column is properly formatted
    df_copy = df.copy()
    df_copy[x] = pd.to_datetime(df_copy[x])
    
    # Resample if requested
    if resample:
        df_copy = df_copy.set_index(x)
        numeric_cols = df_copy[y].select_dtypes(include=[np.number])
        if len(numeric_cols.columns) > 0:
            df_copy = df_copy[numeric_cols.columns].resample(resample).mean()
        df_copy = df_copy.reset_index()
    
    # Create figure
    fig = go.Figure()
    
    for col in y:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            fig.add_trace(go.Scatter(
                x=df_copy[x],
                y=df_copy[col],
                mode='lines',
                name=col,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title=title or f"Time Series: {', '.join(y)} vs {x}",
        xaxis_title=x,
        yaxis_title="Value",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def multi_line_overlay(df: pd.DataFrame, x: str, y_columns: List[str], 
                      title: Optional[str] = None) -> go.Figure:
    """Create multi-line overlay plot."""
    return line_ts(df, x, y_columns, title=title)


def histogram(df: pd.DataFrame, column: str, bins: Optional[int] = None,
              title: Optional[str] = None) -> go.Figure:
    """Create histogram for a numeric column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[column].dropna(),
        nbinsx=bins,
        name=column,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=title or f"Histogram: {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def boxplot(df: pd.DataFrame, columns: Optional[List[str]] = None,
            title: Optional[str] = None) -> go.Figure:
    """Create boxplot for numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns) == 0:
        raise ValueError("No numeric columns found for boxplot")
    
    # Prepare data for boxplot
    box_data = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            box_data.append(go.Box(
                y=df[col].dropna(),
                name=col,
                boxpoints='outliers'
            ))
    
    fig = go.Figure(data=box_data)
    
    fig.update_layout(
        title=title or "Boxplot: Numeric Columns",
        yaxis_title="Value",
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
            title: Optional[str] = None) -> go.Figure:
    """Create scatter plot."""
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"X or Y column not found")
    
    if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
        raise ValueError(f"X and Y columns must be numeric")
    
    # Prepare color mapping
    color_values = None
    if color and color in df.columns:
        color_values = df[color]
    
    fig = go.Figure()
    
    if color_values is not None:
        # Create scatter with color mapping
        unique_colors = color_values.unique()
        for color_val in unique_colors:
            mask = color_values == color_val
            fig.add_trace(go.Scatter(
                x=df[mask][x],
                y=df[mask][y],
                mode='markers',
                name=str(color_val),
                marker=dict(size=6)
            ))
    else:
        # Simple scatter
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[y],
            mode='markers',
            marker=dict(size=6, color='lightblue')
        ))
    
    fig.update_layout(
        title=title or f"Scatter Plot: {y} vs {x}",
        xaxis_title=x,
        yaxis_title=y,
        template="plotly_dark"
    )
    
    return fig


def corr_heatmap(df: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    """Create correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number])
    
    if len(numeric_cols.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation heatmap")
    
    corr_matrix = numeric_cols.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title or "Correlation Heatmap",
        template="plotly_dark",
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    
    return fig


def bar_agg(df: pd.DataFrame, x: str, y: str, agg: str = 'count',
             title: Optional[str] = None) -> go.Figure:
    """Create bar chart with aggregation."""
    if x not in df.columns:
        raise ValueError(f"X column '{x}' not found")
    
    if y and y not in df.columns:
        raise ValueError(f"Y column '{y}' not found")
    
    # Prepare data
    if y and pd.api.types.is_numeric_dtype(df[y]):
        # Group by x and aggregate y
        if agg == 'count':
            grouped = df.groupby(x)[y].count()
        elif agg == 'sum':
            grouped = df.groupby(x)[y].sum()
        elif agg == 'mean':
            grouped = df.groupby(x)[y].mean()
        elif agg == 'median':
            grouped = df.groupby(x)[y].median()
        else:
            grouped = df.groupby(x)[y].mean()
        
        x_values = grouped.index.tolist()
        y_values = grouped.values.tolist()
    else:
        # Simple count of x values
        value_counts = df[x].value_counts()
        x_values = value_counts.index.tolist()
        y_values = value_counts.values.tolist()
    
    # Limit to top 20 for readability
    if len(x_values) > 20:
        x_values = x_values[:20]
        y_values = y_values[:20]
    
    fig = go.Figure(data=go.Bar(
        x=x_values,
        y=y_values,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=title or f"Bar Chart: {x}",
        xaxis_title=x,
        yaxis_title=f"{agg.title() if y else 'Count'}",
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def save_plot(fig: go.Figure, filename: str, output_dir: str = "./outputs/plots") -> str:
    """Save plot to file and return the filepath."""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Save as HTML (interactive)
    fig.write_html(filepath + ".html")
    
    # Save as PNG (static)
    fig.write_image(filepath + ".png", width=800, height=600)
    
    return filepath 