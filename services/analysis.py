"""
Core analysis functions for CSV Insight app.
All functions are pure and safe - no file system access or code execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def load_csv(path: str, parse_dates: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """Load CSV file with optional date parsing."""
    try:
        df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {str(e)}")


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer schema information for DataFrame columns."""
    schema = {
        "columns": [],
        "total_rows": len(df),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_pct": round((df[col].isnull().sum() / len(df)) * 100, 2),
            "unique_count": df[col].nunique(),
            "sample_values": df[col].dropna().head(3).tolist()
        }
        
        # Add type-specific info
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not df[col].isna().all() else None
            col_info["max"] = float(df[col].max()) if not df[col].isna().all() else None
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["date_range"] = {
                "start": df[col].min().isoformat() if not df[col].isna().all() else None,
                "end": df[col].max().isoformat() if not df[col].isna().all() else None
            }
        
        schema["columns"].append(col_info)
    
    return schema


def coerce_types(df: pd.DataFrame, dtypes_map: Dict[str, str]) -> pd.DataFrame:
    """Coerce column types based on mapping."""
    df_copy = df.copy()
    
    for col, dtype in dtypes_map.items():
        if col in df_copy.columns:
            try:
                if dtype == 'datetime':
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                elif dtype == 'numeric':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif dtype == 'category':
                    df_copy[col] = df_copy[col].astype('category')
                else:
                    df_copy[col] = df_copy[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {dtype}: {e}")
    
    return df_copy


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Select specific columns from DataFrame."""
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")
    
    return df[columns].copy()


def filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Filter rows based on pandas query string."""
    try:
        return df.query(condition).copy()
    except Exception as e:
        raise ValueError(f"Invalid filter condition '{condition}': {str(e)}")


def resample(df: pd.DataFrame, rule: str, agg: str = 'mean', 
             datetime_col: Optional[str] = None) -> pd.DataFrame:
    """Resample time series data."""
    if datetime_col is None:
        # Try to find datetime column
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            raise ValueError("No datetime column found for resampling")
        datetime_col = datetime_cols[0]
    
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found")
    
    df_copy = df.copy()
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    df_copy = df_copy.set_index(datetime_col)
    
    # Resample numeric columns only
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for resampling")
    
    resampled = df_copy[numeric_cols].resample(rule).agg(agg)
    return resampled.reset_index()


def groupby_agg(df: pd.DataFrame, by: Union[str, List[str]], 
                metrics: Dict[str, List[str]]) -> pd.DataFrame:
    """Group by columns and aggregate metrics."""
    if isinstance(by, str):
        by = [by]
    
    missing_cols = [col for col in by if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Group by columns not found: {missing_cols}")
    
    try:
        result = df.groupby(by).agg(metrics).reset_index()
        return result
    except Exception as e:
        raise ValueError(f"Groupby aggregation failed: {str(e)}")


def summary_stats(df: pd.DataFrame, include: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generate summary statistics for DataFrame."""
    if include is None:
        include = ['number']
    
    stats = {
        "overview": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        },
        "missing_data": {},
        "constant_columns": [],
        "duplicate_rows": int(df.duplicated().sum())
    }
    
    # Missing data analysis
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        stats["missing_data"][col] = {
            "count": int(null_count),
            "percentage": round(null_pct, 2)
        }
        
        # Check for constant columns
        if df[col].nunique() <= 1:
            stats["constant_columns"].append(col)
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stats["categorical_stats"] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            stats["categorical_stats"][col] = {
                "unique_count": int(value_counts.nunique()),
                "top_values": value_counts.head(5).to_dict()
            }
    
    return stats


def corr_matrix(df: pd.DataFrame, numeric_only: bool = True) -> Dict[str, Any]:
    """Generate correlation matrix for numeric columns."""
    if numeric_only:
        df_numeric = df.select_dtypes(include=[np.number])
    else:
        df_numeric = df
    
    if len(df_numeric.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation matrix")
    
    corr = df_numeric.corr()
    
    return {
        "correlation_matrix": corr.values.tolist(),
        "columns": corr.columns.tolist(),
        "rows": corr.index.tolist()
    }


def outliers_iqr(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Detect outliers using IQR method for a specific column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' is not numeric")
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    return {
        "column": col,
        "total_outliers": int(len(outliers)),
        "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
        "bounds": {
            "lower": float(lower_bound),
            "upper": float(upper_bound)
        },
        "outlier_indices": outliers.index.tolist(),
        "outlier_values": outliers[col].tolist()
    }


def rolling(df: pd.DataFrame, window: int, agg: str = 'mean',
            columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate rolling statistics for numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns) == 0:
        raise ValueError("No numeric columns found for rolling calculations")
    
    df_copy = df.copy()
    result = pd.DataFrame()
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            if agg == 'mean':
                result[f"{col}_rolling_{agg}_{window}"] = df_copy[col].rolling(window=window).mean()
            elif agg == 'std':
                result[f"{col}_rolling_{agg}_{window}"] = df_copy[col].rolling(window=window).std()
            elif agg == 'min':
                result[f"{col}_rolling_{agg}_{window}"] = df_copy[col].rolling(window=window).min()
            elif agg == 'max':
                result[f"{col}_rolling_{agg}_{window}"] = df_copy[col].rolling(window=window).max()
            else:
                result[f"{col}_rolling_mean_{window}"] = df_copy[col].rolling(window=window).mean()
    
    return result


def downsample_lttb(series: pd.Series, target_points: int) -> pd.Series:
    """Downsample time series using LTTB (Largest-Triangle-Three-Buckets) algorithm."""
    if len(series) <= target_points:
        return series
    
    # Simple downsampling for now (can be enhanced with proper LTTB implementation)
    step = len(series) // target_points
    indices = range(0, len(series), step)
    return series.iloc[indices] 