"""
Unit tests for analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

# Import functions to test
from services.analysis import (
    load_csv, infer_schema, coerce_types, select_columns, filter_rows,
    resample, groupby_agg, summary_stats, corr_matrix, outliers_iqr,
    rolling, downsample_lttb
)


class TestAnalysisFunctions:
    """Test cases for analysis functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample DataFrame
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'temperature': np.random.normal(20, 5, 100),
            'humidity': np.random.normal(60, 10, 100),
            'pressure': np.random.normal(1013, 5, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
    
    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_load_csv(self):
        """Test CSV loading functionality."""
        df = load_csv(self.temp_csv.name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert len(df.columns) == 5
    
    def test_infer_schema(self):
        """Test schema inference."""
        schema = infer_schema(self.df)
        
        assert 'columns' in schema
        assert 'total_rows' in schema
        assert 'memory_usage_mb' in schema
        
        assert schema['total_rows'] == 100
        assert len(schema['columns']) == 5
        
        # Check column info
        temp_col = next(col for col in schema['columns'] if col['name'] == 'temperature')
        assert temp_col['dtype'] == 'float64'
        assert temp_col['null_pct'] == 0.0
    
    def test_coerce_types(self):
        """Test type coercion."""
        dtypes_map = {
            'temperature': 'int32',
            'category': 'category'
        }
        
        df_coerced = coerce_types(self.df, dtypes_map)
        
        assert df_coerced['temperature'].dtype == 'int32'
        assert df_coerced['category'].dtype == 'category'
    
    def test_select_columns(self):
        """Test column selection."""
        selected = select_columns(self.df, ['timestamp', 'temperature'])
        
        assert len(selected.columns) == 2
        assert 'timestamp' in selected.columns
        assert 'temperature' in selected.columns
    
    def test_select_columns_missing(self):
        """Test column selection with missing columns."""
        with pytest.raises(ValueError, match="Columns not found"):
            select_columns(self.df, ['timestamp', 'nonexistent'])
    
    def test_filter_rows(self):
        """Test row filtering."""
        filtered = filter_rows(self.df, 'temperature > 20')
        
        assert len(filtered) < len(self.df)
        assert all(filtered['temperature'] > 20)
    
    def test_filter_rows_invalid(self):
        """Test row filtering with invalid condition."""
        with pytest.raises(ValueError):
            filter_rows(self.df, 'invalid condition')
    
    def test_resample(self):
        """Test time series resampling."""
        resampled = resample(self.df, rule='2H', agg='mean')
        
        assert isinstance(resampled, pd.DataFrame)
        assert len(resampled) < len(self.df)  # Should have fewer rows after resampling
    
    def test_groupby_agg(self):
        """Test groupby aggregation."""
        result = groupby_agg(
            self.df, 
            by='category', 
            metrics={'temperature': ['mean', 'std'], 'humidity': ['mean']}
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'category' in result.columns
    
    def test_summary_stats(self):
        """Test summary statistics generation."""
        stats = summary_stats(self.df)
        
        assert 'overview' in stats
        assert 'missing_data' in stats
        assert 'constant_columns' in stats
        assert 'numeric_stats' in stats
        
        assert stats['overview']['total_rows'] == 100
        assert stats['overview']['total_columns'] == 5
    
    def test_corr_matrix(self):
        """Test correlation matrix generation."""
        corr = corr_matrix(self.df)
        
        assert 'correlation_matrix' in corr
        assert 'columns' in corr
        assert 'rows' in corr
        
        # Should have correlation matrix for numeric columns only
        assert len(corr['columns']) == 3  # temperature, humidity, pressure
    
    def test_corr_matrix_insufficient_columns(self):
        """Test correlation matrix with insufficient columns."""
        df_small = self.df[['timestamp', 'temperature']]
        with pytest.raises(ValueError, match="Need at least 2 numeric columns"):
            corr_matrix(df_small)
    
    def test_outliers_iqr(self):
        """Test outlier detection."""
        outliers = outliers_iqr(self.df, 'temperature')
        
        assert 'column' in outliers
        assert 'total_outliers' in outliers
        assert 'outlier_percentage' in outliers
        assert 'bounds' in outliers
        
        assert outliers['column'] == 'temperature'
        assert outliers['total_outliers'] >= 0
    
    def test_outliers_iqr_invalid_column(self):
        """Test outlier detection with invalid column."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            outliers_iqr(self.df, 'nonexistent')
    
    def test_outliers_iqr_non_numeric(self):
        """Test outlier detection with non-numeric column."""
        with pytest.raises(ValueError, match="Column 'category' is not numeric"):
            outliers_iqr(self.df, 'category')
    
    def test_rolling(self):
        """Test rolling statistics."""
        rolling_result = rolling(self.df, window=5, agg='mean')
        
        assert isinstance(rolling_result, pd.DataFrame)
        assert len(rolling_result) == len(self.df)
    
    def test_downsample_lttb(self):
        """Test LTTB downsampling."""
        series = pd.Series(np.random.random(1000))
        downsampled = downsample_lttb(series, 100)
        
        assert len(downsampled) <= 100
        assert isinstance(downsampled, pd.Series)
    
    def test_downsample_lttb_small_series(self):
        """Test LTTB downsampling with small series."""
        series = pd.Series(np.random.random(50))
        downsampled = downsample_lttb(series, 100)
        
        # Should return original series if target is larger
        assert len(downsampled) == 50


class TestAnalysisEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test functions with empty DataFrame."""
        df_empty = pd.DataFrame()
        
        # Test schema inference
        schema = infer_schema(df_empty)
        assert schema['total_rows'] == 0
        assert len(schema['columns']) == 0
        
        # Test summary stats
        stats = summary_stats(df_empty)
        assert stats['overview']['total_rows'] == 0
    
    def test_dataframe_with_all_nulls(self):
        """Test functions with DataFrame containing all nulls."""
        df_nulls = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        schema = infer_schema(df_nulls)
        assert schema['columns'][0]['null_pct'] == 100.0
        
        stats = summary_stats(df_nulls)
        assert len(stats['missing_data']) == 2
    
    def test_single_column_dataframe(self):
        """Test functions with single column DataFrame."""
        df_single = pd.DataFrame({'col': [1, 2, 3]})
        
        schema = infer_schema(df_single)
        assert len(schema['columns']) == 1
        
        # Correlation matrix should fail
        with pytest.raises(ValueError):
            corr_matrix(df_single)


if __name__ == "__main__":
    pytest.main([__file__]) 