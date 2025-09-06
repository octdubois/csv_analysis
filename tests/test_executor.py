"""
Unit tests for the plan executor.
"""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import functions to test
from services.executor import (
    PlanExecutor, Plan, Step, ExecutionResult, 
    validate_plan_json, create_sample_plan
)


class TestPlanValidation:
    """Test plan validation functionality."""
    
    def test_valid_plan(self):
        """Test that a valid plan passes validation."""
        valid_plan = {
            "steps": [
                {
                    "fn": "infer_schema",
                    "args": {},
                    "description": "Analyze data structure"
                }
            ],
            "narrative": "Basic analysis"
        }
        
        executor = PlanExecutor()
        plan = executor.validate_plan(valid_plan)
        
        assert isinstance(plan, Plan)
        assert len(plan.steps) == 1
        assert plan.steps[0].fn == "infer_schema"
    
    def test_invalid_function(self):
        """Test that plans with invalid functions are rejected."""
        invalid_plan = {
            "steps": [
                {
                    "fn": "invalid_function",
                    "args": {},
                    "description": "This should fail"
                }
            ],
            "narrative": "Invalid plan"
        }
        
        executor = PlanExecutor()
        with pytest.raises(ValueError, match="Plan validation failed"):
            executor.validate_plan(invalid_plan)
    
    def test_empty_steps(self):
        """Test that plans with no steps are rejected."""
        empty_plan = {
            "steps": [],
            "narrative": "No steps"
        }
        
        executor = PlanExecutor()
        with pytest.raises(ValueError, match="Plan must contain at least one step"):
            executor.validate_plan(empty_plan)
    
    def test_too_many_steps(self):
        """Test that plans with too many steps are rejected."""
        many_steps = [{"fn": "infer_schema", "args": {}} for _ in range(25)]
        large_plan = {
            "steps": many_steps,
            "narrative": "Too many steps"
        }
        
        executor = PlanExecutor()
        with pytest.raises(ValueError, match="Plan cannot exceed 20 steps"):
            executor.validate_plan(large_plan)
    
    def test_invalid_json(self):
        """Test that invalid JSON is rejected."""
        invalid_json = "{ invalid json }"
        
        executor = PlanExecutor()
        with pytest.raises(ValueError, match="Invalid JSON format"):
            executor.validate_plan(invalid_json)


class TestPlanExecution:
    """Test plan execution functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.executor = PlanExecutor()
        
        # Create sample DataFrame
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'temperature': np.random.normal(20, 5, 100),
            'humidity': np.random.normal(60, 10, 100),
            'pressure': np.random.normal(1013, 5, 100)
        })
    
    def test_simple_plan_execution(self):
        """Test execution of a simple plan."""
        plan_json = {
            "steps": [
                {
                    "fn": "infer_schema",
                    "args": {},
                    "description": "Analyze data structure"
                },
                {
                    "fn": "summary_stats",
                    "args": {},
                    "description": "Generate summary statistics"
                }
            ],
            "narrative": "Basic analysis"
        }
        
        result = self.executor.execute_plan(plan_json, self.df)
        
        assert result.success
        assert result.step_count == 2
        assert len(result.results) == 2
        assert len(result.errors) == 0
    
    def test_plan_with_errors(self):
        """Test execution of a plan with some errors."""
        plan_json = {
            "steps": [
                {
                    "fn": "infer_schema",
                    "args": {},
                    "description": "This should work"
                },
                {
                    "fn": "outliers_iqr",
                    "args": {"col": "nonexistent"},
                    "description": "This should fail"
                }
            ],
            "narrative": "Plan with errors"
        }
        
        result = self.executor.execute_plan(plan_json, self.df)
        
        # Should continue execution despite errors
        assert result.step_count == 1
        assert len(result.errors) == 1
        assert "nonexistent" in result.errors[0]
    
    def test_critical_error_stops_execution(self):
        """Test that critical errors stop execution."""
        plan_json = {
            "steps": [
                {
                    "fn": "load_csv",
                    "args": {"path": "nonexistent.csv"},
                    "description": "This should fail and stop execution"
                },
                {
                    "fn": "infer_schema",
                    "args": {},
                    "description": "This should not run"
                }
            ],
            "narrative": "Plan with critical error"
        }
        
        result = self.executor.execute_plan(plan_json, self.df)
        
        assert not result.success
        assert result.step_count == 0
        assert len(result.errors) > 0
    
    def test_context_updates(self):
        """Test that execution context is properly updated."""
        plan_json = {
            "steps": [
                {
                    "fn": "infer_schema",
                    "args": {},
                    "description": "Analyze data structure"
                },
                {
                    "fn": "select_columns",
                    "args": {"columns": ["timestamp", "temperature"]},
                    "description": "Select specific columns"
                }
            ],
            "narrative": "Test context updates"
        }
        
        result = self.executor.execute_plan(plan_json, self.df)
        
        assert result.success
        assert result.step_count == 2
        
        # Check that context was updated
        context = self.executor.get_context_summary()
        assert context["current_df_shape"] == (100, 2)  # After column selection
        assert "timestamp" in context["current_df_columns"]
        assert "temperature" in context["current_df_columns"]


class TestFunctionRegistry:
    """Test the function registry and execution."""
    
    def test_function_registry_contains_all_functions(self):
        """Test that all expected functions are in the registry."""
        executor = PlanExecutor()
        expected_functions = [
            "load_csv", "infer_schema", "coerce_types", "select_columns",
            "filter_rows", "resample", "groupby_agg", "summary_stats",
            "corr_matrix", "outliers_iqr", "rolling", "line_ts",
            "multi_line_overlay", "histogram", "boxplot", "scatter",
            "corr_heatmap", "bar_agg", "downsample_lttb"
        ]
        
        for func_name in expected_functions:
            assert func_name in executor.function_registry
            assert callable(executor.function_registry[func_name])
    
    def test_function_execution_with_args(self):
        """Test that functions are executed with proper arguments."""
        executor = PlanExecutor()
        
        # Test a function that requires specific arguments
        plan_json = {
            "steps": [
                {
                    "fn": "select_columns",
                    "args": {"columns": ["timestamp", "temperature"]},
                    "description": "Select columns"
                }
            ],
            "narrative": "Test function execution"
        }
        
        result = executor.execute_plan(plan_json, self.df)
        
        assert result.success
        assert result.step_count == 1
        
        # Check that the function was called with correct arguments
        step_result = result.results[0]
        assert "select_columns" in step_result["step_name"]


class TestExecutionResult:
    """Test the ExecutionResult class."""
    
    def test_execution_result_initialization(self):
        """Test ExecutionResult initialization."""
        result = ExecutionResult()
        
        assert not result.success
        assert len(result.results) == 0
        assert len(result.errors) == 0
        assert result.execution_time == 0.0
        assert result.step_count == 0
    
    def test_adding_results(self):
        """Test adding results to ExecutionResult."""
        result = ExecutionResult()
        step = Step(fn="infer_schema", args={})
        
        result.add_result(step, {"test": "data"}, "Test Step")
        
        assert len(result.results) == 1
        assert result.step_count == 1
        assert result.results[0]["step_name"] == "Test Step"
    
    def test_adding_errors(self):
        """Test adding errors to ExecutionResult."""
        result = ExecutionResult()
        step = Step(fn="test_function", args={})
        
        result.add_error(step, "Test error message")
        
        assert len(result.errors) == 1
        assert "test_function" in result.errors[0]
    
    def test_to_dict_conversion(self):
        """Test converting ExecutionResult to dictionary."""
        result = ExecutionResult()
        step = Step(fn="infer_schema", args={})
        
        result.add_result(step, {"test": "data"})
        result.success = True
        result.execution_time = 1.5
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["step_count"] == 1
        assert result_dict["execution_time"] == 1.5
        assert "infer_schema" in str(result_dict["results"])


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sample_plan(self):
        """Test sample plan creation."""
        sample_plan = create_sample_plan()
        
        assert "steps" in sample_plan
        assert "narrative" in sample_plan
        assert len(sample_plan["steps"]) > 0
        
        # Validate the sample plan
        executor = PlanExecutor()
        plan = executor.validate_plan(sample_plan)
        assert isinstance(plan, Plan)
    
    def test_validate_plan_json(self):
        """Test plan JSON validation."""
        valid_plan = {
            "steps": [
                {"fn": "infer_schema", "args": {}}
            ],
            "narrative": "Test"
        }
        
        assert validate_plan_json(json.dumps(valid_plan))
        
        invalid_plan = "invalid json"
        assert not validate_plan_json(invalid_plan)


if __name__ == "__main__":
    pytest.main([__file__]) 