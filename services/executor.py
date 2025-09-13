"""
Plan executor for CSV Insight app.
Safely executes LLM-generated plans using whitelisted functions.
"""

import pandas as pd
import json
import logging
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, ValidationError
import traceback

# Import our safe functions
from . import analysis
from . import plots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define allowed function types
AllowedFn = Literal[
    "load_csv", "infer_schema", "coerce_types", "select_columns", "filter_rows",
    "resample", "groupby_agg", "summary_stats", "corr_matrix", "outliers_iqr",
    "rolling", "line_ts", "multi_line_overlay", "histogram", "boxplot",
    "scatter", "corr_heatmap", "bar_agg", "downsample_lttb"
]


class Step(BaseModel):
    """Single execution step in a plan."""
    fn: AllowedFn
    args: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class Plan(BaseModel):
    """Complete execution plan from LLM."""
    steps: List[Step]
    narrative: str = ""
    metadata: Optional[Dict[str, Any]] = None


class ExecutionResult:
    """Result of plan execution."""
    
    def __init__(self):
        self.success: bool = False
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.plots: Dict[str, Any] = {}
        self.execution_time: float = 0.0
        self.step_count: int = 0
    
    def add_result(self, step: Step, result: Any, step_name: str = None):
        """Add a successful step result."""
        self.results.append({
            "step": step.dict(),
            "result": result,
            "step_name": step_name or step.fn
        })
        self.step_count += 1
    
    def add_error(self, step: Step, error: str):
        """Add an error from a failed step."""
        self.errors.append(f"Step {step.fn}: {error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "success": self.success,
            "step_count": self.step_count,
            "errors": self.errors,
            "results": self.results,
            "execution_time": self.execution_time
        }


class PlanExecutor:
    """Safely executes LLM-generated plans."""
    
    def __init__(self):
        # Function registry mapping function names to safe callables
        self.function_registry = {
            # Data operations
            "load_csv": analysis.load_csv,
            "infer_schema": analysis.infer_schema,
            "coerce_types": analysis.coerce_types,
            "select_columns": analysis.select_columns,
            "filter_rows": analysis.filter_rows,
            "resample": analysis.resample,
            "groupby_agg": analysis.groupby_agg,
            
            # Statistics
            "summary_stats": analysis.summary_stats,
            "corr_matrix": analysis.corr_matrix,
            "outliers_iqr": analysis.outliers_iqr,
            "rolling": analysis.rolling,
            
            # Plotting
            "line_ts": plots.line_ts,
            "multi_line_overlay": plots.multi_line_overlay,
            "histogram": plots.histogram,
            "boxplot": plots.boxplot,
            "scatter": plots.scatter,
            "corr_heatmap": plots.corr_heatmap,
            "bar_agg": plots.bar_agg,
            
            # Utilities
            "downsample_lttb": analysis.downsample_lttb,
        }
        
        # Current execution context
        self.current_df: Optional[pd.DataFrame] = None
        self.context_vars: Dict[str, Any] = {}
    
    def validate_plan(self, plan_json: str) -> Plan:
        """Validate and parse LLM plan JSON."""
        try:
            # Try to parse as JSON first
            if isinstance(plan_json, str):
                plan_data = json.loads(plan_json)
            else:
                plan_data = plan_json
            
            # Validate against Pydantic model
            plan = Plan(**plan_data)
            
            # Additional validation
            if not plan.steps:
                raise ValueError("Plan must contain at least one step")
            
            if len(plan.steps) > 20:
                raise ValueError("Plan cannot exceed 20 steps")
            
            return plan
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except ValidationError as e:
            raise ValueError(f"Plan validation failed: {e}")
        except Exception as e:
            raise ValueError(f"Plan parsing failed: {e}")
    
    def execute_plan(self, plan_json: str, initial_df: Optional[pd.DataFrame] = None) -> ExecutionResult:
        """Execute a validated plan and return results."""
        import time
        
        result = ExecutionResult()
        start_time = time.time()
        
        try:
            # Validate plan
            plan = self.validate_plan(plan_json)
            
            # Set initial dataframe if provided
            if initial_df is not None:
                self.current_df = initial_df.copy()
                self.context_vars["initial_df"] = self.current_df
            
            # Execute each step
            for i, step in enumerate(plan.steps):
                try:
                    step_result = self._execute_step(step, i)
                    result.add_result(step, step_result, f"Step {i+1}")
                    
                except Exception as e:
                    error_msg = f"Step {i+1} failed: {str(e)}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    result.add_error(step, error_msg)
                    
                    # Continue with next step unless it's a critical data operation
                    if step.fn in ["load_csv"]:
                        result.success = False
                        break
            
            # Mark as successful if no critical errors
            if not result.errors or len(result.errors) < len(plan.steps):
                result.success = True
            
        except Exception as e:
            error_msg = f"Plan execution failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            result.errors.append(error_msg)
            result.success = False
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def _execute_step(self, step: Step, step_index: int) -> Any:
        """Execute a single step safely."""
        if step.fn not in self.function_registry:
            raise ValueError(f"Unknown function: {step.fn}")
        
        func = self.function_registry[step.fn]
        
        # Prepare arguments
        args = step.args.copy()
        
        # Handle special cases for data operations
        if step.fn in ["load_csv"]:
            # If we already have a dataframe loaded, skip load_csv step
            if self.current_df is not None:
                logger.info("Skipping load_csv step - dataframe already loaded")
                return self.current_df
            # Otherwise, load_csv needs file path
            pass
        elif step.fn in ["infer_schema", "summary_stats", "corr_matrix", "boxplot", "corr_heatmap"]:
            # These functions need the current dataframe
            if self.current_df is None:
                raise ValueError(f"Step {step.fn} requires a loaded dataframe")
            args["df"] = self.current_df
        elif step.fn in ["line_ts", "multi_line_overlay", "histogram", "scatter", "bar_agg"]:
            # These functions need the current dataframe
            if self.current_df is None:
                raise ValueError(f"Step {step.fn} requires a loaded dataframe")
            args["df"] = self.current_df
        elif step.fn in ["select_columns", "filter_rows", "resample", "groupby_agg", 
                         "coerce_types", "rolling", "outliers_iqr"]:
            # These functions need the current dataframe and update it
            if self.current_df is None:
                raise ValueError(f"Step {step.fn} requires a loaded dataframe")
            args["df"] = self.current_df
        
        # Execute function
        try:
            step_result = func(**args)
            
            # Update context based on function type
            self._update_context(step, step_result)
            
            return step_result
            
        except Exception as e:
            raise ValueError(f"Function {step.fn} execution failed: {str(e)}")
    
    def _update_context(self, step: Step, result: Any):
        """Update execution context after step execution."""
        if step.fn == "load_csv":
            # Update current dataframe
            self.current_df = result
            self.context_vars["current_df"] = self.current_df
            self.context_vars["last_loaded_df"] = self.current_df
            
        elif step.fn in ["select_columns", "filter_rows", "resample", "groupby_agg", 
                         "coerce_types", "rolling"]:
            # Update current dataframe if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                self.current_df = result
                self.context_vars["current_df"] = self.current_df
                self.context_vars[f"step_{step.fn}_result"] = result
        
        # Store result in context
        self.context_vars[f"step_{step.fn}"] = result
    
    def get_current_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the current dataframe in context."""
        return self.current_df
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution context."""
        context_summary = {
            "current_df_shape": None,
            "current_df_columns": [],
            "context_vars": list(self.context_vars.keys())
        }
        
        if self.current_df is not None:
            context_summary["current_df_shape"] = self.current_df.shape
            context_summary["current_df_columns"] = self.current_df.columns.tolist()
        
        return context_summary


def create_sample_plan() -> Dict[str, Any]:
    """Create a sample plan for demonstration."""
    return {
        "steps": [
            {
                "fn": "infer_schema",
                "args": {},
                "description": "Analyze the dataset structure"
            },
            {
                "fn": "summary_stats",
                "args": {},
                "description": "Generate summary statistics"
            },
            {
                "fn": "corr_heatmap",
                "args": {},
                "description": "Create correlation heatmap"
            }
        ],
        "narrative": "Basic dataset analysis including schema inference, summary statistics, and correlation analysis."
    }


def validate_plan_json(plan_json: str) -> bool:
    """Quick validation of plan JSON format."""
    try:
        executor = PlanExecutor()
        executor.validate_plan(plan_json)
        return True
    except Exception:
        return False