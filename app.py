"""
CSV Insight App - Local CSV Analysis with Offline LLM
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

# Import our services
from services.ollama_client import OllamaClient
from services.executor import PlanExecutor, validate_plan_json
from services.analysis import load_csv
from services.plots import save_plot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = OllamaClient()
    if 'executor' not in st.session_state:
        st.session_state.executor = PlanExecutor()
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'training_examples' not in st.session_state:
        st.session_state.training_examples = load_training_examples()
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def load_training_examples() -> List[Dict[str, Any]]:
    """Load training examples from file."""
    try:
        if os.path.exists('training_examples.json'):
            with open('training_examples.json', 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load training examples: {e}")
    return []

def save_training_examples(examples: List[Dict[str, Any]]):
    """Save training examples to file."""
    try:
        with open('training_examples.json', 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(examples)} training examples")
    except Exception as e:
        logger.error(f"Failed to save training examples: {e}")

def add_training_example(original_query: str, corrected_query: str, context: str = ""):
    """Add a new training example."""
    example = {
        "timestamp": datetime.now().isoformat(),
        "original_query": original_query,
        "corrected_query": corrected_query,
        "context": context,
        "usage_count": 0
    }
    st.session_state.training_examples.append(example)
    save_training_examples(st.session_state.training_examples)
    st.success("✅ Training example added successfully!")

def get_training_context() -> str:
    """Get training context from examples."""
    if not st.session_state.training_examples:
        return ""
    
    context = "\n\nTRAINING EXAMPLES (use these patterns):\n"
    for i, example in enumerate(st.session_state.training_examples[-5:], 1):  # Last 5 examples
        context += f"{i}. Original: {example['original_query']}\n"
        context += f"   Corrected: {example['corrected_query']}\n"
        context += f"   Context: {example['context']}\n\n"
    
    return context

def sidebar():
    """Render sidebar with Ollama status and settings."""
    st.sidebar.title("🔧 Settings")
    
    # Ollama status
    if st.session_state.ollama_client.is_available():
        st.sidebar.success("✅ Ollama Connected")
        
        # Model selection
        models = st.session_state.ollama_client.get_models()
        if models:
            selected_model = st.sidebar.selectbox(
                "Select Model:",
                models,
                index=0
            )
            st.session_state.selected_model = selected_model
        else:
            st.sidebar.warning("No models available")
    else:
        st.sidebar.error("❌ Ollama Not Available")
        st.sidebar.info("Please ensure Ollama is running on localhost:11434")
    
    # Training examples count
    st.sidebar.metric("Training Examples", len(st.session_state.training_examples))
    
    # Clear training data
    if st.sidebar.button("🗑️ Clear Training Data"):
        st.session_state.training_examples = []
        save_training_examples([])
        st.rerun()

def generate_analysis_plan(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Generate analysis plan using Ollama with training context."""
    if not st.session_state.ollama_client.is_available():
        return {"error": "Ollama not available"}
    
    # Get training context
    training_context = get_training_context()
    
    # Create enhanced system prompt with training examples
    system_prompt = f"""You are a local CSV analysis agent for the CSV Insight app.

CRITICAL: You must return ONLY a valid JSON object. No explanations, no markdown, no code blocks.

Return this exact format:
{{
  "steps": [
    {{
      "fn": "function_name",
      "args": {{"arg1": "value1"}},
      "description": "What this step does"
    }}
  ],
  "narrative": "Brief description"
}}

ALLOWED FUNCTIONS WITH CORRECT ARGUMENTS:

Data Operations:
- load_csv(path, parse_dates=[])
- infer_schema(df) - no args needed
- coerce_types(df, dtypes_map) - dtypes_map is {{"col_name": "dtype"}}
- select_columns(df, columns) - columns is ["col1", "col2"]
- filter_rows(df, condition) - condition must be valid pandas query syntax
- resample(df, rule, agg, datetime_col)
- groupby_agg(df, by, metrics)

Statistics:
- summary_stats(df) - no args needed
- corr_matrix(df) - no args needed
- outliers_iqr(df, col) - col is column name
- rolling(df, window, agg, columns)

Charts:
- line_ts(df, x, y, resample, title)
- multi_line_overlay(df, x, y_columns, title)
- histogram(df, column, bins, title)
- boxplot(df, columns, title)
- scatter(df, x, y, color, title)
- corr_heatmap(df, title)
- bar_agg(df, x, y, agg, title)

Utilities:
- downsample_lttb(series, target_points)

FILTER SYNTAX RULES (CRITICAL):
- Use simple comparisons: column_name > value, column_name <= value
- Use quotes for strings: column_name == "string_value"
- Use & for AND, | for OR: (condition1) & (condition2)
- Use parentheses for complex conditions: (col1 > 10) & (col2 < 20)
- For dates: use string format: timestamp <= "2024-01-01"
- NEVER use: +, -, *, /, or complex expressions in filter conditions
- NEVER use: time_of_first_row, 24h, or similar expressions

VALID FILTER EXAMPLES:
- "temperature > 20"
- "timestamp <= \"2024-01-01\""
- "(temperature > 20) & (humidity < 80)"
- "category == \"A\""

INVALID FILTER EXAMPLES (DO NOT USE):
- "time <= time_of_first_row + 24h"
- "value + 10 > 20"
- "timestamp + pd.Timedelta(days=1)"

{training_context}

RULES:
1. Start with infer_schema, then summary_stats
2. Max 6 charts per plan
3. For time series: use resample(rule="1H") if >50k rows
4. Filter conditions must be simple pandas query syntax
5. Return ONLY the JSON object, nothing else

Example:
{{
  "steps": [
    {{"fn": "infer_schema", "args": {{}}, "description": "Analyze data structure"}},
    {{"fn": "summary_stats", "args": {{}}, "description": "Get overview statistics"}}
  ],
  "narrative": "Basic analysis"
}}"""
    
    # Create user prompt with data context
    user_prompt = f"""Analyze this dataset based on the query: "{query}"

Dataset info:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data types: {dict(df.dtypes)}

Please generate an analysis plan using only the allowed functions."""
    
    try:
        # Generate response
        response = st.session_state.ollama_client.generate_sync(
            prompt=user_prompt,
            system=system_prompt,
            model=st.session_state.get('selected_model', 'llama3.1:8b')
        )
        
        # DEBUG: Show the raw response
        st.write("🔍 **Debug - Raw LLM Response:**")
        st.code(response)
        
        # Try multiple JSON extraction methods
        json_str = None
        
        # Method 1: Look for JSON between curly braces
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            st.write(f"🔍 **Debug - Extracted JSON (Method 1):** {json_str[:200]}...")
        
        # Method 2: Look for JSON in code blocks
        if not json_str:
            import re
            code_blocks = re.findall(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
            if code_blocks:
                json_str = code_blocks[0]
                st.write(f"🔍 **Debug - Extracted JSON (Method 2):** {json_str[:200]}...")
        
        # Method 3: Look for any JSON-like structure
        if not json_str:
            import re
            json_matches = re.findall(r'{[^{}]*"steps"[^{}]*}', response, re.DOTALL)
            if json_matches:
                json_str = json_matches[0]
                st.write(f"🔍 **Debug - Extracted JSON (Method 3):** {json_str[:200]}...")
        
        if json_str:
            try:
                plan = json.loads(json_str)
                st.write("✅ **Debug - JSON Parsed Successfully!**")
                return plan
            except json.JSONDecodeError as e:
                st.error(f"❌ **Debug - JSON Parse Error:** {e}")
                st.write(f"**Problematic JSON:** {json_str}")
                return {"error": f"Invalid JSON format: {e}"}
        else:
            st.error("❌ **Debug - No JSON found in response**")
            st.write("**Full response:**")
            st.text(response)
            return {"error": "No valid JSON found in response"}
            
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        return {"error": str(e)}

def display_execution_results(result):
    """Display execution results with query editing capabilities."""
    if result.success:
        st.success(f"✅ Plan executed successfully in {result.execution_time:.2f}s")
        
        # Display results
        for i, step_result in enumerate(result.results):
            with st.expander(f"Step {i+1}: {step_result['step']['fn']}", expanded=True):
                st.write(f"**Description:** {step_result['step'].get('description', 'N/A')}")
                
                # Show step arguments
                if step_result['step']['args']:
                    st.write("**Arguments:**")
                    st.json(step_result['step']['args'])
                
                # Show result
                if isinstance(step_result['result'], pd.DataFrame):
                    st.write(f"**Result DataFrame ({step_result['result'].shape[0]} rows, {step_result['result'].shape[1]} columns):**")
                    st.dataframe(step_result['result'].head())
                elif isinstance(step_result['result'], dict):
                    st.write("**Result:**")
                    st.json(step_result['result'])
                else:
                    st.write(f"**Result:** {step_result['result']}")
        
        # Display plots
        if result.plots:
            st.write("**Generated Plots:**")
            for plot_name, plot_obj in result.plots.items():
                st.plotly_chart(plot_obj, use_container_width=True)
    
    else:
        st.error("❌ Plan execution failed")
        
        # Display errors with query editing
        for error in result.errors:
            st.error(error)
            
            # Check if it's a filter_rows error
            if "filter_rows" in error and "Invalid filter condition" in error:
                st.warning("🔧 This looks like a filter syntax error. You can correct it below:")
                
                # Extract the problematic condition
                import re
                condition_match = re.search(r"Invalid filter condition '([^']+)'", error)
                if condition_match:
                    problematic_condition = condition_match.group(1)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        corrected_condition = st.text_input(
                            "Edit the filter condition:",
                            value=problematic_condition,
                            help="Fix the pandas query syntax"
                        )
                    
                    with col2:
                        if st.button("💾 Save as Training Example"):
                            add_training_example(
                                original_query=problematic_condition,
                                corrected_query=corrected_condition,
                                context="Filter condition correction"
                            )
                            st.rerun()
                    
                    # Show suggested corrections
                    st.info("💡 **Suggested corrections:**")
                    st.code("""
# Instead of: time <= time_of_first_row + 24h
# Use: time <= "2024-01-02"

# Instead of: value + 10 > 20
# Use: value > 10

# Instead of: (col1 + col2) > 100
# Use: (col1 > 50) & (col2 > 50)
                    """)

def main():
    st.set_page_config(
        page_title="CSV Insight - Local Analysis Agent",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 CSV Insight - Local Analysis Agent")
    st.markdown("*Offline CSV analysis powered by local LLM*")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload", "🤖 Auto-Analyze", "💬 Chat with Data", "📈 Charts", "⚙️ Settings"])
    
    with tab1:
        st.header("📁 Upload CSV")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file to analyze"
        )
        
        if uploaded_file is not None:
            try:
                # Load CSV
                df = load_csv(uploaded_file)
                st.session_state.current_df = df
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
            except Exception as e:
                st.error(f"❌ Failed to load CSV: {e}")
    
    with tab2:
        st.header("🤖 Auto-Analyze")
        
        if st.session_state.current_df is not None:
            if st.button("🚀 Run Auto-Analysis", type="primary"):
                with st.spinner("Generating analysis plan..."):
                    # Generate plan
                    plan = generate_analysis_plan("Analyze this dataset", st.session_state.current_df)
                    
                    if "error" in plan:
                        st.error(f"❌ {plan['error']}")
                    else:
                        # Execute plan
                        with st.spinner("Executing analysis..."):
                            result = st.session_state.executor.execute_plan(
                                json.dumps(plan),
                                st.session_state.current_df
                            )
                        
                        # Display results
                        display_execution_results(result)
        else:
            st.info("👆 Please upload a CSV file first")
    
    with tab3:
        st.header("💬 Chat with Data")
        
        if st.session_state.current_df is not None:
            # Chat input
            user_query = st.text_input(
                "Ask questions about your data:",
                placeholder="e.g., 'Show me temperature trends over time'",
                help="Ask natural language questions about your data"
            )
            
            if user_query:
                if st.button("🔍 Analyze", type="primary"):
                    with st.spinner("Generating analysis plan..."):
                        # Generate plan
                        plan = generate_analysis_plan(user_query, st.session_state.current_df)
                        
                        if "error" in plan:
                            st.error(f"❌ {plan['error']}")
                        else:
                            # Show the generated plan
                            st.subheader("📝 Generated Analysis Plan")
                            st.json(plan)
                            
                            # Execute plan
                            with st.spinner("Executing analysis..."):
                                result = st.session_state.executor.execute_plan(
                                    json.dumps(plan),
                                    st.session_state.current_df
                                )
                            
                            # Display results
                            display_execution_results(result)
        else:
            st.info("👆 Please upload a CSV file first")
    
    with tab4:
        st.header("📈 Charts")
        
        if st.session_state.current_df is not None:
            st.info("Charts will be generated automatically during analysis. Use the Auto-Analyze or Chat tabs to create visualizations.")
        else:
            st.info("👆 Please upload a CSV file first")
    
    with tab5:
        st.header("⚙️ Settings")
        
        # Training examples management
        st.subheader("🎓 Training Examples")
        
        if st.session_state.training_examples:
            st.write(f"You have {len(st.session_state.training_examples)} training examples:")
            
            for i, example in enumerate(st.session_state.training_examples):
                with st.expander(f"Example {i+1}: {example['original_query'][:50]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Original:** {example['original_query']}")
                        st.write(f"**Corrected:** {example['corrected_query']}")
                        st.write(f"**Context:** {example['context']}")
                        st.write(f"**Added:** {example['timestamp']}")
                    
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            st.session_state.training_examples.pop(i)
                            save_training_examples(st.session_state.training_examples)
                            st.rerun()
        else:
            st.info("No training examples yet. Correct some LLM suggestions to build your training dataset!")
        
        # System prompt editor
        st.subheader("📝 System Prompt Editor")
        
        if os.path.exists('prompts/system.csv_analyst.txt'):
            with open('prompts/system.csv_analyst.txt', 'r', encoding='utf-8') as f:
                current_prompt = f.read()
            
            edited_prompt = st.text_area(
                "Edit the system prompt:",
                value=current_prompt,
                height=400,
                help="Modify the system prompt to improve LLM behavior"
            )
            
            if st.button("💾 Save System Prompt"):
                with open('prompts/system.csv_analyst.txt', 'w', encoding='utf-8') as f:
                    f.write(edited_prompt)
                st.success("✅ System prompt saved!")
                st.rerun()
        else:
            st.warning("System prompt file not found")

if __name__ == "__main__":
    main() 