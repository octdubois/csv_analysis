# CSV Insight App

A local-only CSV analysis application that uses an offline LLM (Ollama) to automatically analyze datasets and generate charts through a safe, pre-defined API.

## 🚀 Features

- **Offline Only**: No internet/network calls other than localhost Ollama
- **Local LLM**: Uses Ollama (llama3.1:8b or mistral:7b) at http://localhost:11434
- **Auto-Analysis**: Intelligent playbook for dataset exploration
- **Natural Language Queries**: Chat with your data using natural language
- **Safe Execution**: Only whitelisted functions can be executed
- **Beautiful Charts**: Interactive Plotly visualizations
- **Dark Theme**: Modern, responsive UI

## 📋 Requirements

- Python 3.8+
- Ollama installed and running locally
- At least one LLM model downloaded (llama3.1:8b recommended)

## 🛠️ Installation

### 1. Install Ollama

**Windows:**
```bash
# Download from https://ollama.ai/download
# Run the installer and start Ollama
```

**macOS:**
```bash
brew install ollama
ollama serve
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

### 2. Download a Model

```bash
# Download the recommended model
ollama pull llama3.1:8b

# Or try Mistral
ollama pull mistral:7b
```

### 3. Install Python Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd csv-insight-app

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Usage

### 1. Upload CSV
- Go to the **Upload** tab
- Select your CSV file
- Preview the data and schema information

### 2. Auto-Analyze
- Click **Run Auto-Analysis** in the **Auto-Analyze** tab
- The app will automatically:
  - Analyze data structure
  - Generate summary statistics
  - Create correlation heatmaps
  - Plot time series (if datetime columns exist)
  - Profile categorical data
  - Detect outliers
- Results are saved to `./outputs/`

### 3. Chat with Data
- Use the **Chat with Data** tab
- Ask natural language questions like:
  - "Plot temperature over time and show correlation with humidity"
  - "Find outliers in the pressure column"
  - "Create a histogram of temperature values"
- The LLM generates a safe execution plan
- Results are displayed with interactive charts

### 4. Manual Charts
- Use the **Charts** tab for custom visualizations
- Select chart type and configure parameters
- Generate plots on demand

## 🔧 Configuration

### Ollama Settings
- **Base URL**: `http://localhost:11434` (default)
- **Model**: Select from available models
- **Temperature**: Control randomness (0.0-1.0)
- **Top-p**: Control diversity (0.1-1.0)

### System Prompt
The system prompt in `prompts/system.csv_analyst.txt` controls how the LLM generates analysis plans. It's designed to:
- Restrict function usage to safe, whitelisted functions
- Ensure proper JSON plan format
- Guide the LLM toward best practices

## 🛡️ Safety Features

- **No Internet Access**: All operations are local
- **No File System Access**: Only writes to `./outputs/`
- **No Code Execution**: Only uses whitelisted functions
- **Input Validation**: All LLM plans are validated before execution
- **Function Registry**: Safe function mapping with argument validation

## 📊 Allowed Functions

### Data Operations
- `load_csv(path, parse_dates=[])` - Load CSV file
- `infer_schema(df)` - Analyze dataset structure
- `coerce_types(df, dtypes_map)` - Convert column types
- `select_columns(df, columns)` - Select specific columns
- `filter_rows(df, condition)` - Filter rows by condition
- `resample(df, rule, agg)` - Resample time series data
- `groupby_agg(df, by, metrics)` - Group and aggregate data

### Statistics
- `summary_stats(df)` - Generate summary statistics
- `corr_matrix(df)` - Calculate correlation matrix
- `outliers_iqr(df, col)` - Detect outliers using IQR
- `rolling(df, window, agg)` - Calculate rolling statistics

### Charts
- `line_ts(df, x, y)` - Time series line plot
- `histogram(df, column)` - Histogram
- `boxplot(df, columns)` - Box plot
- `scatter(df, x, y, color)` - Scatter plot
- `corr_heatmap(df)` - Correlation heatmap
- `bar_agg(df, x, y, agg)` - Bar chart with aggregation

### Utilities
- `downsample_lttb(series, target_points)` - Downsample time series

## 📁 Project Structure

```
csv-insight-app/
├── app.py                      # Main Streamlit application
├── services/
│   ├── analysis.py            # Core analysis functions
│   ├── plots.py               # Chart generation functions
│   ├── executor.py            # Plan execution engine
│   └── ollama_client.py       # Ollama API client
├── prompts/
│   └── system.csv_analyst.txt # LLM system prompt
├── outputs/                    # Generated reports and plots
│   ├── reports/
│   └── plots/
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

## 📝 Example Analysis

### Sample CSV Structure
```csv
time,temperature_c,humidity_pct,pressure_hpa,lux
2024-01-01 00:00:00,22.5,65.2,1013.25,120
2024-01-01 00:01:00,22.6,65.1,1013.24,125
...
```

### Natural Language Query
**User**: "Plot temperature & humidity over time (hourly resample), show correlation heatmap, and flag outliers in temperature."

**LLM Generated Plan**:
```json
{
  "steps": [
    {"fn": "infer_schema", "args": {}, "description": "Understand data structure"},
    {"fn": "resample", "args": {"rule": "1H"}, "description": "Resample to hourly data"},
    {"fn": "line_ts", "args": {"x": "time", "y": ["temperature_c", "humidity_pct"]}, "description": "Plot time series"},
    {"fn": "corr_heatmap", "args": {}, "description": "Show correlations"},
    {"fn": "outliers_iqr", "args": {"col": "temperature_c"}, "description": "Detect temperature outliers"}
  ],
  "narrative": "Analyze environmental data with time series visualization, correlation analysis, and outlier detection"
}
```

## 🚨 Troubleshooting

### Ollama Not Connecting
- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is available
- Verify firewall settings

### Model Not Found
- Download a model: `ollama pull llama3.1:8b`
- Check available models: `ollama list`

### Memory Issues
- Use smaller models (7B instead of 13B+)
- Reduce dataset size for preview
- Close other applications

### Chart Generation Fails
- Ensure numeric columns exist for statistical functions
- Check datetime format for time series
- Verify column names match exactly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM capabilities
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for interactive charts
- [Pandas](https://pandas.pydata.org/) for data manipulation

---

**Built with ❤️ for local, private data analysis** 