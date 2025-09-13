# CSV Insight

Minimal analytics runner for CSV Insight application.

## Installation

```bash
pip install -e .[dev]
```

## Usage

Run a plan on a CSV file:

```bash
csvi run --plan plan.json --data data.csv --out out_dir/
```

Generate a report from analysis results:

```bash
csvi report --analysis out_dir/analysis.json --out out_dir/
```

Run a synthetic benchmark:

```bash
csvi bench --rows 1000000 --out out_dir/
```

## Sample plan

```json
{
  "steps": [
    {"name": "load_csv", "args": {"path": "data.csv"}},
    {"name": "coerce_types", "args": {"columns": {"time": "datetime"}}},
    {"name": "resample", "args": {"rule": "1T", "agg": "mean", "datetime_col": "time"}},
    {"name": "summary_stats"},
    {"name": "corr_matrix"},
    {"name": "outliers_iqr", "args": {"columns": ["temperature_c"]}}
  ]
}
```

## Generate sample data

```python
from csvi.bench import generate_synthetic_data

df = generate_synthetic_data(rows=1000)
df.to_csv("data.csv", index=False)
```

## Testing

```bash
pytest
```
