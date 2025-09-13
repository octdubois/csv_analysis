import json
from pathlib import Path

import pandas as pd

from csvi.plan_runner import run_plan
from csvi.bench import generate_synthetic_data


def test_resample_and_summary(tmp_path: Path) -> None:
    df = generate_synthetic_data(100)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    plan = {
        "steps": [
            {"name": "load_csv", "args": {"path": str(csv)}},
            {"name": "coerce_types", "args": {"columns": {"time": "datetime"}}},
            {"name": "resample", "args": {"rule": "1T", "agg": "mean", "datetime_col": "time"}},
            {"name": "summary_stats"},
        ]
    }
    res = run_plan(plan)
    assert "summary_stats" in res
    assert res["summary_stats"]["n_rows"] > 0


def test_rolling_and_iqr(tmp_path: Path) -> None:
    df = generate_synthetic_data(200)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    plan = {
        "steps": [
            {"name": "load_csv", "args": {"path": str(csv)}},
            {"name": "rolling", "args": {"columns": ["temperature_c"], "window": 60, "stats": ["mean"]}},
            {"name": "outliers_iqr", "args": {"columns": ["temperature_c"]}},
        ]
    }
    res = run_plan(plan)
    assert "outliers_iqr" in res
    assert isinstance(res["outliers_iqr"], dict)
