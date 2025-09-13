from pathlib import Path

import pandas as pd
try:
    from hypothesis import given, strategies as st
except Exception:
    import pytest
    pytest.skip("hypothesis not installed", allow_module_level=True)

from csvi.plan_runner import run_plan


@given(st.lists(st.floats(-1e6, 1e6, allow_nan=True), min_size=10, max_size=50))
def test_property_edges(vals, tmp_path: Path) -> None:
    n = len(vals)
    times = pd.date_range("2020", periods=n, freq="T").to_list()
    if n > 1:
        times[0] = times[1]  # duplicate timestamp
    df = pd.DataFrame({"time": times, "temperature_c": vals, "constant": 1})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    plan = {
        "steps": [
            {"name": "load_csv", "args": {"path": str(csv)}},
            {"name": "summary_stats"},
            {"name": "outliers_iqr", "args": {"columns": ["temperature_c"]}},
        ]
    }
    res = run_plan(plan)
    stats = res["summary_stats"]
    assert "constant" in stats["constant_columns"]
    assert "temperature_c" in res["outliers_iqr"]
