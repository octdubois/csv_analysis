from pathlib import Path

import pytest

from csvi.plan_runner import run_plan
from csvi.bench import generate_synthetic_data


def test_unknown_step(tmp_path: Path) -> None:
    df = generate_synthetic_data(10)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    plan = {
        "steps": [
            {"name": "load_csv", "args": {"path": str(csv)}},
            {"name": "totally_unknown"},
            {"name": "summary_stats"},
        ]
    }
    res = run_plan(plan)
    assert "summary_stats" in res


def test_missing_args(tmp_path: Path) -> None:
    plan = {"steps": [{"name": "load_csv", "args": {}}]}
    res = run_plan(plan)
    assert "data" in res
