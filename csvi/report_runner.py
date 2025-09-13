"""Generate structured insight reports from analysis results."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _markdown_section(title: str, content: List[str]) -> str:
    body = "\n".join(content)
    return f"## {title}\n{body}\n"


def generate_insight_report(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create structured JSON and Markdown report."""
    summary = analysis.get("summary_stats", {})
    outliers = analysis.get("outliers_iqr", {})
    corr = analysis.get("corr_matrix", {})

    n_rows = summary.get("n_rows", 0)
    missing_pct = summary.get("missing_pct", {})
    avg_missing = float(pd.Series(missing_pct).mean()) if missing_pct else 0.0
    data_quality = max(0.0, 1.0 - avg_missing)

    outlier_pct = float(np.mean([v.get("pct", 0.0) for v in outliers.values()])) if outliers else 0.0

    flags: Dict[str, Any] = {}
    const_cols = summary.get("constant_columns", [])
    if const_cols:
        flags["constant_columns"] = const_cols
    high_missing = [c for c, p in missing_pct.items() if p >= 0.3]
    if high_missing:
        flags["high_missingness"] = high_missing

    guide = {
        "dataset": {"rows": n_rows},
        "scores": {"data_quality": data_quality, "outliers": 1.0 - outlier_pct},
        "findings": flags,
        "recommendations": {},
        "next_steps": [],
    }

    md_sections = [
        "# Executive Summary",
        f"Rows analysed: {n_rows}\n",
        _markdown_section("Key Findings", [str(flags)]),
        _markdown_section("Column Quality", [str(missing_pct)]),
        _markdown_section("Recommended Transforms", ["Consider dropping constant columns"]) ,
        _markdown_section("Visualization Shortlist", ["corr_matrix" if corr else ""]),
        _markdown_section("Operational Tips", ["Ensure time column is correctly parsed"]),
        _markdown_section("Next Steps", ["Collect more data"]),
    ]
    markdown = "\n".join(md_sections)
    return {"title": "CSV Insight Report", "guide": guide, "markdown": markdown}
