from csvi.report_runner import generate_insight_report


def test_report_generation() -> None:
    analysis = {
        "summary_stats": {
            "n_rows": 10,
            "missing_pct": {"a": 0.1},
            "constant_columns": ["b"],
        },
        "outliers_iqr": {"a": {"pct": 0.1}},
        "corr_matrix": {"a": {"a": 1.0}},
    }
    rep = generate_insight_report(analysis)
    assert "guide" in rep
    assert "markdown" in rep
    assert "Executive Summary" in rep["markdown"]
    assert "constant_columns" in rep["guide"]["findings"]
