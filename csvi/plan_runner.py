"""Plan runner executing analysis steps."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Tuple

import pandas as pd

from .io_utils import load_csv
from .stats_utils import format_corr_matrix, iqr_outliers, rolling_stats
from .time_utils import ensure_datetime
from .logging_conf import configure_logging

logger = configure_logging(name=__name__)


# Step implementations -----------------------------------------------------

def step_load_csv(_: pd.DataFrame | None, path: str, **kwargs: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = load_csv(path, **kwargs)
    return df, {}


def step_infer_schema(df: pd.DataFrame | None, **_: Any) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    return df, {c: str(t) for c, t in df.dtypes.items()}


def step_coerce_types(df: pd.DataFrame | None, columns: Dict[str, str]) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    for col, typ in columns.items():
        if col not in df.columns:
            continue
        if typ.startswith("datetime"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            try:
                df[col] = df[col].astype(typ)
            except Exception:
                logger.warning("failed to coerce %s to %s", col, typ)
    return df, {}


def step_select_columns(df: pd.DataFrame | None, columns: Iterable[str]) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    cols = [c for c in columns if c in df.columns]
    return df[cols], {}


def step_filter_rows(df: pd.DataFrame | None, query: str) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    try:
        return df.query(query), {}
    except Exception as exc:
        logger.warning("filter_rows failed: %s", exc)
        return df, {}


def step_resample(df: pd.DataFrame | None, rule: str, agg: str = "mean", datetime_col: str = "time") -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None or datetime_col not in df.columns:
        logger.warning("resample: missing column %s", datetime_col)
        return df, {}
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.set_index(datetime_col)
    res = df.resample(rule).agg(agg)
    res = res.reset_index()
    return res, {}


def step_groupby_agg(df: pd.DataFrame | None, by: Iterable[str], agg: Dict[str, str]) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    cols = [c for c in by if c in df.columns]
    if not cols:
        return df, {}
    res = df.groupby(cols).agg(agg).reset_index()
    return res, {}


def step_summary_stats(df: pd.DataFrame | None) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None or df.empty:
        return df, {}
    missing = df.isna().mean().to_dict()
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    duplicates = int(df.duplicated().sum())
    numeric = df.select_dtypes("number").describe().to_dict()
    return df, {
        "missing_pct": missing,
        "constant_columns": constant_cols,
        "duplicate_rows": duplicates,
        "numeric_stats": numeric,
        "n_rows": len(df),
    }


def step_corr_matrix(df: pd.DataFrame | None) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    numeric = df.select_dtypes("number")
    return df, format_corr_matrix(numeric)


def step_outliers_iqr(df: pd.DataFrame | None, columns: Iterable[str], k: float = 1.5) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    cols = [c for c in columns if c in df.columns]
    res = iqr_outliers(df, cols, k=k)
    return df, res


def step_rolling(df: pd.DataFrame | None, columns: Iterable[str], window: int, stats: Iterable[str]) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    if df is None:
        return df, {}
    if "time" in df.columns:
        df = df.set_index(pd.to_datetime(df["time"], errors="coerce"))
    try:
        df = rolling_stats(df, columns, window, stats)
    except Exception as exc:
        logger.warning("rolling failed: %s", exc)
    df = df.reset_index().rename(columns={"index": "time"})
    return df, {}


def step_downsample_lttb(df: pd.DataFrame | None, **_: Any) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    return df, {}


def step_noop(df: pd.DataFrame | None, **_: Any) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    return df, {}


ALLOWED_FUNCS: Dict[str, Callable[..., Tuple[pd.DataFrame | None, Dict[str, Any]]]] = {
    "load_csv": step_load_csv,
    "infer_schema": step_infer_schema,
    "coerce_types": step_coerce_types,
    "select_columns": step_select_columns,
    "filter_rows": step_filter_rows,
    "resample": step_resample,
    "groupby_agg": step_groupby_agg,
    "summary_stats": step_summary_stats,
    "corr_matrix": step_corr_matrix,
    "outliers_iqr": step_outliers_iqr,
    "rolling": step_rolling,
    "downsample_lttb": step_downsample_lttb,
    # chart placeholders
    "line_ts": step_noop,
    "multi_line_overlay": step_noop,
    "histogram": step_noop,
    "boxplot": step_noop,
    "scatter": step_noop,
    "corr_heatmap": step_noop,
    "bar_agg": step_noop,
}


# Runner ------------------------------------------------------------------

def run_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a plan and return results."""
    df: pd.DataFrame | None = None
    results: Dict[str, Any] = {}
    for step in plan.get("steps", []):
        name = step.get("name")
        args = step.get("args", {}) or {}
        func = ALLOWED_FUNCS.get(name)
        if not func:
            logger.warning("Unknown step %s", name)
            continue
        try:
            df, res = func(df, **args)
            if res:
                results[name] = res
        except Exception as exc:
            logger.warning("Step %s failed: %s", name, exc)
    results["data"] = df
    return results
