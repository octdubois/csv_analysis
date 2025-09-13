"""Time related helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a series into timezone aware datetimes."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def ensure_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = parse_datetime(df[column])
    return df


def estimate_cadence(series: pd.Series) -> Optional[pd.Timedelta]:
    if series.empty:
        return None
    s = series.sort_values().dropna()
    if len(s) < 2:
        return None
    deltas = s.diff().dropna()
    if deltas.empty:
        return None
    return deltas.mode().iloc[0]
