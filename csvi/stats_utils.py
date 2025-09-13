"""Statistical helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def iqr_outliers(df: pd.DataFrame, columns: Iterable[str], k: float = 1.5) -> Dict[str, Dict[str, float]]:
    """Detect outliers using the IQR rule."""
    results: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            results[col] = {"count": 0, "pct": 0.0}
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        pct = float(count / len(series)) if len(series) else 0.0
        results[col] = {"count": count, "pct": pct, "lower": lower, "upper": upper}
    return results


def rolling_stats(df: pd.DataFrame, columns: Iterable[str], window: int, stats: Iterable[str]) -> pd.DataFrame:
    """Add rolling statistics columns."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Rolling stats require DatetimeIndex")
    for col in columns:
        if col not in df.columns:
            continue
        roll = df[col].rolling(f"{window}T")
        for stat in stats:
            new_col = f"rolling_{stat}_{window}_{col}"
            if stat == "mean":
                df[new_col] = roll.mean()
            elif stat == "std":
                df[new_col] = roll.std()
    return df


def format_corr_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    corr = df.corr().fillna(0.0)
    return {c: corr.loc[c].to_dict() for c in corr.columns}
