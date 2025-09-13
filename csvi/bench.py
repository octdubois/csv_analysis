"""Synthetic data generation and benchmarks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .stats_utils import iqr_outliers

try:  # optional
    import psutil
except Exception:  # pragma: no cover - optional
    psutil = None


def generate_synthetic_data(rows: int, start_ts: str = "2020-01-01", cadence_s: int = 60, noise: float = 1.0,
                             nan_pct: float = 0.0, reboot_frequency: Optional[int] = None) -> pd.DataFrame:
    rng = pd.date_range(start=start_ts, periods=rows, freq=f"{cadence_s}S")
    temp = 20 + np.random.randn(rows) * noise
    hum = 50 + np.random.randn(rows) * noise
    uptime = np.arange(rows) * cadence_s * 1000
    if reboot_frequency and reboot_frequency > 0:
        uptime = uptime % (reboot_frequency * cadence_s * 1000)
    df = pd.DataFrame({"time": rng, "temperature_c": temp, "humidity_pct": hum, "uptime_ms": uptime})
    if nan_pct > 0:
        mask = np.random.rand(rows) < nan_pct
        df.loc[mask, "temperature_c"] = np.nan
    return df


def run_benchmark(rows: int) -> Dict[str, Any]:
    df = generate_synthetic_data(rows)

    t0 = time.perf_counter()
    _ = df.set_index("time").resample("1T").mean()
    resample_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    df_roll = df.set_index("time")
    df_roll["rolling"] = df_roll["temperature_c"].rolling("60T").mean()
    rolling_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    _ = iqr_outliers(df, ["temperature_c"])
    iqr_time = time.perf_counter() - t2

    mem = None
    if psutil:
        try:
            process = psutil.Process()
            mem = process.memory_info().rss
        except Exception:
            mem = None

    return {"resample_s": resample_time, "rolling_s": rolling_time, "iqr_s": iqr_time, "peak_memory": mem, "rows": rows}
