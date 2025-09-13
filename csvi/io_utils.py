"""IO utilities for CSV Insight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd


def load_csv(path: str | Path, chunksize: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file.

    Parameters
    ----------
    path: str or Path
        CSV file path.
    chunksize: Optional[int]
        If provided, load CSV in chunks and concatenate.
    kwargs: Any
        Additional ``pandas.read_csv`` arguments.
    """
    if chunksize:
        chunks: Iterator[pd.DataFrame] = pd.read_csv(path, chunksize=chunksize, **kwargs)
        return pd.concat(chunks, ignore_index=True)
    return pd.read_csv(path, **kwargs)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_dataframe(path: str | Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)
