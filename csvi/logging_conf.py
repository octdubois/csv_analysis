"""Logging configuration for CSV Insight."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger.

    Parameters
    ----------
    level: int
        Logging level.
    name: Optional[str]
        Logger name; defaults to ``__name__``.
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    logger.setLevel(level)
    return logger
