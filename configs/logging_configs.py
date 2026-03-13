from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal
import numpy as np
import warnings



class DeltaFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._last_created: float | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        current = record.created  # seconds since epoch (set by logging)
        if self._last_created is None:
            delta = 0.0
        else:
            delta = current - self._last_created

        self._last_created = current
        record.delta = f"{delta:8.3f}s"
        return True


def setup_logging(
    name: str,
    log_file: Path,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> logging.Logger:
    """
    Function to initialise the logging configuration for the application.

    Parameters
    ----------
    name: str
        Name of the logger
    log_file : Path
        The path to the log file where logs will be written.
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
        The logging level to set for the logger. Defaults to "INFO".

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """

    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Set additional information
    if level in ["DEBUG", "WARNING", "ERROR", "CRITICAL"]:
        np.seterr(all="raise")
        warnings.simplefilter("error")


    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicates if setup_logger is called again
    logger.handlers.clear()
    logger.filters.clear()

    formatter = logging.Formatter(
        "%(asctime)s | +%(delta)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Delta computed once per record
    logger.addFilter(DeltaFilter())

    return logger
