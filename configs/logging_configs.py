from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import time


class DeltaFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._last_time = None

    def format(self, record):
        current_time = time.time()

        if self._last_time is None:
            delta = 0.0
        else:
            delta = current_time - self._last_time

        self._last_time = current_time

        record.delta = f"{delta:8.3f}s"
        return super().format(record)


def setup_logging(
    name: str, 
    log_file: Path,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
)->logging.Logger:
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

    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing handlers (avoids duplicate logs on repeated setup)
    logger.handlers.clear()

    formatter: DeltaFormatter =  DeltaFormatter(
    "%(asctime)s | +%(delta)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
    # Rotating file handler
    file_handler: RotatingFileHandler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger