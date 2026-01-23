"""
Centralized Logging Configuration for ML_OSINT

Provides consistent logging setup across all modules.

Usage:
    from config.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training...")
    logger.warning("Missing data for date range")
    logger.error("Model checkpoint not found", exc_info=True)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.paths import LOG_DIR, ensure_dir


# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# Cached loggers
_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_file: Optional[str] = None,
    detailed: bool = False,
) -> logging.Logger:
    """Get or create a logger with consistent configuration.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_to_file: Whether to also log to a file
        log_file: Custom log file name (default: ml_osint.log)
        detailed: Use detailed format with filename and line numbers

    Returns:
        Configured logger instance
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Choose format
    fmt = DETAILED_FORMAT if detailed else DEFAULT_FORMAT
    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_to_file:
        ensure_dir(LOG_DIR)
        log_filename = log_file or "ml_osint.log"
        file_path = LOG_DIR / log_filename

        file_handler = logging.FileHandler(file_path, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Cache the logger
    _loggers[name] = logger

    return logger


def get_training_logger(model_name: str) -> logging.Logger:
    """Get a logger specifically for training runs.

    Creates a separate log file for each training run with timestamp.

    Args:
        model_name: Name of the model being trained

    Returns:
        Configured logger for training
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_{model_name}_{timestamp}.log"

    return get_logger(
        f"training.{model_name}",
        level=logging.DEBUG,
        log_to_file=True,
        log_file=log_file,
        detailed=True,
    )


def get_probe_logger() -> logging.Logger:
    """Get a logger for probe execution."""
    return get_logger(
        "probes",
        level=logging.INFO,
        log_to_file=True,
        log_file="probes.log",
    )


def get_data_loader_logger() -> logging.Logger:
    """Get a logger for data loading operations."""
    return get_logger(
        "data_loader",
        level=logging.INFO,
        log_to_file=True,
        log_file="data_loading.log",
    )


def setup_root_logger(level: int = logging.INFO, detailed: bool = False):
    """Configure the root logger for the entire application.

    Call this at the start of entry point scripts.

    Args:
        level: Logging level
        detailed: Use detailed format
    """
    fmt = DETAILED_FORMAT if detailed else DEFAULT_FORMAT
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ensure_dir(LOG_DIR) / "ml_osint.log", mode="a"),
        ],
    )


def silence_library_loggers():
    """Reduce verbosity of third-party library loggers."""
    noisy_loggers = [
        "matplotlib",
        "PIL",
        "urllib3",
        "requests",
        "httpx",
        "httpcore",
        "asyncio",
        "fsspec",
        "s3fs",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
