"""Logging utilities for Thunders AI.

Provides a configurable logging system with support for console and file output,
structured logging, and integration with the Rich library for beautiful terminal output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ThundersLogger:
    """Custom logger for Thunders AI with Rich integration.
    
    Provides colorful, formatted console output and optional file logging
    with automatic rotation and structured formatting.
    
    Example:
        >>> logger = ThundersLogger("my_module", level="DEBUG")
        >>> logger.info("Processing started")
        >>> logger.error("Something went wrong", exc_info=True)
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    
    def __init__(
        self,
        name: str = "thunders_ai",
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
    ):
        """Initialize the logger.
        
        Args:
            name: Logger name, typically module name.
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Optional file path for log output.
            format_string: Custom format string for log messages.
        """
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self._logger = self._get_or_create_logger(name, level, log_file, format_string)
    
    def _get_or_create_logger(
        self,
        name: str,
        level: str,
        log_file: Optional[str],
        format_string: Optional[str],
    ) -> logging.Logger:
        """Get existing logger or create a new one."""
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()
        
        fmt = format_string or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self._loggers[name] = logger
        return logger
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self._logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log an exception message with traceback."""
        self._logger.exception(message, *args, **kwargs)


def get_logger(name: str = "thunders_ai", level: str = "INFO") -> ThundersLogger:
    """Get a ThundersLogger instance.
    
    Args:
        name: Logger name.
        level: Logging level.
        
    Returns:
        Configured ThundersLogger instance.
    """
    return ThundersLogger(name=name, level=level)
