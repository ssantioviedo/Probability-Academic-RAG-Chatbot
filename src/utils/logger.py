"""
Logging module for application-wide logging configuration.

Provides dual output logging to both console and file, with
configurable log levels and formatted output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Log format constants
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log directory
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "app.log"

# Store initialized loggers
_loggers: dict[str, logging.Logger] = {}
_initialized: bool = False


def setup_logger(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: str = DEFAULT_LOG_FILE
) -> logging.Logger:
    """
    Set up the root logger with console and file handlers.
    
    Configures logging to output INFO+ to console and DEBUG+ to file.
    Creates log directory if it doesn't exist.
    
    Args:
        log_level: Minimum log level for console output (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files. Defaults to ./logs.
        log_file: Name of the log file. Defaults to app.log.
    
    Returns:
        Configured root logger instance.
    
    Example:
        >>> logger = setup_logger("DEBUG")
        >>> logger.info("Application started")
    """
    global _initialized
    
    if _initialized:
        return logging.getLogger("rag_chatbot")
    
    # Set up log directory
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    # Create root logger
    root_logger = logging.getLogger("rag_chatbot")
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Prevent duplicate handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Console handler (INFO+ by default, configurable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = getattr(logging, log_level.upper(), logging.INFO)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (DEBUG+ always)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log initialization
    root_logger.info(f"Logger initialized - Console: {log_level}, File: DEBUG")
    root_logger.debug(f"Log file: {log_path.absolute()}")
    
    _initialized = True
    _loggers["rag_chatbot"] = root_logger
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger that inherits from the root logger.
    
    Creates a child logger of the root 'rag_chatbot' logger,
    which inherits all handlers and configuration.
    
    Args:
        name: Name for the logger (typically __name__).
    
    Returns:
        Logger instance with the given name.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    # Ensure root logger is set up
    if not _initialized:
        setup_logger()
    
    # Create child logger
    full_name = f"rag_chatbot.{name}" if not name.startswith("rag_chatbot") else name
    
    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)
    
    return _loggers[full_name]


class LoggerMixin:
    """
    Mixin class to provide logging capability to any class.
    
    Adds a 'logger' property that returns a logger named after the class.
    
    Example:
        >>> class MyProcessor(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("Processing...")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger named after this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger




def log_processing_stats(
    logger: logging.Logger,
    operation: str,
    items_processed: int,
    duration_seconds: float,
    extra_info: Optional[dict] = None
) -> None:
    """
    Log standardized processing statistics.
    
    Args:
        logger: Logger instance to use.
        operation: Name of the operation performed.
        items_processed: Number of items processed.
        duration_seconds: Time taken in seconds.
        extra_info: Additional information to log.
    
    Example:
        >>> log_processing_stats(logger, "PDF extraction", 10, 5.5)
    """
    rate = items_processed / duration_seconds if duration_seconds > 0 else 0
    
    message = (
        f"{operation} completed: "
        f"{items_processed} items in {duration_seconds:.2f}s "
        f"({rate:.2f} items/sec)"
    )
    
    if extra_info:
        extra_str = ", ".join(f"{k}={v}" for k, v in extra_info.items())
        message += f" [{extra_str}]"
    
    logger.info(message)
