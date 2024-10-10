"""
This module provides utility functions for configuring and managing loggers 
within the proteusPy package.

Functions:
    get_logger(name): Returns a logger with the specified name, configured to use a shared 
    StreamHandler.
    set_logger_level(name, level): Sets the logging level for the logger with the 
    specified name.

Example usage:
    logger = get_logger("example_logger")
    logger.info("This is an info message")
    set_logger_level("example_logger", "ERROR")
    logger.info("This info message will not be shown")
    logger.error("This is an error message")
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Disable the root logger
logging.getLogger().disabled = True


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_dir: str = None,
    max_bytes: int = 10**6,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Returns a logger with the specified name, configured to use both StreamHandler
    and RotatingFileHandler with the predefined formats.

    :param name: The name of the logger.
    :type name: str
    :param log_level: The logging level, defaults to logging.INFO
    :type log_level: int, optional
    :param log_dir: Directory where log files will be stored, defaults to current directory
    :type log_dir: Optional[str], optional
    :param max_bytes: Maximum size in bytes before a log file is rotated, defaults to 1,000,000
    :type max_bytes: int, optional
    :param backup_count: Number of backup files to keep, defaults to 5
    :type backup_count: int, optional
    :return: Configured logger instance.
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define formatter
    formatter = logging.Formatter(
        "proteusPy: %(levelname)s %(asctime)s - %(name)s.%(funcName)s - %(message)s"
    )

    # StreamHandler setup
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Prevent log messages from being propagated to the root logger
    logger.propagate = False

    # Determine log directory
    if log_dir is None:
        log_dir = os.getcwd()
    else:
        os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.abspath(os.path.join(log_dir, f"{name}.log"))

    try:
        # RotatingFileHandler setup
        file_handler = RotatingFileHandler(
            log_path,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.debug(f"Logging to file: {log_path}")

    except Exception as e:
        logger.error(f"Failed to set up file handler: {e}")

    return logger


# set_logger_level("example_logger", "ERROR")
# logger.info("This info message will not be shown")
# logger.error("This is an error message")


def set_logger_level(name, level):
    """
    Sets the logging level for the logger with the specified name.

    :param name: The name of the logger.
    :type name: str
    :param level: The logging level to set. Must be one of ["WARNING", "ERROR", "INFO"].
    :type level: str
    :raises ValueError: If the provided level is not one of the allowed values.
    """
    level_dict = {
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    if level not in level_dict:
        raise ValueError(
            f"--> set_logger_level(): Invalid logging level: {level}. Must be one of ['WARNING', 'ERROR', 'INFO', 'DEBUG']"
        )

    _logger = logging.getLogger(name)
    _logger.setLevel(level_dict[level])


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
