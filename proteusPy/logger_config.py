"""
This module provides utility functions for configuring and managing loggers 
within the proteusPy package. The functions are used within the package to
convey logging information at a fine-grained level. The functions are completely
independent of the application and can be used in any Python project.

Author: Eric G. Suchanek, PhD
Last updated 2024-11-19 -egs-
"""

import logging
from pathlib import Path


def set_logging_level_for_all_handlers(log_level: int):
    """
    Sets the logging level for all handlers of all loggers in the proteusPy package.

    :param log_level: The logging level to set.
    :type log_level: int
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Set the level for the root logger
    root_logger.setLevel(log_level)

    # Iterate through all loggers
    for logger_name in logging.Logger.manager.loggerDict:
        _logger = logging.getLogger(logger_name)

        # Set the level for the logger itself
        _logger.setLevel(log_level)

        # Iterate through all handlers of the logger
        for handler in _logger.handlers:
            handler.setLevel(log_level)


def disable_stream_handlers_for_namespace(namespace: str):
    """
    Disables all stream handlers for all loggers under the specified namespace.

    :param namespace: The namespace whose stream handlers should be disabled.
    :type namespace: str
    """
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith(namespace):
            _logger = logging.getLogger(logger_name)
            for handler in _logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    _logger.removeHandler(handler)


def configure_master_logger(
    log_file: str,
    file_path: str = "~/logs",
    log_level: int = logging.DEBUG,
):
    """
    Configures the root logger to write to a specified log file.

    Args:
        log_file (str): Name of the log file.
        file_path (str): Path to the directory where log files will be stored. Defaults to '~/logs'.
        max_bytes (int): Maximum size of the log file before rotating.
        backup_count (int): Number of backup files to keep.
    """
    # Expand user path
    file_path = Path(file_path).expanduser()

    # Ensure the directory exists
    file_path.mkdir(parents=True, exist_ok=True)

    # Full path to the log file
    full_log_file_path = file_path / log_file

    root_logger = logging.getLogger()

    # Set the root logger level to DEBUG to capture all messages
    root_logger.setLevel(log_level)

    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a new FileHandler
    handler = logging.FileHandler(full_log_file_path, mode="w")

    formatter = logging.Formatter(
        "proteusPy: %(levelname)s %(asctime)s - %(name)s.%(funcName)s - %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    root_logger.disabled = False  # Enable the root logger
    for handler in root_logger.handlers:
        handler.setLevel(log_level)


def create_logger(
    name: str,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Returns a logger with the specified name, configured to use both StreamHandler
    and RotatingFileHandler with the predefined formats.

    :param name: The name of the logger.
    :type name: str
    :param log_level: The logging level, defaults to logging.INFO
    :type log_level: int, optional
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

    # Allows log messages to propagate to the root logger
    logger.propagate = True

    return logger


def set_logger_level(name, level):
    """
    Sets the logging level for the logger with the specified name.

    :param name: The name of the logger.
    :type name: str
    :param level: The logging level to set. Must be one of ["WARNING", "ERROR", "INFO", "DEBUG"].
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
            (
                f"--> set_logger_level(): Invalid logging level: {level}."
                f"Must be one of ['WARNING', 'ERROR', 'INFO', 'DEBUG']"
            )
        )

    _logger = logging.getLogger(name)
    _logger.setLevel(level_dict[level])

    for handler in _logger.handlers:
        handler.setLevel(level_dict[level])


def toggle_stream_handler(name, enable):
    """
    Enables or disables the StreamHandler for the logger with the specified name.

    :param name: The name of the logger.
    :type name: str
    :param enable: If True, enables the StreamHandler; if False, disables it.
    :type enable: bool
    """
    logger = logging.getLogger(name)
    stream_handler = None

    # Find the StreamHandler if it exists
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream_handler = handler
            break

    if enable:
        if stream_handler is None:
            # Define formatter
            formatter = logging.Formatter(
                "stream proteusPy: %(levelname)-7s %(asctime)s - %(name)s.%(funcName)s - %(message)s"
            )
            # Create and add a new StreamHandler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logger.level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    else:
        if stream_handler is not None:
            # Remove the existing StreamHandler
            logger.removeHandler(stream_handler)


def list_all_loggers():
    """
    Lists all loggers that have been created in the application.

    :return: List of logger names.
    :rtype: list
    """
    logger_dict = logging.Logger.manager.loggerDict
    loggers = [
        name for name in logger_dict if isinstance(logger_dict[name], logging.Logger)
    ]
    return loggers


def list_handlers(name):
    """
    Lists the handlers for the logger with the specified name.

    :param name: The name of the logger.
    :type name: str
    :return: List of handler types and their configurations.
    :rtype: list
    """
    logger = logging.getLogger(name)
    handlers_info = []

    for handler in logger.handlers:
        handler_type = type(handler).__name__
        handler_info = {
            "type": handler_type,
            "level": logging.getLevelName(handler.level),
            "formatter": handler.formatter._fmt if handler.formatter else None,
        }
        handlers_info.append(handler_info)

    return handlers_info


def set_logger_level_for_module(pkg_name, level=""):
    """
    Set the logging level for all loggers within a specified package.

    This function iterates through all registered loggers and sets the logging
    level for those that belong to the specified package.

    :param pkg_name: The name of the package for which to set the logging level.
    :type pkg_name: str
    :param level: The logging level to set (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
                  If not specified, the logging level will not be changed.
    :type level: str, optional
    :return: A list of logger names that were found and had their levels set.
    :rtype: list
    """
    logger_dict = logging.Logger.manager.loggerDict
    registered_loggers = [
        name
        for name, logger in logger_dict.items()
        if isinstance(logger, logging.Logger) and name.startswith(pkg_name)
    ]
    for logger_name in registered_loggers:
        logger = logging.getLogger(logger_name)
        if level:
            logger.setLevel(level)

    return registered_loggers


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
