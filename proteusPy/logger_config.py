"""
This module provides utility functions for configuring and managing loggers within the proteusPy package.

Functions:
    get_logger(name): Returns a logger with the specified name, configured to use a shared StreamHandler.
    set_logger_level(name, level): Sets the logging level for the logger with the specified name.

Example usage:
    logger = get_logger("example_logger")
    logger.info("This is an info message")
    set_logger_level("example_logger", "ERROR")
    logger.info("This info message will not be shown")
    logger.error("This is an error message")
"""

import logging

# Disable the root logger
logging.getLogger().disabled = True


def get_logger(name):
    """
    Returns a logger with the specified name, configured to use the shared StreamHandler
    with the predefined format. Ensures that the StreamHandler is not added multiple times.

    :param name: The name of the logger.
    :type name: str
    :return: Configured logger instance.
    :rtype: logging.Logger
    """
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.ERROR)  # Set the logger level to INFO by default
    # Check if the logger already has handlers to avoid adding multiple handlers
    if not _logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "proteusPy: %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    # Enable the specific logger
    _logger.disabled = False
    return _logger


# example
# logger = get_logger("example_logger")
# logger.info("This is an info message")
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
    }

    if level not in level_dict:
        raise ValueError(
            f"--> set_logger_level(): Invalid logging level: {level}. Must be one of ['WARNING', 'ERROR', 'INFO']"
        )

    _logger = logging.getLogger(name)
    _logger.setLevel(level_dict[level])


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
