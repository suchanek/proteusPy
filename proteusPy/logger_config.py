"""
logger_config.py

This module configures logging for the application. It sets up a single StreamHandler
with a specific format and provides a function to get loggers with consistent settings.

Functions:
    get_logger(name): Returns a logger with the specified name, configured to use the
                      shared StreamHandler with the predefined format.

Usage:
    from logger_config import get_logger

    logger = get_logger('my_module')
    logger.info('This is an info message')
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a single StreamHandler
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)


def get_logger(name):
    """
    Returns a logger with the specified name, configured to use the shared StreamHandler
    with the predefined format. Ensures that the StreamHandler is not added multiple times.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Check if the logger already has the StreamHandler to avoid adding it multiple times
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        logger.addHandler(stream_handler)
    return logger


# end of file
