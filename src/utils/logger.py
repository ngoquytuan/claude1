"""
Logging utilities for Keystroke GAN Demo
"""

import logging
import os
from datetime import datetime
from config.settings import LOG_PATH, LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str = 'keystroke_gan', log_file: str = None) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Optional custom log file name

    Returns:
        Configured logger instance
    """
    # Create logs directory if not exists
    os.makedirs(LOG_PATH, exist_ok=True)

    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'

    log_file_path = os.path.join(LOG_PATH, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)

    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_file_path}")

    return logger


def get_logger(name: str = 'keystroke_gan') -> logging.Logger:
    """
    Get existing logger or create new one

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger
