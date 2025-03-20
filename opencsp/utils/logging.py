# opencsp/utils/logging.py
"""
Centralized logging module for openCSP.

This module provides a standardized logging system for the entire openCSP framework,
ensuring consistent formatting and log level control across all components.
"""

import os
import sys
import logging
from typing import Optional, Dict, Union, Any, List
import datetime

# Global variables
_LOGGER = None
_FILE_HANDLER = None
_CONSOLE_HANDLER = None
_LOG_LEVEL = logging.INFO
_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(
    log_file: Optional[str] = None,
    log_level: Union[int, str] = logging.INFO,
    console: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up the global logging configuration.
    
    Args:
        log_file: Path to the log file (default: None, no file logging)
        log_level: Logging level (default: logging.INFO)
        console: Whether to log to console (default: True)
        log_format: Custom log format (default: None, use standard format)
        date_format: Custom date format (default: None, use standard format)
        
    Returns:
        Logger instance
        
    Example:
        >>> from opencsp.utils.logging import setup_logging
        >>> logger = setup_logging(log_file='csp.log', log_level=logging.DEBUG)
    """
    global _LOGGER, _FILE_HANDLER, _CONSOLE_HANDLER, _LOG_LEVEL
    
    # Convert string level to int if needed
    if isinstance(log_level, str):
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        log_level = level_map.get(log_level.lower(), logging.INFO)
    
    _LOG_LEVEL = log_level
    
    # Create root logger if it doesn't exist
    if _LOGGER is None:
        _LOGGER = logging.getLogger('opencsp')
    
    # Reset handlers
    if _LOGGER.handlers:
        for handler in _LOGGER.handlers[:]:
            _LOGGER.removeHandler(handler)
    
    # Set log level
    _LOGGER.setLevel(_LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(
        log_format or _LOG_FORMAT,
        datefmt=date_format or _DATE_FORMAT
    )
    
    # Add file handler if log_file is provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        _FILE_HANDLER = logging.FileHandler(log_file)
        _FILE_HANDLER.setFormatter(formatter)
        _LOGGER.addHandler(_FILE_HANDLER)
    
    # Add console handler if console is True
    if console:
        _CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
        _CONSOLE_HANDLER.setFormatter(formatter)
        _LOGGER.addHandler(_CONSOLE_HANDLER)
    
    # Prevent propagation to root logger
    _LOGGER.propagate = False
    
    return _LOGGER

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default: None, use root opencsp logger)
        
    Returns:
        Logger instance
        
    Example:
        >>> from opencsp.utils.logging import get_logger
        >>> logger = get_logger('opencsp.algorithms.genetic')
        >>> logger.info('Starting genetic algorithm')
    """
    global _LOGGER
    
    # Initialize root logger if it doesn't exist
    if _LOGGER is None:
        setup_logging()
    
    # Return root logger if name is None or 'opencsp'
    if name is None or name == 'opencsp':
        return _LOGGER
    
    # Get child logger
    logger = logging.getLogger(name)
    
    # Ensure child logger inherits the root logger's handlers and level
    if not logger.handlers:
        logger.handlers = _LOGGER.handlers
        logger.setLevel(_LOGGER.level)
        logger.propagate = False
    
    return logger

def set_log_level(level: Union[int, str]) -> None:
    """
    Set the logging level for all loggers.
    
    Args:
        level: Logging level (int or string like 'debug', 'info', etc.)
        
    Example:
        >>> from opencsp.utils.logging import set_log_level
        >>> set_log_level('debug')  # Set to debug level
    """
    global _LOGGER, _LOG_LEVEL
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        level = level_map.get(level.lower(), logging.INFO)
    
    _LOG_LEVEL = level
    
    # Set level for root logger
    if _LOGGER:
        _LOGGER.setLevel(level)
        
        # Set level for all handlers
        for handler in _LOGGER.handlers:
            handler.setLevel(level)

def add_file_handler(log_file: str) -> None:
    """
    Add a file handler to the root logger.
    
    Args:
        log_file: Path to the log file
        
    Example:
        >>> from opencsp.utils.logging import add_file_handler
        >>> add_file_handler('debug.log')
    """
    global _LOGGER, _FILE_HANDLER
    
    if _LOGGER is None:
        setup_logging()
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    
    # Create and add file handler
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.setLevel(_LOG_LEVEL)
    _LOGGER.addHandler(handler)
    
    # Update global file handler
    _FILE_HANDLER = handler

def log_dict(data: Dict[str, Any], level: int = logging.INFO, logger_name: Optional[str] = None) -> None:
    """
    Log a dictionary in a formatted way.
    
    Args:
        data: Dictionary to log
        level: Logging level (default: logging.INFO)
        logger_name: Logger name (default: None, use root logger)
        
    Example:
        >>> from opencsp.utils.logging import log_dict
        >>> config = {'optimizer': 'ga', 'population_size': 50}
        >>> log_dict(config, level=logging.DEBUG)
    """
    import json
    
    logger = get_logger(logger_name)
    formatted_data = json.dumps(data, indent=2, default=str)
    logger.log(level, f"Data: \n{formatted_data}")

def create_run_logger(run_id: Optional[str] = None, output_dir: str = './logs') -> logging.Logger:
    """
    Create a logger for a specific optimization run.
    
    This creates a logger with both console and file output, where the file
    is named based on the run ID and timestamp.
    
    Args:
        run_id: Run identifier (default: None, auto-generate)
        output_dir: Directory for log files (default: './logs')
        
    Returns:
        Logger instance
        
    Example:
        >>> from opencsp.utils.logging import create_run_logger
        >>> logger = create_run_logger(run_id='ga_run1', output_dir='results/logs')
    """
    # Generate run ID if not provided
    if run_id is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"run_{timestamp}"
    
    # Create log directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file path
    log_file = os.path.join(output_dir, f"{run_id}.log")
    
    # Set up logging
    logger = setup_logging(
        log_file=log_file,
        log_level=_LOG_LEVEL,
        console=True
    )
    
    logger.info(f"Started logging for run {run_id}")
    return logger

# Initialize default logger
setup_logging()
