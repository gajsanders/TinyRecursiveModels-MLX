import logging


def log_msg(msg, level="INFO"):
    """
    Log a message with the specified level using Python's logging module.
    
    Args:
        msg (str): The message to log
        level (str): The logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
                     Defaults to "INFO".
    """
    # Convert string level to logging constant
    level = level.upper()
    level_num = getattr(logging, level, logging.INFO)  # Default to INFO if level is invalid
    
    # Configure logging if not already configured
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get logger
    logger = logging.getLogger(__name__)
    
    # Log the message with the specified level
    logger.log(level_num, msg)


# Convenience functions for specific log levels
def debug(msg):
    """Log a DEBUG message."""
    log_msg(msg, "DEBUG")


def info(msg):
    """Log an INFO message."""
    log_msg(msg, "INFO")


def warning(msg):
    """Log a WARNING message."""
    log_msg(msg, "WARNING")


def error(msg):
    """Log an ERROR message."""
    log_msg(msg, "ERROR")


def critical(msg):
    """Log a CRITICAL message."""
    log_msg(msg, "CRITICAL")