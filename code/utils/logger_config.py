import logging
import os
from datetime import datetime
from config import get_path

def setup_logger(name):
    """
    Creates a logger instance with consistent configuration.
    
    Parameters:
    - name (str): The name of the logger (usually __name__ from the calling module)
    
    Returns:
    - logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create logger with module name
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added yet
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler with timestamp in filename
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(logs_dir, f'comfyui_{current_time}.log')
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger 