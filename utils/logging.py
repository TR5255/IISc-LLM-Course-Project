import logging
import sys

def setup_logger(name: str = "smart_ai_router", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a consistent console logger for research traces."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
