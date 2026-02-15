import logging
import sys
import os


def setup_logging():
    ml_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(ml_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ml_service.log"),  
            logging.StreamHandler(sys.stdout)            
        ]
    )
    
    error_handler = logging.FileHandler("logs/errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    logging.getLogger().addHandler(error_handler)