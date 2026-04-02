import sys
import os
import logging


def setup_logging():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logs_file_path = os.path.join(log_dir, "service_logs.log")
    errors_file_path = os.path.join(log_dir, "errros.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_file_path),  
            logging.StreamHandler(sys.stdout)            
        ],
        force=True
    )

    error_handler = logging.FileHandler(errors_file_path)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logging.getLogger().addHandler(error_handler)