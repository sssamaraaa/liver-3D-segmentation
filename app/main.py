import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.predict import router_predict
from app.services.model_service import ModelService 


def setup_logging(root: str):
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "logs.log")
    error_file = os.path.join(log_dir, "errors.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  
            logging.StreamHandler(sys.stdout)            
        ],
        force=True
    )
    
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    logging.getLogger().addHandler(error_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    setup_logging("app")
    logger = logging.getLogger(__name__)
    logger.info(f"Application starting...")
    WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")
    DEVICE = os.getenv("DEVICE")
    logger.info(f"Loading model")

    try:
        model_service = ModelService(WEIGHTS_PATH)
        model_service.set_device(DEVICE)
        model_service.create_model()
        model_service.load_weights()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e
    
    logger.info(f"Model loaded successfully")
    app.state.model_service = model_service
    yield
    logger.info(f"Application shutting down...")


app = FastAPI(title="Liver Segmentation v1.0", lifespan=lifespan)
app.include_router(router_predict)