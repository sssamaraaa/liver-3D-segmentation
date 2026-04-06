import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.logging_conf import setup_logging
from app.api.predict import router_predict
from app.services.model_service import ModelService 


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    setup_logging()
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