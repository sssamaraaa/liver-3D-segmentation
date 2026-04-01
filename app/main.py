import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.logs.logging_conf import setup_logging
from app.api.predict import router_predict
from app.services.model_service import ModelService 


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")
    logging.info(f"Loading model")

    try:
        model_service = ModelService(WEIGHTS_PATH)
        model_service.set_device("cuda")
        model_service.create_model()
        model_service.load_weights()
    except Exception as e:
        logging.info(f"Error loading model: {e}")
        raise e
    
    logging.info(f"Model loaded successfully")

    app.state.model_service = model_service

    yield

app = FastAPI(title="Liver Segmentation v1.0", lifespan=lifespan)

app.include_router(router_predict)