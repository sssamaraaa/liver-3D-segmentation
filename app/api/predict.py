import tempfile
import os
import logging
from fastapi import UploadFile, APIRouter, Request


router_predict = APIRouter()
logger = logging.getLogger(__name__)

@router_predict.post("/predict")
async def predict(file: UploadFile, request: Request):
    content = await file.read()
    model_service = request.app.state.model_service

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        mask = model_service.predict(tmp_file_path)
        logger.info(f"Success! Mask shape: {mask.shape}")
        return {"shape": mask.shape}

    except Exception as e:
        print(e)
        raise e
    finally:
        os.remove(tmp_file_path)