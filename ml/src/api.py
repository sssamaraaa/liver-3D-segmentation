import os
import tempfile
import torch
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from ml.src.logging_conf import setup_logging
from ml.src.utils import load_model_from_checkpoint
from ml.src.inference import inference
from dotenv import load_dotenv

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH")
STORAGE_DIR = os.getenv("STORAGE_DIR")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}...")

    app.state.model = load_model_from_checkpoint(MODEL_PATH, device=DEVICE)
    app.state.device = DEVICE
    app.state.model.eval()

    logger.info(f"Model loaded successfully on {DEVICE} with dtype {next(app.state.model.parameters()).dtype}")
    yield
    logger.info("Shutting down ML service...")

app = FastAPI(lifespan=lifespan)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(await file.read())
        nifti_path = tmp.name

    try:
        os.makedirs(STORAGE_DIR, exist_ok=True)
        output_filename = f"{Path(file.filename).stem}_mask.nii.gz"
        output_path = os.path.join(STORAGE_DIR, output_filename)

        inference(
            nifti_path,
            model=app.state.model,
            device=app.state.device,
            save_path=output_path,
            checkpoint_path=None
        )

        return {
            "mask_path": output_path,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        try:
            os.remove(nifti_path)
        except:
            pass
