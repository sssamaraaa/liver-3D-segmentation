from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import tempfile
from src.model_loader import load_model
from src.inference import inference

MODEL_PATH = "ml/model/unet.pth"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.model_info = load_model(MODEL_PATH)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(await file.read())
        nifti_path = tmp.name

    try:
        output_path = nifti_path.replace(".nii.gz", "_mask.nii.gz")

        inference(
            nifti_path,
            model=app.state.model,
            device=app.state.model_info["device"],
            save_path=output_path
        )

        return {
            "mask_path": output_path,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
