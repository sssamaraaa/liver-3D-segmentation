import os
import tempfile as temp
from fastapi import APIRouter, Request, UploadFile, File
from backend.app.services.inference_service import run_inference


router = APIRouter()

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    tmp = temp.mkdtemp()
    input_path = os.path.join(tmp, file.filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    model = request.app.state.model  
    model_info = request.app.state.model_info
    result = run_inference(input_path, model=model, device=model_info["device"], output_dir=tmp)

    return result
