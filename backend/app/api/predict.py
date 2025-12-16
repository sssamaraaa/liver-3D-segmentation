import os
from fastapi import APIRouter, Request, UploadFile, File
from fastapi import HTTPException
from backend.app.services.inference_service import run_inference
from backend.app.services.convert_dcm import convert_dcm_to_nifti, is_dicom_folder, is_archive


router_predict = APIRouter()

@router_predict.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    input_path = os.path.join("storage", file.filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    to_infer = input_path

    # DICOM archive
    if is_archive(input_path):
        to_infer = convert_dcm_to_nifti(input_path)

    # DICOM folder
    elif os.path.isdir(input_path) and is_dicom_folder(input_path):
        to_infer = convert_dcm_to_nifti(input_path)

    elif not (input_path.endswith(".nii") or input_path.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported input format. Upload DICOM, zip/tar with DICOM, or NIfTI (.nii/.nii.gz)."
        )

    model = request.app.state.model  
    model_info = request.app.state.model_info

    result = run_inference(
        to_infer, 
        model=model, 
        device=model_info["device"], 
        output_dir="storage"
    )

    return result
