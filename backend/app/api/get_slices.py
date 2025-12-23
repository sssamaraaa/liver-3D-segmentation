from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import nibabel as nib
import numpy as np
import base64
import cv2
import os


router_slices = APIRouter()

class SliceRequest(BaseModel):
    mask_path: str      
    axis: str           

def slice_stack_to_base64(stack: np.ndarray):
    images = []

    for i in range(stack.shape[0]):
        slice_ = stack[i].astype(np.float32)

        min_val = slice_.min()
        range_val = np.ptp(slice_) + 1e-6

        slice_ = (slice_ - min_val) / range_val
        slice_ = (slice_ * 255).astype(np.uint8)

        png = cv2.imencode(".png", slice_)[1]
        b64 = base64.b64encode(png).decode("utf-8")
        images.append(b64)

    return images

@router_slices.post("/stack")
def get_slice_stack(req: SliceRequest):
    path = req.mask_path.lstrip("/")

    if not os.path.exists(path):
        raise HTTPException(404, "Mask file not found")

    nii = nib.load(path)
    data = nii.get_fdata()

    data = (data > 0).astype(np.uint8)

    if req.axis == "axial":
        stack = np.transpose(data, (2, 1, 0))
    elif req.axis == "coronal":
        stack = np.transpose(data, (1, 2, 0))
    elif req.axis == "sagittal":
        stack = np.transpose(data, (0, 2, 1))
    else:
        raise HTTPException(400, "Invalid axis")

    images = slice_stack_to_base64(stack)

    return {
        "count": len(images),
        "slices": images
    }
