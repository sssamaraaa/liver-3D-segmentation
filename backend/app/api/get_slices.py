from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import nibabel as nib
import numpy as np
import cv2
import base64
import os


router_slices = APIRouter()

class OverlayRequest(BaseModel):
    ct_path: str
    mask_path: str
    axis: str
    window: int = 400
    level: int = 40
    alpha: float = 0.4

def window_ct(ct, window, level):
    low = level - window / 2
    high = level + window / 2
    ct = np.clip(ct, low, high)
    ct = (ct - low) / (high - low)
    return (ct * 255).astype(np.uint8)

@router_slices.post("/overlay")
def overlay_slices(req: OverlayRequest):
    ct_path = req.ct_path.lstrip("/")
    mask_path = req.mask_path.lstrip("/")

    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        raise HTTPException(404, "CT or mask not found")

    ct = nib.load(ct_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    if ct.shape != mask.shape:
        raise HTTPException(
            400,
            f"Shape mismatch: CT {ct.shape}, mask {mask.shape}",
        )

    if req.axis == "axial":
        ct = np.transpose(ct, (2, 1, 0))
        mask = np.transpose(mask, (2, 1, 0))
    elif req.axis == "coronal":
        ct = np.transpose(ct, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))
    elif req.axis == "sagittal":
        ct = np.transpose(ct, (0, 2, 1))
        mask = np.transpose(mask, (0, 2, 1))
    else:
        raise HTTPException(400, "Invalid axis")

    images = []

    for i in range(ct.shape[0]):
        ct_slice = window_ct(ct[i], req.window, req.level)

        rgb = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)

        mask_bool = mask[i] > 0
        overlay_color = np.zeros_like(rgb)
        overlay_color[..., 0] = 255  

        rgb[mask_bool] = (
            (1 - req.alpha) * rgb[mask_bool]
            + req.alpha * overlay_color[mask_bool]
        ).astype(np.uint8)

        _, png = cv2.imencode(".png", rgb)
        images.append(base64.b64encode(png).decode())

    return {
        "count": len(images),
        "slices": images,
    }
