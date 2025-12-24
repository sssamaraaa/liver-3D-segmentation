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
    axis: str  # "axial", "coronal", "sagittal"
    window: int = 400
    level: int = 40
    alpha: float = 0.4

def window_ct(ct, window, level):
    low = level - window / 2
    high = level + window / 2
    ct = np.clip(ct, low, high)
    ct = (ct - low) / (high - low)
    return (ct * 255).astype(np.uint8)

def process_slice(ct_slice, mask_slice, axis, alpha, window, level):
    if axis == "axial":
        ct_slice = np.rot90(ct_slice, k=1)
        mask_slice = np.rot90(mask_slice, k=1)
        ct_slice = np.fliplr(ct_slice)
        mask_slice = np.fliplr(mask_slice)
    elif axis == "coronal":
        ct_slice = np.rot90(ct_slice, k=1)
        mask_slice = np.rot90(mask_slice, k=1)
        ct_slice = np.fliplr(ct_slice)
        mask_slice = np.fliplr(mask_slice)
    elif axis == "sagittal":
        ct_slice = np.rot90(ct_slice, k=1)
        mask_slice = np.rot90(mask_slice, k=1)
        ct_slice = np.fliplr(ct_slice)
        mask_slice = np.fliplr(mask_slice)

    ct_windowed = window_ct(ct_slice, window, level)
    rgb = cv2.cvtColor(ct_windowed, cv2.COLOR_GRAY2RGB)

    mask_bool = mask_slice > 0
    overlay_color = np.zeros_like(rgb)
    overlay_color[..., 0] = 255  
    if mask_bool.any():
        rgb[mask_bool] = ((1 - alpha) * rgb[mask_bool] + alpha * overlay_color[mask_bool]).astype(np.uint8)

    return rgb

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

    images = []

    ct_shape = ct.shape  
    
    if req.axis == "axial":
        max_slices = ct_shape[2]  # Ось Z - аксиальные срезы
        slice_dim1 = ct_shape[0]  # Ось X
        slice_dim2 = ct_shape[1]  # Ось Y
    elif req.axis == "coronal":
        max_slices = ct_shape[1]  # Ось Y - корональные срезы
        slice_dim1 = ct_shape[0]  # Ось X
        slice_dim2 = ct_shape[2]  # Ось Z
    elif req.axis == "sagittal":
        max_slices = ct_shape[0]  # Ось X - сагиттальные срезы
        slice_dim1 = ct_shape[1]  # Ось Y
        slice_dim2 = ct_shape[2]  # Ось Z
    else:
        raise HTTPException(400, "Invalid axis. Use 'axial', 'coronal', or 'sagittal'")

    for i in range(max_slices):
        if req.axis == "axial":
            ct_slice = ct[:, :, i]  # (X, Y) срез на позиции i по Z
            mask_slice = mask[:, :, i]
        elif req.axis == "coronal":
            ct_slice = ct[:, i, :]  # (X, Z) срез на позиции i по Y
            mask_slice = mask[:, i, :]
        elif req.axis == "sagittal":
            ct_slice = ct[i, :, :]  # (Y, Z) срез на позиции i по X
            mask_slice = mask[i, :, :]

        if ct_slice.shape != (slice_dim1, slice_dim2):
            print(f"Warning: slice {i} shape {ct_slice.shape}, expected ({slice_dim1}, {slice_dim2})")

        rgb = process_slice(ct_slice, mask_slice, req.axis, req.alpha, req.window, req.level)

        _, png = cv2.imencode(".png", rgb)
        images.append(base64.b64encode(png).decode())

    return {
        "count": len(images),
        "slices": images,
    }