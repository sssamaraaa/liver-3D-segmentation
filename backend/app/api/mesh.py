import os
import tempfile
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.services.mesh_service import build_liver_mesh


router_mesh = APIRouter()

class MeshRequest(BaseModel):
    mask_path: str
    smooth_iter: int = 30
    decimate_ratio: float = 0.5

@router_mesh.post("/build")
def build_mesh(request: MeshRequest):
    if not os.path.exists(request.mask_path):
        raise HTTPException(status_code=404, detail="Mask file not found")

    output_dir = tempfile.mkdtemp(prefix="mesh_")

    try:
        result = build_liver_mesh(
            mask_path=request.mask_path,
            output_dir=output_dir,
            smooth_iter=request.smooth_iter,
            decimate_ratio=request.decimate_ratio
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "metrics": result["metrics"],
        "files": result["outputs"]
    }
