from fastapi import APIRouter, UploadFile, File


router_mesh = APIRouter()

@router_mesh.post("/build_mesh")
async def mesh():
    return