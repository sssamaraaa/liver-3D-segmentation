from fastapi import FastAPI
from ml.src.model_loader import load_model
from backend.app.api.predict import router_predict
from backend.app.api.mesh import router_mesh
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="Liver Segmentation Service",
    version="0.1"
)

app.mount("/storage", StaticFiles(directory="storage"), name="storage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite dev server
        "http://localhost",           # Nginx
        "http://127.0.0.1:5173",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CKPT = "ml\\src\\results_ft\\run2\\Unet3D_dice97(after_ft).pth"
model, model_info = load_model(MODEL_CKPT, device="cuda", base_filters=16, use_fp16_on_gpu=True)

app.state.model = model
app.state.model_info = model_info
app.include_router(router_predict, prefix="/segmentation")
app.include_router(router_mesh, prefix="/mesh")

