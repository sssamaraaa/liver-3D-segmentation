from fastapi import FastAPI
from ml.src.model_loader import load_model
from backend.app.api.endpoints import router as api_router


app = FastAPI(
    title="Liver Segmentation Service",
    version="0.1"
)

MODEL_CKPT = "ml\\src\\results_ft\\run2\\Unet3D_dice97(after_ft).pth"
model, model_info = load_model(MODEL_CKPT, device="cuda", base_filters=16, use_fp16_on_gpu=True)

app.state.model = model
app.state.model_info = model_info
app.include_router(api_router, prefix="/api")