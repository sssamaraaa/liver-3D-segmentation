import os
import time
from datetime import datetime, timedelta
from threading import Thread
from fastapi import FastAPI
from ml.src.model_loader import load_model
from backend.app.api.predict import router_predict
from backend.app.api.mesh import router_mesh
from backend.app.api.get_slices import router_slices
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(
    title="Liver Segmentation Service",
    version="0.1"
)

app.mount("/storage", StaticFiles(directory="storage"), name="storage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"      # Vite dev server
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

STORAGE_DIR = os.getenv("STORAGE_DIR")
EXISTENCE_SECONDS = os.getenv("EXISTENCE_SECONDS")
MODEL_PATH = os.getenv("MODEL_PATH")

if not all([STORAGE_DIR, MODEL_PATH]):
    missing = []
    if not STORAGE_DIR: missing.append("STORAGE_DIR")
    if not MODEL_PATH: missing.append("MODEL_PATH")
    if not EXISTENCE_SECONDS: missing.append("EXISTENCE_SECONDS")
    raise ValueError(f"Missing required environment variables: {missing}")

def cleanup_storage():
    while True:
        now = datetime.now()
        for fname in os.listdir(STORAGE_DIR):
            fpath = os.path.join(STORAGE_DIR, fname)
            if os.path.isfile(fpath):
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                if now - mtime > timedelta(seconds=EXISTENCE_SECONDS):
                    try:
                        os.remove(fpath)
                        print(f"Deleted {fpath}")
                    except Exception as e:
                        print(f"Error deleting {fpath}: {e}")
        time.sleep(60)

Thread(target=cleanup_storage, daemon=True).start()

model, model_info = load_model(MODEL_PATH, device="cuda", base_filters=16, use_fp16_on_gpu=True)

app.state.model = model
app.state.model_info = model_info
app.include_router(router_predict, prefix="/segmentation")
app.include_router(router_mesh, prefix="/mesh")
app.include_router(router_slices, prefix="/slices")

