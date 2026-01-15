import os
import time
from datetime import datetime, timedelta
from threading import Thread

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from backend.app.api.predict import router_predict
from backend.app.api.mesh import router_mesh
from backend.app.api.get_slices import router_slices

load_dotenv()

app = FastAPI(
    title="Liver Segmentation Service",
    version="0.1"
)

STORAGE_DIR = os.getenv("STORAGE_DIR")
EXISTENCE_SECONDS = os.getenv("EXISTENCE_SECONDS")

if not all([STORAGE_DIR, EXISTENCE_SECONDS]):
    missing = []
    if not STORAGE_DIR:
        missing.append("STORAGE_DIR")
    if not EXISTENCE_SECONDS:
        missing.append("EXISTENCE_SECONDS")
    raise ValueError(f"Missing required environment variables: {missing}")

app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")


def cleanup_storage():
    while True:
        now = datetime.now()
        for fname in os.listdir(STORAGE_DIR):
            fpath = os.path.join(STORAGE_DIR, fname)
            if os.path.isfile(fpath):
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                if now - mtime > timedelta(seconds=int(EXISTENCE_SECONDS)):
                    try:
                        os.remove(fpath)
                        print(f"Deleted {fpath}")
                    except Exception as e:
                        print(f"Error deleting {fpath}: {e}")
        time.sleep(60)


Thread(target=cleanup_storage, daemon=True).start()

app.include_router(router_predict, prefix="/segmentation")
app.include_router(router_mesh, prefix="/mesh")
app.include_router(router_slices, prefix="/slices")
