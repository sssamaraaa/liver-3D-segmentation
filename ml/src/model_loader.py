import os
import torch
from model import UNet3D
from inference import load_checkpoint_fast


def load_model(ckpt_path, device, base_filters, use_fp16_on_gpu,):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_t = torch.device(device)

    model = UNet3D(in_ch=1, out_ch=1, base_filters=base_filters)
    model = load_checkpoint_fast(model, ckpt_path, device_t)
    model.to(device_t)

    if device_t.type == "cuda" and use_fp16_on_gpu:
        try:
            model.half()
        except Exception:
            pass

    model.eval()

    info = {
        "device": str(device_t),
        "fp16": (device_t.type == "cuda" and use_fp16_on_gpu),
        "ckpt": os.path.basename(ckpt_path),
        "base_filters": base_filters,
    }
    return model, info