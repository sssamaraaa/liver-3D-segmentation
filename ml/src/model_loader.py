import os
import torch
import time
from .model import UNet3D


def load_checkpoint(model, ckpt_path, device):
    start = time.time()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Checkpoint loaded in {time.time() - start:.2f}s")
    
    if isinstance(ck, dict):
        state_dict = ck.get("model_state", ck.get("state_dict", ck.get("model", ck)))
    else:
        state_dict = ck
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # OP : handle DataParallel model prefixes for compatibility
        if k.startswith('module.'):  
            k = k[7:]
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

def load_model(ckpt_path, device, base_filters, use_fp16_on_gpu,):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_t = torch.device(device)

    model = UNet3D(in_ch=1, out_ch=1, base_filters=base_filters)
    model = load_checkpoint(model, ckpt_path, device_t)
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