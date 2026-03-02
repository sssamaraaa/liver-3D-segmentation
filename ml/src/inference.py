import torch
import numpy as np
import nibabel as nib
import time
from torch.amp import autocast
from .data_preprocessing import resample_to_isotropic, resample_mask_to_original, intensity_clip_normalize, save_mask_nifti
from .utils import load_model_from_checkpoint


def get_start_points(max_size, patch_size, stride):
    if max_size <= patch_size:
        return [0]

    starts = []
    i = 0
    while i + patch_size <= max_size:
        starts.append(i)
        i += stride

    if starts and starts[-1] + patch_size < max_size:
        starts.append(max_size - patch_size)

    return sorted(set(starts))

def sliding_window_inference(volume, model, device, patch_size, stride_factor=0.5, batch_size=4):
    # OP 1: use torch.inference_mode() for faster inference with less overhead
    with torch.inference_mode():
        z_max, y_max, x_max = volume.shape
        dz, dy, dx = patch_size
        sz, sy, sx = [max(1, int(d * stride_factor)) for d in patch_size]

        # OP 2: pre-allocate probability and count maps on CPU (not GPU) to reduce GPU memory pressure
        prob_map = torch.zeros((z_max, y_max, x_max), dtype=torch.float32, device='cpu')
        count_map = torch.zeros((z_max, y_max, x_max), dtype=torch.float32, device='cpu')

        z_starts = get_start_points(z_max, dz, sz)
        y_starts = get_start_points(y_max, dy, sy)  
        x_starts = get_start_points(x_max, dx, sx)

        # OP 3: pre-allocate batch tensor on GPU once to avoid repeated allocations
        use_fp16 = device.type == 'cuda'
        batch_tensor = torch.zeros((batch_size, 1, dz, dy, dx), 
                                 dtype=torch.float16 if use_fp16 else torch.float32,
                                 device=device)

        coords_list = []
        batch_count = 0
        total_patches = 0

        def process_batch(batch_count, coords_list):
            if batch_count == 0:
                return
                
            # OP 4: use mixed precision (autocast) for faster computation on CUDA GPUs
            with autocast(device_type="cuda", enabled=use_fp16):
                logits = model(batch_tensor[:batch_count])
                # OP 5: convert to float32 after sigmoid for accumulation precision
                probs = torch.sigmoid(logits).squeeze(1).float()  
                
            # OP 6: move results to CPU for accumulation to reduce GPU memory usage
            probs_cpu = probs.cpu()
            
            for i, (z0, y0, x0) in enumerate(coords_list):
                prob_slice = probs_cpu[i]
                z1 = min(z0 + dz, z_max)
                y1 = min(y0 + dy, y_max) 
                x1 = min(x0 + dx, x_max)
                
                actual_z = z1 - z0
                actual_y = y1 - y0
                actual_x = x1 - x0
                
                # truncating prob_slice to actual size
                prob_slice_cropped = prob_slice[:actual_z, :actual_y, :actual_x]
                
                # accumulating the results
                prob_map[z0:z1, y0:y1, x0:x1] += prob_slice_cropped
                count_map[z0:z1, y0:y1, x0:x1] += 1

        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = volume[z:z+dz, y:y+dy, x:x+dx]
                    
                    if patch.shape != (dz, dy, dx):
                        pad_z = dz - patch.shape[0]
                        pad_y = dy - patch.shape[1]
                        pad_x = dx - patch.shape[2]
                        patch = np.pad(patch, 
                                     ((0, pad_z), (0, pad_y), (0, pad_x)), 
                                     mode='constant', constant_values=0)
                    
                    patch_tensor = torch.from_numpy(patch.astype(np.float32))
                    # OP 7: convert to half precision for GPU 
                    if use_fp16:
                        patch_tensor = patch_tensor.half()
                    
                    batch_tensor[batch_count, 0] = patch_tensor
                    coords_list.append((z, y, x))
                    batch_count += 1
                    total_patches += 1

                    if batch_count == batch_size:
                        process_batch(batch_count, coords_list)
                        batch_count = 0
                        coords_list = []

        process_batch(batch_count, coords_list)
        
        prob_map = prob_map / torch.clamp(count_map, min=1e-8)
        return prob_map.numpy()

def inference(nifti_path, device, save_path, new_spacing=[1.5, 1.5, 1.5], patch_size=[80, 160, 160], stride_factor=0.5, batch_size=8, clip=[-200, 250], threshold=0.45, checkpoint_path=None):
    # ensure we have a model
    temp_model_loaded = False

    if checkpoint_path is None:
        raise ValueError("Either `checkpoint_path` must be provided.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_checkpoint(checkpoint_path)

    # load nifti and preprocess 
    img = nib.load(nifti_path)
    orig_affine = img.affine
    orig_header = img.header.copy()

    vol_rs, _, affine_rs = resample_to_isotropic(img, new_spacing=new_spacing, order=1)
    vol_rs = intensity_clip_normalize(vol_rs, clip_min=float(clip[0]), clip_max=float(clip[1]))

    # run sliding-window inference 
    prob = sliding_window_inference(
        vol_rs,
        model,
        torch.device("cuda" if device == "cuda" or device == "cuda:0" else "cpu"),
        patch_size=patch_size,
        stride_factor=stride_factor,
        batch_size=batch_size,
    )

    # threshold and postproccess
    mask_new_spacing = (prob >= threshold).astype(np.uint8)
    mask_orig = resample_mask_to_original(mask_new_spacing, affine_rs, img)

    # if needed
    if save_path:
        save_mask_nifti(mask_orig, orig_affine, orig_header, save_path)

    # if we have temporarily uploaded the model
    if temp_model_loaded:
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    return mask_orig.astype(np.uint8) 

def main():
    return 

if __name__ == "__main__":
    main()