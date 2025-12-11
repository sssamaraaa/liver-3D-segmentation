import torch
import numpy as np
import nibabel as nib
import argparse
import time
from scipy.ndimage import zoom
from torch.amp import autocast
from .data_preprocessing import resample_to_spacing, intensity_clip_normalize
from .model_loader import load_checkpoint, load_model
from .model import UNet3D


def load_nifti_with_meta(path):
    nii = nib.load(path)
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    spacing = nii.header.get_zooms()  
    affine = nii.affine
    return vol, tuple(spacing), affine

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
    model.eval()
    
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

        print(f"Processing {len(z_starts)}x{len(y_starts)}x{len(x_starts)} = {len(z_starts)*len(y_starts)*len(x_starts)} patches")
        print(f"Stride: z={sz}, y={sy}, x={sx}")

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

        print(f"Processed {total_patches} patches total")
        
        prob_map = prob_map / torch.clamp(count_map, min=1e-8)
        return prob_map.numpy()

def postprocess_mask_to_orig(mask_pred, orig_spacing, new_spacing, orig_shape):
    factors = [n / o for o, n in zip(orig_spacing, new_spacing)]
    mask_rs = zoom(mask_pred.astype(np.uint8), factors, order=0)
    
    if mask_rs.shape != tuple(orig_shape):
        padded = np.zeros(orig_shape, dtype=np.uint8)
        min_z = min(mask_rs.shape[0], orig_shape[0])
        min_y = min(mask_rs.shape[1], orig_shape[1])
        min_x = min(mask_rs.shape[2], orig_shape[2])
        padded[:min_z, :min_y, :min_x] = mask_rs[:min_z, :min_y, :min_x]
        mask_rs = padded
        
    return mask_rs.astype(np.uint8)

def save_mask_nifti(mask, affine, out_path):
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(nii, out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--input", required=True, help="Path to input NIfTI (.nii/.nii.gz) or DICOM folder")
    p.add_argument("--output", required=True, help="Path to save output mask (.nii.gz)")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--new_spacing", nargs=3, type=float, default=[1.5, 1.5, 1.5], help="target spacing used for model (z y x)")
    p.add_argument("--patch_size", nargs=3, type=int, default=[80, 160, 160], help="patch size (z y x) used for inference")
    p.add_argument("--stride_factor", type=float, default=0.5, help="stride factor for sliding window (увеличено для скорости)")
    p.add_argument("--batch_size", type=int, default=4, help="batch size for sliding window inference (увеличено для скорости)")
    p.add_argument("--base_filters", type=int, default=16, help="base filters for UNet3D architecture")
    p.add_argument("--clip", nargs=2, type=float, default=[-200, 250], help="HU clip min and max")
    p.add_argument("--fast_mode", action="store_true", help="Включить агрессивные оптимизации для скорости")
    args = p.parse_args()

    total_start = time.time()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    # OP 9: enable CUDA optimizations for faster computation
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # optimization for convolution
        torch.backends.cuda.matmul.allow_tf32 = True  # fast matmul
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA optimizations enabled")

    new_spacing = tuple(args.new_spacing)
    patch_size = tuple(args.patch_size)

    # OP 10: fast mode configuration for maximum speed
    if args.fast_mode:
        args.stride_factor = 0.5  
        if device.type == 'cuda':
            args.batch_size = 8   
        print("Fast mode enabled")

    # data loading
    load_start = time.time()
    input_path = args.input

    print("Loading NIfTI...")
    vol, orig_spacing, affine = load_nifti_with_meta(input_path)
    orig_shape = vol.shape
    is_nifti = True

    print(f"Original data loaded in {time.time() - load_start:.2f}s")
    print(f"Original shape: {orig_shape}, orig_spacing: {orig_spacing}")

    # prerprocessing
    preprocess_start = time.time()
    print(f"Resampling to model spacing: {new_spacing}")
    vol_rs = resample_to_spacing(vol, orig_spacing, new_spacing, order=1)
    vol_rs = intensity_clip_normalize(vol_rs, clip_min=float(args.clip[0]), clip_max=float(args.clip[1]))
    print(f"Preprocessing completed in {time.time() - preprocess_start:.2f}s")
    print(f"Resampled shape: {vol_rs.shape}")

    # model loading
    model_start = time.time()
    print("Building model and loading checkpoint (fast)...")
    model = UNet3D(in_ch=1, out_ch=1, base_filters=args.base_filters)
    model = load_checkpoint(model, args.model, device)
    model.to(device)
    
    # OP 11: use half precision (FP16) for the entire model on GPU
    if device.type == 'cuda':
        model = model.half()
        print("Using FP16 precision")
    
    model.eval()
    print(f"Model loaded in {time.time() - model_start:.2f}s")

    # inference
    inference_start = time.time()
    print("Running optimized sliding-window inference...")
    print(f"Batch size: {args.batch_size}, Stride factor: {args.stride_factor}")
    
    prob = sliding_window_inference(
        vol_rs, model, device, 
        patch_size=patch_size, 
        stride_factor=args.stride_factor, 
        batch_size=args.batch_size
    )
    
    inference_time = time.time() - inference_start
    print(f"Inference completed in {inference_time:.2f}s")

    # postprocessing
    postprocess_start = time.time()
    mask_new_spacing = (prob >= 0.5).astype(np.uint8)

    print("Postprocessing mask back to original space...")
    mask_orig = postprocess_mask_to_orig(mask_new_spacing, orig_spacing, new_spacing, orig_shape)
    print(f"Postprocessing completed in {time.time() - postprocess_start:.2f}s")

    # saving results
    save_start = time.time()
    out_path = args.output
    print(f"Saving mask to {out_path} ...")
    save_mask_nifti(mask_orig, affine, out_path)
    print(f"Mask saved in {time.time() - save_start:.2f}s")

    total_time = time.time() - total_start
    print(f"\n=== TOTAL PROCESSING TIME: {total_time:.2f}s ===")
    print(f"Breakdown:")
    print(f"  - Data loading: {load_start - total_start:.2f}s")
    print(f"  - Preprocessing: {preprocess_start - load_start:.2f}s") 
    print(f"  - Model loading: {model_start - preprocess_start:.2f}s")
    print(f"  - Inference: {inference_time:.2f}s")
    print(f"  - Postprocessing: {time.time() - inference_start - inference_time:.2f}s")

def inference(nifti_path, model, device, save_path, new_spacing=[1.5, 1.5, 1.5], patch_size=[80, 160, 160], stride_factor=0.5, batch_size=8, clip=[-200, 250], threshold=0.45, checkpoint_path=None):
    # ensure we have a model
    temp_model_loaded = False
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either `model` or `checkpoint_path` must be provided.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_tmp, _ = load_model(checkpoint_path, device=device)
        model = model_tmp
        temp_model_loaded = True
    else:
        try:
            device = next(model.parameters()).device.type
        except StopIteration:
            device = "cpu"

    # load nifti and preprocess 
    vol, orig_spacing, affine = load_nifti_with_meta(nifti_path)
    orig_shape = vol.shape

    vol_rs = resample_to_spacing(vol, orig_spacing, new_spacing, order=1)
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
    mask_orig = postprocess_mask_to_orig(mask_new_spacing, orig_spacing, new_spacing, orig_shape)

    # if needed
    if save_path:
        save_mask_nifti(mask_orig, affine, save_path)

    # if we have temporarily uploaded the model
    if temp_model_loaded:
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    return mask_orig.astype(np.uint8)

if __name__ == "__main__":
    main()