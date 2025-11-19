import os
import torch
import numpy as np
import nibabel as nib
import pydicom
import argparse
from data_preprocessing import resample_to_spacing, intensity_clip_normalize
from scipy.ndimage import zoom
from model import UNet3D
from pydicom.filereader import dcmread
from torch.amp import autocast
from glob import glob


def load_nifti_with_meta(path):
    nii = nib.load(path)
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    spacing = nii.header.get_zooms()  
    affine = nii.affine
    return vol, tuple(spacing), affine

def load_dicom_series(folder):
    """
    Read DICOM series from folder and return (volume, spacing, affine)
    Tries to build a reasonable affine; if unsuccessful, affine will be identity.
    """
    files = sorted(glob(os.path.join(folder, "*")))
    dcm_files = [f for f in files if f.lower().endswith(".dcm") or True]  # include all - pydicom will try to read
    if len(dcm_files) == 0:
        raise ValueError("No files found in DICOM folder")

    # read all dicom slices that can be read
    slices = []
    for f in dcm_files:
        try:
            ds = dcmread(f, force=True)
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
                slices.append(ds)
            else:
                # still append; will try best-effort
                slices.append(ds)
        except Exception as e:
            # ignore unreadable files
            continue

    if len(slices) == 0:
        raise ValueError("No readable DICOM slices found")

    # sort by ImagePositionPatient (z)
    def _pos(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(ipp[2])
        # fallback to InstanceNumber or slice file order
        if hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
        return 0.0

    slices = sorted(slices, key=_pos)

    # stack pixel arrays
    try:
        vol = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    except Exception as e:
        # if shapes mismatch, try converting to a common shape
        arrs = [np.asarray(s.pixel_array, dtype=np.float32) for s in slices]
        max_shape = np.max([a.shape for a in arrs], axis=0)
        padded = []
        for a in arrs:
            pad = [(0, max_shape[0] - a.shape[0]), (0, max_shape[1] - a.shape[1])]
            a_p = np.pad(a, pad, mode="constant", constant_values=0)
            padded.append(a_p)
        vol = np.stack(padded).astype(np.float32)

    # spacing: try to get PixelSpacing and SliceThickness/spacing between slices
    ds0 = slices[0]
    pixel_spacing = None
    try:
        ps = getattr(ds0, "PixelSpacing", None)
        if ps is not None:
            pixel_spacing = (float(ps[0]), float(ps[1]))  # typically (row, col)
    except Exception:
        pixel_spacing = None

    # slice spacing: compute from z positions if possible
    slice_spacings = []
    zs = []
    for s in slices:
        ipp = getattr(s, "ImagePositionPatient", None)
        if ipp is not None:
            zs.append(float(ipp[2]))
    if len(zs) >= 2:
        slice_spacings = np.diff(np.sort(zs))
        slice_spacing = float(np.median(slice_spacings))
    else:
        slice_spacing = getattr(ds0, "SliceThickness", None)
        if slice_spacing is None:
            slice_spacing = 1.0  # fallback

    # construct spacing in (z, y, x) order
    if pixel_spacing is not None:
        spacing = (slice_spacing, float(pixel_spacing[0]), float(pixel_spacing[1]))
    else:
        spacing = (slice_spacing, 1.0, 1.0)

    # build a rough affine matrix using ImageOrientationPatient and ImagePositionPatient
    affine = np.eye(4, dtype=np.float32)
    try:
        iop = getattr(ds0, "ImageOrientationPatient", None)
        ipp = getattr(ds0, "ImagePositionPatient", None)
        if iop is not None and ipp is not None:
            # direction cosines
            row_cosine = np.array(iop[0:3], dtype=float)
            col_cosine = np.array(iop[3:6], dtype=float)
            slice_cosine = np.cross(row_cosine, col_cosine)
            ps = getattr(ds0, "PixelSpacing", [1.0, 1.0])
            px = float(ps[1]); py = float(ps[0])  # column (x), row (y)
            affine[:3, 0] = row_cosine * px
            affine[:3, 1] = col_cosine * py
            # approximate slice spacing magnitude
            affine[:3, 2] = slice_cosine * spacing[0]
            affine[:3, 3] = np.array(ipp, dtype=float)
    except Exception:
        affine = np.eye(4, dtype=np.float32)

    return vol, tuple(spacing), affine

def get_start_points(max_size, patch_size, stride):
    """
    It generates starting indexes so that:
    - a non-empty list is always returned;
    - the last window is guaranteed to cover the right/bottom/back edge (if necessary).
    """
    # if the size is less than or equal to the patch, start from 0
    if max_size <= patch_size:
        return [0]

    starts = []
    i = 0
    # moving forward until the window fits completely
    while i + patch_size <= max_size:
        starts.append(i)
        i += stride

    # if the last window does not capture the right/bottom edge, add precise alignment
    if starts:
        if starts[-1] + patch_size < max_size:
            starts.append(max_size - patch_size)
    else:
        # in case of any strange input data, we return an aligned start
        starts = [max_size - patch_size]

    # remove duplicates (in case the start coincides with an already added one)
    # and return them in sorted order
    starts = sorted(set(starts))
    return starts


def sliding_window_inference(volume, model, device, patch_size, stride_factor=0.5, batch_size=1):
    model.eval()
    with torch.no_grad():
        z_max, y_max, x_max = volume.shape
        dz, dy, dx = patch_size
        # voxel step (int), min 1
        sz, sy, sx = [max(1, int(d * stride_factor)) for d in patch_size]

        prob_map = np.zeros_like(volume, dtype=np.float32)
        count_map = np.zeros_like(volume, dtype=np.float32)

        z_starts = get_start_points(z_max, dz, sz)
        y_starts = get_start_points(y_max, dy, sy)
        x_starts = get_start_points(x_max, dx, sx)

        patches = []
        coords = []

        def process_batch(patches, coords):
            # patches: numpy list [1,1,D,H,W] of the same D,H,W
            batch_np = np.concatenate(patches, axis=0).astype(np.float32)  # shape [B,1,D,H,W]
            patches_t = torch.from_numpy(batch_np).to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(patches_t)  # [B, 1, D, H, W]
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # [B, D, H, W]

            for p_idx, (z0, y0, x0) in enumerate(coords):
                p = probs[p_idx]
                # we take into account that the edge may have exactly the same dimensions, but just in case, we will truncate
                z1 = min(z0 + dz, z_max)
                y1 = min(y0 + dy, y_max)
                x1 = min(x0 + dx, x_max)
                p = p[: (z1 - z0), : (y1 - y0), : (x1 - x0)]
                prob_map[z0:z1, y0:y1, x0:x1] += p
                count_map[z0:z1, y0:y1, x0:x1] += 1

        # we form patches strictly according to patch_size — get_start_points guarantees this
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    # extracting a patch of the required size
                    z1 = z + dz
                    y1 = y + dy
                    x1 = x + dx
                    patch = volume[z:z1, y:y1, x:x1]

                    # just in case, if the patch is smaller for some reason, pad it to the desired size
                    # (this protects against unexpected problems with get_start_points)
                    if patch.shape != (dz, dy, dx):
                        pad_z = dz - patch.shape[0]
                        pad_y = dy - patch.shape[1]
                        pad_x = dx - patch.shape[2]
                        patch = np.pad(patch,
                                       ((0, pad_z), (0, pad_y), (0, pad_x)),
                                       mode='constant', constant_values=0)

                    patches.append(patch[None, None, ...])  # [1,1,D,H,W]
                    coords.append((z, y, x))

                    if len(patches) == batch_size:
                        process_batch(patches, coords)
                        patches, coords = [], []

        if len(patches) > 0:
            process_batch(patches, coords)

        # avoiding division by zero
        prob_map = prob_map / np.maximum(count_map, 1e-8)
        return prob_map

def postprocess_mask_to_orig(mask_pred, orig_spacing, new_spacing, orig_shape):
    # factors to go new_spacing -> orig_spacing: factor = new/old (since resample_to_spacing used orig/new)
    factors = [n / o for o, n in zip(orig_spacing, new_spacing)]
    mask_rs = zoom(mask_pred.astype(np.uint8), factors, order=0)
    # crop/pad to exact orig_shape
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

def load_checkpoint(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict):
        if "model_state" in ck:
            state = ck["model_state"]
        elif "state_dict" in ck:
            state = ck["state_dict"]
        elif "model" in ck:
            state = ck["model"]
        else:
            state = ck
    else:
        state = ck
    model.load_state_dict(state)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--input", required=True, help="Path to input NIfTI (.nii/.nii.gz) or DICOM folder")
    p.add_argument("--output", required=True, help="Path to save output mask (.nii.gz)")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--new_spacing", nargs=3, type=float, default=[1.5, 1.5, 1.5], help="target spacing used for model (z y x)")
    p.add_argument("--patch_size", nargs=3, type=int, default=[64, 128, 128], help="patch size (z y x) used for inference")
    p.add_argument("--stride_factor", type=float, default=0.5, help="stride factor for sliding window")
    p.add_argument("--batch_size", type=int, default=2, help="batch size for sliding window inference")
    p.add_argument("--base_filters", type=int, default=16, help="base filters for UNet3D architecture")
    p.add_argument("--clip", nargs=2, type=float, default=[-200, 250], help="HU clip min and max")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    new_spacing = tuple(args.new_spacing)
    patch_size = tuple(args.patch_size)

    # 1) Load input
    input_path = args.input
    if os.path.isdir(input_path):
        print("Loading DICOM series from folder...")
        vol, orig_spacing, affine = load_dicom_series(input_path)
        orig_shape = vol.shape
        is_nifti = False
    else:
        print("Loading NIfTI...")
        vol, orig_spacing, affine = load_nifti_with_meta(input_path)
        orig_shape = vol.shape
        is_nifti = True

    print(f"Original shape: {orig_shape}, orig_spacing: {orig_spacing}")

    # 2) Preprocess (resample -> clip -> normalize)
    print("Resampling to model spacing:", new_spacing)
    vol_rs = resample_to_spacing(vol, orig_spacing, new_spacing, order=1)
    vol_rs = intensity_clip_normalize(vol_rs, clip_min=float(args.clip[0]), clip_max=float(args.clip[1]))

    # 3) build and load model
    print("Building model and loading checkpoint...")
    model = UNet3D(in_ch=1, out_ch=1, base_filters=args.base_filters)
    model = load_checkpoint(model, args.model, device)
    model.to(device)
    model.eval()

    # 4) Inference
    print("Running sliding-window inference...")
    prob = sliding_window_inference(vol_rs, model, device, patch_size=patch_size, stride_factor=args.stride_factor, batch_size=args.batch_size)

    # 5) Threshold to binary mask
    mask_new_spacing = (prob >= 0.5).astype(np.uint8)

    # 6) Postprocess: resample mask back to original spacing/shape
    print("Postprocessing mask back to original space...")
    mask_orig = postprocess_mask_to_orig(mask_new_spacing, orig_spacing, new_spacing, orig_shape)

    # 7) Save result as NIfTI using original affine if available
    out_path = args.output
    print(f"Saving mask to {out_path} ...")
    save_mask_nifti(mask_orig, affine, out_path)
    print("Done.")

if __name__ == "__main__":
    main()