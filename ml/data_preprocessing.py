import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
from glob import glob


def resample_to_spacing(volume, orig_spacing, new_spacing, order=1):
    """Resample the 3D volume to a new isotropic spacing."""
    factors = [o / n for o, n in zip(orig_spacing, new_spacing)]
    vol_rs = zoom(volume, factors, order=order)
    return vol_rs

def intensity_clip_normalize(volume, clip_min=-200, clip_max=250):
    """Clip HU values and normalize to [0,1]."""
    vol = np.clip(volume, clip_min, clip_max)
    vol = (vol - clip_min) / (clip_max - clip_min)
    return vol.astype(np.float32)

def to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def preprocess_case(img_path, mask_path, out_img_dir, out_mask_dir, new_spacing):
    """Preprocess a single case: load, resample, clip, save, and record metadata."""
    img_obj = nib.load(img_path)
    msk_obj = nib.load(mask_path)

    img = np.asarray(img_obj.dataobj, dtype=np.float32)
    msk = np.asarray(msk_obj.dataobj, dtype=np.uint8)

    orig_spacing = img_obj.header.get_zooms()
    affine = img_obj.affine

    # resample
    img = resample_to_spacing(img, orig_spacing, new_spacing, order=1)
    msk = resample_to_spacing(msk, orig_spacing, new_spacing, order=0)

    # clip + normalize
    img = intensity_clip_normalize(img, -200, 250)

    case_id = os.path.basename(img_path).replace(".nii.gz", "")
    np.save(os.path.join(out_img_dir, f"{case_id}.npy"), img)
    np.save(os.path.join(out_mask_dir, f"{case_id}.npy"), msk)

    return {
        "case_id": case_id,
        "orig_spacing": tuple(float(s) for s in orig_spacing),
        "new_spacing": tuple(float(s) for s in new_spacing),
        "orig_shape": tuple(int(x) for x in msk_obj.shape),
        "new_shape": tuple(int(x) for x in img.shape),
        "affine": affine.tolist(),
        "image_path": os.path.join(out_img_dir, f"{case_id}.npy"),
        "mask_path": os.path.join(out_mask_dir, f"{case_id}.npy"),
        "hu_clip": {"min": -200, "max": 250}
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Path to original data")
    p.add_argument("--output_dir", type=str, default="./preprocessed", help="Path to save results")
    p.add_argument("--spacing", nargs=3, type=float, default=[1.5, 1.5, 1.5], help="Target isotropic spacing (mm)")
    args = p.parse_args()

    out_img_dir = os.path.join(args.output_dir, "imagesTr_npy")
    out_mask_dir = os.path.join(args.output_dir, "labelsTr_npy")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(args.data_dir, "imagesTr", "*.nii*")))
    mask_paths = sorted(glob(os.path.join(args.data_dir, "labelsTr", "*.nii*")))
    assert len(image_paths) == len(mask_paths), "Images and labels count mismatch!"

    metadata = []
    for img_path, mask_path in tqdm(list(zip(image_paths, mask_paths)), desc="Preprocessing"):
        info = preprocess_case(img_path, mask_path, out_img_dir, out_mask_dir, args.spacing)
        metadata.append(info)

    # save metadata safely
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=to_serializable)

    print(f"\nPreprocessing complete: {len(metadata)} cases saved to {args.output_dir}")


if __name__ == "__main__":
    main()
