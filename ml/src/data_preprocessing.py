import os
import json
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm
from nibabel.processing import resample_to_output, resample_from_to
from glob import glob

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Path to original data")
    p.add_argument("--output_dir", type=str, default="./preprocessed", help="Path to save results")
    p.add_argument("--spacing", nargs=3, type=float, default=[1.5, 1.5, 1.5], help="Target isotropic spacing (mm)")
    return p.parse_args()

def load_nifti_pair(img_path, mask_path):
    img_obj = nib.load(img_path)
    mask_obj = nib.load(mask_path)
    return img_obj, mask_obj

def save_mask_nifti(mask, affine, orig_header, out_path):
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine, orig_header)
    nii.header['xyzt_units'] = orig_header['xyzt_units']
    nii.set_data_dtype(np.uint8)
    nib.save(nii, out_path)

def resample_to_isotropic(nifti_image, new_spacing=(1.5, 1.5, 1.5), order=1):
    img_rs = resample_to_output(
        nifti_image,
        voxel_sizes=new_spacing,
        order=order,
        mode='constant',
        cval=0
    )

    volume = img_rs.get_fdata(dtype=np.float32)
    spacing = img_rs.header.get_zooms()[:3]
    affine = img_rs.affine

    return volume, spacing, affine

def resample_mask_to_original(mask_rs, rs_affine, orig_img):
    mask_img = nib.Nifti1Image(mask_rs.astype(np.uint8), rs_affine)

    mask_orig_img = resample_from_to(
        mask_img,
        orig_img,
        order=0
    )

    return mask_orig_img.get_fdata().astype(np.uint8)

def intensity_clip_normalize(volume, clip_min=-200, clip_max=250):
    """Clip HU values and normalize to [0,1]"""
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - clip_min) / (clip_max - clip_min)
    return volume.astype(np.float32)

def preprocess_case(img_obj, mask_obj, case_id, out_img_dir, out_mask_dir, new_spacing):
    """Function performs offline preprocessing to avoid CPU/IO bottlenecks during training."""
    orig_spacing = img_obj.header.get_zooms()
    orig_affine = img_obj.affine

    img, _, new_affine = resample_to_isotropic(img_obj, new_spacing, order=1)
    mask, _, _ = resample_to_isotropic(mask_obj, new_spacing, order=0)

    img = intensity_clip_normalize(img, -200, 250)

    np.save(os.path.join(out_img_dir, f"{case_id}.npy"), img)
    np.save(os.path.join(out_mask_dir, f"{case_id}.npy"), mask)

    return {
        "case_id": case_id,
        "orig_spacing": tuple(float(s) for s in orig_spacing),
        "new_spacing": tuple(float(s) for s in new_spacing),
        "orig_shape": tuple(int(x) for x in mask_obj.shape),
        "new_shape": tuple(int(x) for x in img.shape),
        "orig_affine": orig_affine.tolist(),
        "new_affine": new_affine.tolist(),
        "image_path": os.path.join(out_img_dir, f"{case_id}.npy"),
        "mask_path": os.path.join(out_mask_dir, f"{case_id}.npy"),
        "hu_clip": {"min": -200, "max": 250}
    }

def main(args):
    out_images_dir = os.path.join(args.output_dir, "imagesTr_npy")
    out_masks_dir = os.path.join(args.output_dir, "labelsTr_npy")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    images_paths = sorted(glob(os.path.join(args.data_dir, "imagesTr_npy", "*.nii*")))
    masks_paths = sorted(glob(os.path.join(args.data_dir, "labelsTr_npy", "*.nii*")))
    assert len(images_paths) == len(masks_paths), "Images and masks count mismatch!"

    metadata = []
    for img_path, mask_path in tqdm(zip(images_paths, masks_paths), desc="Preprocessing", total=len(images_paths)):
        case_id = os.path.basename(img_path).replace(".nii.gz", "")
        img_obj, mask_obj = load_nifti_pair(img_path, mask_path)
        info = preprocess_case(img_obj, mask_obj, case_id, out_images_dir, out_masks_dir, args.spacing)
        metadata.append(info)

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nPreprocessing complete: {len(metadata)} cases saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
