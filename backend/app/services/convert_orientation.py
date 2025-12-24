import os
import nibabel as nib


def convert_orientation_to_canonical(nifti_path, suffix: str = "_canon"):
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(nifti_path)

    img = nib.load(nifti_path)

    if nib.aff2axcodes(img.affine) == ("R", "A", "S"):
        return nifti_path

    canon_img = nib.as_closest_canonical(img)

    base, ext = os.path.splitext(nifti_path)
    if ext == ".gz":
        base, _ = os.path.splitext(base)
        ext = ".nii.gz"

    out_path = f"{base}{suffix}{ext}"

    nib.save(canon_img, out_path)

    return out_path
