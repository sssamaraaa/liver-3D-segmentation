import os
import tempfile
import shutil
import zipfile
import tarfile
from dicom2nifti import convert_dicom
from pathlib import Path


def is_archive(path):
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)

def is_dicom_folder(path):
    files = os.listdir(path)
    return any(f.lower().endswith(".dcm") for f in files) or len(files) > 1

def convert_dcm_to_nifti(inpt):
    input_path = Path(inpt)

    # create temp directory for conversion 
    tmp_dir = tempfile.mkdtemp()
    dcm_dir = os.path.join(tmp_dir, "dcm")
    os.makedirs(dcm_dir, exist_ok=True)

    # if it's an archive — extract 
    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(dcm_dir)

    elif tarfile.is_tarfile(input_path):
        with tarfile.open(input_path, "r:*") as tar:
            tar.extractall(dcm_dir)

    elif input_path.is_dir():
        # copy directory to prevent accidental modification
        shutil.copytree(input_path, dcm_dir, dirs_exist_ok=True)

    else:
        raise ValueError(f"Input '{input_path}' is not a folder or archive")

    # path for result 
    output_path = os.path.join(tmp_dir, "converted.nii.gz")

    # convert
    try:
        convert_dicom.dicom_series_to_nifti(
            dcm_dir, output_path, reorient_nifti=False
        )
    except Exception as e:
        raise e 

    if not os.path.exists(output_path):
        raise RuntimeError("Conversion failed: no NIfTI output produced")

    return output_path
