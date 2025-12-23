from ml.src.inference import inference
import os
from datetime import datetime


def run_inference(nifti_path, model, device, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_filename = f"mask_{timestamp}.nii.gz"
    mask_path = os.path.join(output_dir, mask_filename)

    _ = inference(
        nifti_path,
        model=model,
        device=device,
        save_path=mask_path
    )

    abs_mask_path = os.path.abspath(mask_path)
    
    return {
        "ct_path": os.path.abspath(nifti_path),
        "mask_path": abs_mask_path,
        "filename": mask_filename,
        "status": "success"
    }
