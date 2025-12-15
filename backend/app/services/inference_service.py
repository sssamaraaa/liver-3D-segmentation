from ml.src.inference import inference
import os


def run_inference(nifti_path, model, device, output_dir):
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    _ = inference(
        nifti_path,
        model=model,
        device=device,
        save_path=mask_path
    )

    return mask_path
