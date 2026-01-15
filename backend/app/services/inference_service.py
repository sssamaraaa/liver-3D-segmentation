import requests
import os

ML_URL = "http://ml:8000/infer"

def run_inference(nifti_path, output_dir):
    with open(nifti_path, "rb") as f:
        response = requests.post(
            ML_URL,
            files={"file": f}
        )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    result = response.json()

    return {
        "ct_path": os.path.abspath(nifti_path),
        "mask_path": result["mask_path"],
        "filename": os.path.basename(result["mask_path"]),
        "status": "success"
    }
