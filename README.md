# liver-3D-segmentation

End-to-end **3D liver segmentation pipeline for CT scans**.

The project includes model training on the public **Medical Segmentation Decathlon (MSD)** dataset, evaluation on private customer data using **4-fold cross-validation**, experiment tracking with **MLflow**, configuration management with **Hydra**, and deployment through a lightweight **FastAPI** service with **Docker and CUDA support**.

## Dataset Information 
### Medical Segmentation Decathlon (MSD)
- **Dataset:** Medical Segmentation Decathlon - Liver Task 
- **Task:** Liver and liver tumor segmentation from CT scans 
- **License:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) 
- **Release:** Version 1.0 (May 4, 2018) - **Tensor Image Size:** 3D 
- **Modality:** CT (Computed Tomography) ### Authors **MSD Consortium:** The Medical Segmentation Decathlon was organized by a large international team of researchers and clinicians. For the complete list of authors and contributors, please refer to the [original publication](https://www.nature.com/articles/s41467-022-30695-9#author-information) and the [Medical Segmentation Decathlon website](http://medicaldecathlon.com/).

### Private Customer Dataset

Additional experiments and model validation were performed using private customer CT data.

Due to confidentiality restrictions, these data are not included in this repository.

The evaluation protocol used 4-fold cross-validation.

## Results

The model was evaluated on private customer data using **4-fold cross-validation**.

| Fold     | Test Mean Dice |
| -------- | -------------: |
| Fold 1   |     **0.9643** |
| Fold 2   |         0.9073 |
| Fold 3   |         0.9598 |
| Fold 4   |         0.9347 |
| **Mean** |     **0.9415** |

**Best fold Dice:** `0.9643`  
**Mean Dice across 4 folds:** `0.9415`
> The customer dataset is private and is not included in this repository.

### Detailed Evaluation

Additional metrics for the best-performing fold:

| Case     |       Dice |        IoU |  Precision |     Recall |       HD95 |       ASSD |
| -------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Case 1   |     0.9588 |     0.9209 |     0.9731 |     0.9449 |     7.2801 |     1.2810 |
| Case 2   |     0.9699 |     0.9415 |     0.9536 |     0.9867 |     1.4142 |     0.5030 |
| **Mean** | **0.9643** | **0.9312** | **0.9634** | **0.9658** | **4.3472** | **0.8920** |

### Metrics

The primary metric was the **Dice Similarity Coefficient**:

$$
\mathrm{Dice}(P, G) =
\frac{2 \cdot |P \cap G|}
{|P| + |G| + \epsilon}
$$

Additional metrics:

- **IoU**
- **Precision**
- **Recall**
- **HD95**
- **ASSD**

Higher values are better for Dice, IoU, Precision, and Recall. Lower values are better for HD95 and ASSD.

## Pipeline

```text
CT Scan
   ↓
Preprocessing
   ↓
Model Training (MSD)
   ↓
Fine-Tuning / Evaluation
   ↓
4-Fold Cross-Validation
   ↓
Inference
   ↓
FastAPI Service
```

Experiments and configurations are managed using:

- MLflow - experiment tracking and metric logging
- Hydra - configuration management

## Inference Service

The trained model is wrapped in a lightweight **FastAPI** service.

| **Input** | **Output** |
|--------|-------------|
|  A CT volume in NIfTI format: **scan.nii** or **scan.nii.gz** | The service performs segmentation and returns the shape of the generated mask. |


## Disclaimer
This project is intended for research and development purposes and is not a certified medical device.