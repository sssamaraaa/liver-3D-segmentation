import nibabel as nib
import matplotlib.pyplot as plt
import random


ct_path = "datasets\MSDTASK3\Task03_Liver\imagesTr\liver_0.nii.gz"
mask_path = "datasets\MSDTASK3\Task03_Liver\labelsTr\liver_0.nii.gz"
mask_file = "ml\\test\\test1.nii"

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def visualize_msd_sample(ct_file, mask_file, num_slices=5, grid_cols=3):
    ct_data = load_nifti(ct_file)
    mask_data = load_nifti(mask_file)

    total_slices = ct_data.shape[2]
    slice_indices = sorted(random.sample(range(total_slices), num_slices))

    grid_rows = (num_slices + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols*2, figsize=(grid_cols*6, grid_rows*4))

    axes = axes.flatten()

    for i, idx in enumerate(slice_indices):
        # ct without mask
        axes[2*i].imshow(ct_data[:, :, idx], cmap="gray")
        axes[2*i].set_title(f"CT Slice {idx}")
        axes[2*i].axis("off")

        # ct with mask
        axes[2*i+1].imshow(ct_data[:, :, idx], cmap="gray")
        axes[2*i+1].imshow(mask_data[:, :, idx], alpha=0.4, cmap="Reds")
        axes[2*i+1].set_title(f"CT + Mask Slice {idx}")
        axes[2*i+1].axis("off")

    for j in range(2*num_slices, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_mask(mask_file, num_slices=6, grid_cols=3, cmap="gray"):
    mask_data = load_nifti(mask_file)

    total_slices = mask_data.shape[2]
    slice_indices = sorted(random.sample(range(total_slices), num_slices))

    grid_rows = (num_slices + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*4, grid_rows*4))

    axes = axes.flatten()

    for i, idx in enumerate(slice_indices):
        axes[i].imshow(mask_data[:, :, idx], cmap=cmap)
        axes[i].set_title(f"Mask Slice {idx}")
        axes[i].axis("off")

    for j in range(num_slices, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

visualize_mask(mask_file)