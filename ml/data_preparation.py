import nibabel as nib
import matplotlib.pyplot as plt


ct_path = "MSD TASK3\Task03_Liver\imagesTr\liver_0.nii.gz"
mask_path = "MSD TASK3\Task03_Liver\labelsTr\liver_0.nii.gz"

def show_image(ct_path, mask_path):
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)

    ct_data = ct_img.get_fdata()
    mask_data = mask_img.get_fdata()

    z = ct_data.shape[2] // 2
    ct_slice = ct_data[:, :, z]
    mask_slice = mask_data[:, :, z]
    ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
    mask_scaled = mask_slice * 255

    plt.imshow(ct_norm, cmap="gray")
    plt.imshow(mask_scaled, cmap="Reds", alpha=0.5) 
    plt.title(f"Slice {z}")
    plt.axis("off")
    plt.show()
