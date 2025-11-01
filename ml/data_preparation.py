import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider


ct_path = "MSD TASK3\Task03_Liver\imagesTr\liver_1.nii.gz"
mask_path = "MSD TASK3\Task03_Liver\labelsTr\liver_0.nii.gz"

def interactive_msd_viewer(image_path, mask_path):
    img = nib.load(image_path)
    mask = nib.load(mask_path)
    
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()
    
    @interact(slice_idx=IntSlider(min=0, max=img_data.shape[2]-1, value=img_data.shape[2]//2))
    def update_slice(slice_idx):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_data[:, :, slice_idx], cmap='gray', vmin=-100, vmax=180)
        axes[0].set_title(f'CT - срез {slice_idx}')
        
        axes[1].imshow(mask_data[:, :, slice_idx], cmap='tab10')
        axes[1].set_title('Маска')
        
        axes[2].imshow(img_data[:, :, slice_idx], cmap='gray', vmin=-100, vmax=180)
        axes[2].imshow(mask_data[:, :, slice_idx], cmap='jet', alpha=0.4)
        axes[2].set_title('Маска на CT')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()
        
        mask_slice = mask_data[:, :, slice_idx]
        print(f"Вокселей печени в срезе: {mask_slice.sum()} из {mask_slice.size}")

interactive_msd_viewer(ct_path, mask_path)
