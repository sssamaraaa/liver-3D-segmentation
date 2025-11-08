import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import numpy as np


ct_path = "datasets/130/segmentations/segmentation-50.nii"
mask_path = "datasets/130/volume_pt5/volume-50.nii"

def vizualize_msd_sample(image_path, mask_path):
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

def visualize_lits_sample(volume_path, segmentation_path, slice_idx=None, save_path=None):
    try:
        ct_scan = nib.load(volume_path)
        ct_data = ct_scan.get_fdata()
        
        mask = nib.load(segmentation_path)
        mask_data = mask.get_fdata()
    except Exception as e:
        print(f"Ошибка загрузки файлов: {e}")
        return
    
    if ct_data.shape != mask_data.shape:
        print(f"Предупреждение: размеры не совпадают! CT: {ct_data.shape}, Mask: {mask_data.shape}")
    
    if slice_idx is None:
        slice_idx = ct_data.shape[2] // 2 
    
    ct_slice = ct_data[:, :, slice_idx]
    mask_slice = mask_data[:, :, slice_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title(f'КТ-срез {slice_idx}\n(Печень + Опухоли)')
    axes[0].axis('off')
    
    axes[1].imshow(mask_slice, cmap='jet')
    axes[1].set_title('Маска сегментации\n(0:фон, 1:печень, 2:опухоль)')
    axes[1].axis('off')
    
    axes[2].imshow(ct_slice, cmap='gray')
    colored_mask = np.zeros((*mask_slice.shape, 4))
    colored_mask[mask_slice == 1] = [0, 1, 0, 0.3] 
    colored_mask[mask_slice == 2] = [1, 0, 0, 0.5] 
    axes[2].imshow(colored_mask)
    axes[2].set_title('Наложение маски на КТ\n(зеленый: печень, красный: опухоль)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Изображение сохранено: {save_path}")
    
    plt.show()
    
    print(f"Размерность volume: {ct_data.shape}")
    print(f"Размерность mask: {mask_data.shape}")
    print(f"Уникальные значения в маске: {np.unique(mask_data)}")
    print(f"Диапазон HU в КТ: [{ct_data.min():.1f}, {ct_data.max():.1f}]")


visualize_lits_sample(mask_path, ct_path)