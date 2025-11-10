import nibabel as nib
import numpy as np
import random
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate, gaussian_filter, map_coordinates


def augment_ct3d(img, mask):
    # spatial flips
    if random.random() < 0.5:
        img = np.flip(img, axis=0).copy(); mask = np.flip(mask, axis=0).copy()
    if random.random() < 0.5:
        img = np.flip(img, axis=1).copy(); mask = np.flip(mask, axis=1).copy()
    if random.random() < 0.5:
        img = np.flip(img, axis=2).copy(); mask = np.flip(mask, axis=2).copy()

    # random rotation (<=10°) 
    if random.random() < 0.3:
        axes = random.choice([(0, 1), (1, 2), (0, 2)])
        angle = random.uniform(-10, 10)
        img = rotate(img, angle, axes=axes, reshape=False, order=1, mode='nearest')
        mask = rotate(mask, angle, axes=axes, reshape=False, order=0, mode='nearest')

    # random intensity shift/scale 
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        shift = random.uniform(-0.1, 0.1)
        img = img * scale + shift

    # gamma correction (contrast)
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.5)
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)
        img = img ** gamma
        img = img * (img_max - img_min) + img_min

    # Gaussian noise 
        noise = np.random.normal(0, 0.02, size=img.shape)
        img = img + noise

    # Elastic deformation (small)
    if random.random() < 0.15:
        alpha = random.uniform(20, 40)
        sigma = random.uniform(3, 6)
        shape = img.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = (z + dz, y + dy, x + dx)
        img = map_coordinates(img, indices, order=1, mode='reflect')
        mask = map_coordinates(mask, indices, order=0, mode='reflect')

    return img.astype(np.float32), mask.astype(np.float32)

def load_nifti(path):
    nii = nib.load(path)
    img = nii.get_fdata().astype(np.float32)
    spacing = nii.header.get_zooms()  
    return img, spacing

def resample_to_spacing(volume, orig_spacing, new_spacing):
    factors = tuple([o / n for o, n in zip(orig_spacing, new_spacing)])
    vol_rs = zoom(volume, factors, order=1)  
    return vol_rs

def intensity_clip_normalize(volume, clip_min=-200, clip_max=250):
    vol = np.clip(volume, clip_min, clip_max)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    return vol.astype(np.float32)

class LiverPatchDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_size=(64,128,128),
                 samples_per_volume=16, transform=None,
                 resample_spacing=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = tuple(patch_size)
        self.samples_per_volume = samples_per_volume
        self.transform = transform
        self.resample_spacing = resample_spacing # expensive operation (not implemented yet)

        assert len(image_paths) == len(mask_paths), "Images and masks mismatch"

    def __len__(self):
        return len(self.image_paths) * self.samples_per_volume

    def __getitem__(self, idx):
        volume_idx = idx // self.samples_per_volume

        img, _ = load_nifti(self.image_paths[volume_idx])
        mask, _ = load_nifti(self.mask_paths[volume_idx])

        img = intensity_clip_normalize(img, clip_min=-200, clip_max=250)

        dz, dy, dx = self.patch_size
        zmax, ymax, xmax = img.shape

        # add padding if the size is smaller than the patch
        pad_z = max(0, dz - zmax)
        pad_y = max(0, dy - ymax)
        pad_x = max(0, dx - xmax)

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            img = np.pad(img, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            zmax, ymax, xmax = img.shape

        # patch selection
        has_liver = (mask > 0)
        if random.random() < self.pos_ratio and has_liver.any():
            # select a random voxel with a liver
            z_coords, y_coords, x_coords = np.where(has_liver)
            idx_center = random.randint(0, len(z_coords) - 1)
            cz, cy, cx = z_coords[idx_center], y_coords[idx_center], x_coords[idx_center]

            # center the patch around this voxel
            z1 = max(0, min(cz - dz // 2, zmax - dz))
            y1 = max(0, min(cy - dy // 2, ymax - dy))
            x1 = max(0, min(cx - dx // 2, xmax - dx))
        else:
            # random patch
            z1 = 0 if zmax == dz else np.random.randint(0, zmax - dz)
            y1 = 0 if ymax == dy else np.random.randint(0, ymax - dy)
            x1 = 0 if xmax == dx else np.random.randint(0, xmax - dx)

        z2, y2, x2 = z1 + dz, y1 + dy, x1 + dx

        patch_img = img[z1:z2, y1:y2, x1:x2]
        patch_mask = mask[z1:z2, y1:y2, x1:x2]

        # augmentations
        if self.transform:
            patch_img, patch_mask = self.transform(patch_img, patch_mask)

        # add chanel (1,D,H,W)
        patch_img = np.expand_dims(patch_img, 0).astype(np.float32)
        patch_mask = np.expand_dims((patch_mask > 0).astype(np.float32), 0)

        return patch_img, patch_mask