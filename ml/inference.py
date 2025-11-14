import torch
import numpy as np
from torch.amp import autocast


def get_start_points(max_size, patch_size, stride):
    """
    It generates starting indexes so that:
    - a non-empty list is always returned;
    - the last window is guaranteed to cover the right/bottom/back edge (if necessary).
    """
    # if the size is less than or equal to the patch, start from 0
    if max_size <= patch_size:
        return [0]

    starts = []
    i = 0
    # moving forward until the window fits completely
    while i + patch_size <= max_size:
        starts.append(i)
        i += stride

    # if the last window does not capture the right/bottom edge, add precise alignment
    if starts:
        if starts[-1] + patch_size < max_size:
            starts.append(max_size - patch_size)
    else:
        # in case of any strange input data, we return an aligned start
        starts = [max_size - patch_size]

    # remove duplicates (in case the start coincides with an already added one)
    # and return them in sorted order
    starts = sorted(set(starts))
    return starts


def sliding_window_inference(volume, model, device, patch_size, stride_factor=0.5, batch_size=1):
    model.eval()
    with torch.no_grad():
        z_max, y_max, x_max = volume.shape
        dz, dy, dx = patch_size
        # voxel step (int), min 1
        sz, sy, sx = [max(1, int(d * stride_factor)) for d in patch_size]

        prob_map = np.zeros_like(volume, dtype=np.float32)
        count_map = np.zeros_like(volume, dtype=np.float32)

        z_starts = get_start_points(z_max, dz, sz)
        y_starts = get_start_points(y_max, dy, sy)
        x_starts = get_start_points(x_max, dx, sx)

        patches = []
        coords = []

        def process_batch(patches, coords):
            # patches: numpy list [1,1,D,H,W] of the same D,H,W
            batch_np = np.concatenate(patches, axis=0).astype(np.float32)  # shape [B,1,D,H,W]
            patches_t = torch.from_numpy(batch_np).to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(patches_t)  # [B, 1, D, H, W]
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # [B, D, H, W]

            for p_idx, (z0, y0, x0) in enumerate(coords):
                p = probs[p_idx]
                # we take into account that the edge may have exactly the same dimensions, but just in case, we will truncate
                z1 = min(z0 + dz, z_max)
                y1 = min(y0 + dy, y_max)
                x1 = min(x0 + dx, x_max)
                p = p[: (z1 - z0), : (y1 - y0), : (x1 - x0)]
                prob_map[z0:z1, y0:y1, x0:x1] += p
                count_map[z0:z1, y0:y1, x0:x1] += 1

        # we form patches strictly according to patch_size — get_start_points guarantees this
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    # extracting a patch of the required size
                    z1 = z + dz
                    y1 = y + dy
                    x1 = x + dx
                    patch = volume[z:z1, y:y1, x:x1]

                    # just in case, if the patch is smaller for some reason, pad it to the desired size
                    # (this protects against unexpected problems with get_start_points)
                    if patch.shape != (dz, dy, dx):
                        pad_z = dz - patch.shape[0]
                        pad_y = dy - patch.shape[1]
                        pad_x = dx - patch.shape[2]
                        patch = np.pad(patch,
                                       ((0, pad_z), (0, pad_y), (0, pad_x)),
                                       mode='constant', constant_values=0)

                    patches.append(patch[None, None, ...])  # [1,1,D,H,W]
                    coords.append((z, y, x))

                    if len(patches) == batch_size:
                        process_batch(patches, coords)
                        patches, coords = [], []

        if len(patches) > 0:
            process_batch(patches, coords)

        # avoiding division by zero
        prob_map = prob_map / np.maximum(count_map, 1e-8)
        return prob_map
