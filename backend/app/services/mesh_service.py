import os
import numpy as np
import nibabel as nib
from skimage import measure
from scipy import ndimage
import pyvista as pv
import trimesh


def load_mask(mask_path):
    img = nib.load(mask_path)
    mask = img.get_fdata().astype(np.uint8)
    spacing = img.header.get_zooms()  # (z, y, x)
    return mask, spacing

def remove_small_components(mask, min_size=5000):
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))

    cleaned = np.zeros_like(mask)
    for i, s in enumerate(sizes, 1):
        if s >= min_size:
            cleaned[labeled == i] = 1
    return cleaned

def mask_to_mesh(mask, spacing):
    verts, faces, _, _ = measure.marching_cubes(
        volume=mask,
        level=0.5,
        spacing=spacing
    )
    # prepare faces for pyvista: each face prefixed by 3
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv)
    return mesh

def postprocess_mesh(mesh, smooth_iter=30, decimate_ratio=0.5):
    if smooth_iter and smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.01)
    if decimate_ratio and 0 < decimate_ratio < 1:
        # PyVista.decimate expects target_reduction between 0..1
        mesh = mesh.decimate(target_reduction=decimate_ratio)
    return mesh

def pyvista_to_trimesh(pv_mesh):
    verts = pv_mesh.points.copy()
    # pv_mesh.faces — flat array like [n, v0, v1, v2, n, v0, v1, v2, ...]
    faces_flat = pv_mesh.faces.reshape((-1, 4))
    faces = faces_flat[:, 1:4].copy()
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return tm

def compute_metrics(mask, spacing, pv_mesh):
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])  # mm^3
    volume_mm3 = float(np.sum(mask > 0) * voxel_volume)
    volume_ml = volume_mm3 / 1000.0

    tm = pyvista_to_trimesh(pv_mesh)

    surface_mm2 = float(tm.area)
    # center_of_mass returns (x,y,z)
    try:
        com = tm.center_mass.tolist()
    except Exception:
        com = pv_mesh.center_of_mass().tolist()

    return {
        "spacing": spacing,
        "volume_ml": volume_ml,
        "volume_mm3": volume_mm3,
        "surface_mm2": surface_mm2,
        "center_of_mass": [float(c) for c in com]
    }

def export_mesh(mesh, output_dir, name="liver"):
    os.makedirs(output_dir, exist_ok=True)
    tm = pyvista_to_trimesh(mesh)

    stl_path = os.path.join(output_dir, f"{name}.stl")
    ply_path = os.path.join(output_dir, f"{name}.ply")

    tm.export(stl_path)
    tm.export(ply_path)

    pv_preview = os.path.join(output_dir, f"{name}_pv.vtk")
    try:
        mesh.save(pv_preview)
    except Exception:
        pv_preview = None

    return {
        "stl": stl_path,
        "ply": ply_path,
        "pv_preview": pv_preview
    }

def build_liver_mesh( mask_path, output_dir, min_component_size=5000, smooth_iter=30, decimate_ratio=0.5, mesh_name="liver"):
    mask, spacing = load_mask(mask_path)
    cleaned_mask = remove_small_components(mask, min_size=min_component_size)
    pv_mesh = mask_to_mesh(cleaned_mask, spacing)
    pv_mesh = postprocess_mesh(pv_mesh, smooth_iter=smooth_iter, decimate_ratio=decimate_ratio)
    metrics = compute_metrics(cleaned_mask, spacing, pv_mesh)
    outputs = export_mesh(pv_mesh, output_dir, name=mesh_name)

    return {
        "mesh_paths": outputs,
        "metrics": metrics
    }
