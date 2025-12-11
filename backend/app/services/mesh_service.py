import os
import json
import datetime
import numpy as np
import nibabel as nib
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage


def load_mask(mask_path):
    img = nib.load(mask_path)
    mask = img.get_fdata().astype(np.uint8)
    spacing = img.header.get_zooms()  # (z, y, x)
    affine = img.affine
    return mask, spacing, affine

def remove_small_components(mask, min_size=5000, fill_holes=True):
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))

    cleaned = np.zeros_like(mask)
    for i, s in enumerate(sizes, 1):
        if s >= min_size:
            cleaned[labeled == i] = 1

    if fill_holes:
        struct = ndimage.generate_binary_structure(3, 1)  
        cleaned = ndimage.binary_closing(cleaned, structure=struct, iterations=1).astype(np.uint8)

    return cleaned

def mask_to_mesh(mask, spacing, level=0.5, step_size=1):
    if mask.sum() == 0:
        raise ValueError("Mask is empty!")
    
    verts, faces, _, _ = measure.marching_cubes(
        volume=mask,
        level=level,
        spacing=spacing,
        step_size=step_size,  
        allow_degenerate=False
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
    faces_flat = pv_mesh.faces.reshape((-1, 4))
    faces = faces_flat[:, 1:4].copy()
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return tm

def compute_metrics(mask, spacing, pv_mesh):
    tm = pyvista_to_trimesh(pv_mesh)
    
    voxel_vol = spacing[0]*spacing[1]*spacing[2]
    volume_voxels = float(np.sum(mask > 0) * voxel_vol)
    volume_mesh = float(tm.volume) if tm.is_watertight else volume_voxels
    
    surface_area = float(tm.area)
    
    com = tm.center_mass.tolist() if hasattr(tm, 'center_mass') else pv_mesh.center_of_mass().tolist()
    
    bbox = tm.bounds  # (min_x, max_x, min_y, max_y, min_z, max_z)
    dimensions = [
        float(bbox[1] - bbox[0]),  # width (x)
        float(bbox[3] - bbox[2]),  # height (y)
        float(bbox[5] - bbox[4])   # depth (z)
    ]
    
    mesh_quality = {
        "is_watertight": bool(tm.is_watertight),
        "euler_characteristic": int(tm.euler_number),
        "number_of_triangles": int(len(tm.faces)),
        "number_of_vertices": int(len(tm.vertices))
    }
    
    return {
        "volume_ml": round(volume_mesh / 1000.0, 2),
        "volume_mm3": round(volume_mesh, 2),
        "surface_mm2": round(surface_area, 2),
        "center_of_mass": [round(c, 2) for c in com],
        
        "dimensions_mm": [round(d, 2) for d in dimensions],
        "voxel_based_volume_ml": round(volume_voxels / 1000.0, 2),
        "volume_discrepancy_percent": round(
            abs(volume_mesh - volume_voxels) / volume_voxels * 100, 2
        ) if volume_voxels > 0 else 0.0,
        
        "mesh_quality": mesh_quality,
        
        "spacing_mm": [float(s) for s in spacing],
        "mask_shape": list(mask.shape)
    }

def export_mesh(mesh, mask, affine, output_dir, name="liver", save_mask=True, generate_previews=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # mesh (STL, PLY)
    tm = pyvista_to_trimesh(mesh)
    
    stl_path = os.path.join(output_dir, f"{name}.stl")
    ply_path = os.path.join(output_dir, f"{name}.ply")
    
    tm.export(stl_path, file_type='stl_binary')
    tm.export(ply_path, file_type='ply')
    
    # mask (NIfTI and PNG)
    mask_paths = {}
    if save_mask and mask is not None and affine is not None:
        # NIfTI
        mask_nii_path = os.path.join(output_dir, f"{name}_mask.nii.gz")
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        nib.save(mask_img, mask_nii_path)
        mask_paths["nifti"] = mask_nii_path
        
        # PNG (axial, coronal, saggital)
        if generate_previews:
            preview_dir = os.path.join(output_dir, "previews")
            os.makedirs(preview_dir, exist_ok=True)
            
            z_center = mask.shape[0] // 2
            y_center = mask.shape[1] // 2
            x_center = mask.shape[2] // 2
            
            # (axial) - XY plane
            axial_path = os.path.join(preview_dir, f"{name}_axial_z{z_center}.png")
            plt.imsave(axial_path, mask[z_center, :, :], cmap='gray', vmin=0, vmax=1)
            
            # (coronal) - XZ plane
            coronal_path = os.path.join(preview_dir, f"{name}_coronal_y{y_center}.png")
            plt.imsave(coronal_path, mask[:, y_center, :], cmap='gray', vmin=0, vmax=1)
            
            # (sagittal) - YZ plane
            sagittal_path = os.path.join(preview_dir, f"{name}_sagittal_x{x_center}.png")
            plt.imsave(sagittal_path, mask[:, :, x_center], cmap='gray', vmin=0, vmax=1)
            
            mask_paths["previews"] = {
                "axial": axial_path,
                "coronal": coronal_path,
                "sagittal": sagittal_path
            }
    
    pv_preview = os.path.join(output_dir, f"{name}_preview.vtk")
    try:
        mesh.save(pv_preview)
    except Exception:
        pv_preview = None
    
    return {
        "mesh_stl": stl_path,
        "mesh_ply": ply_path,
        "mesh_preview": pv_preview,
        "mask": mask_paths
    }

def build_liver_mesh(mask_path,  output_dir, min_component_size=5000, smooth_iter=30,  decimate_ratio=0.5, mesh_name="liver", fill_holes=True, level=0.5, generate_report=True):
    mask, spacing, affine = load_mask(mask_path)
    cleaned_mask = remove_small_components(mask, min_size=min_component_size, fill_holes=fill_holes)
    pv_mesh = mask_to_mesh(cleaned_mask, spacing, level=level)
    pv_mesh = postprocess_mesh(pv_mesh, smooth_iter=smooth_iter, decimate_ratio=decimate_ratio)
    metrics = compute_metrics(cleaned_mask, spacing, pv_mesh)
    outputs = export_mesh(pv_mesh, cleaned_mask, affine, output_dir, name=mesh_name, save_mask=True, generate_previews=True)
    
    if generate_report:
        report = {
            "metadata": {
                "input_mask": mask_path,
                "output_directory": output_dir,
                "processing_date": datetime.datetime.now().isoformat(),
                "mesh_name": mesh_name
            },
            "parameters": {
                "min_component_size": min_component_size,
                "smooth_iterations": smooth_iter,
                "decimate_ratio": decimate_ratio,
                "fill_holes": fill_holes,
                "marching_cubes_level": level
            },
            "metrics": metrics,
            "output_files": outputs
        }
        
        report_path = os.path.join(output_dir, f"{mesh_name}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        outputs["report_json"] = report_path
    
    return {
        "mesh": pv_mesh,  # object
        "cleaned_mask": cleaned_mask,  # mask
        "outputs": outputs,  # paths to files
        "metrics": metrics  # all metrics
    }