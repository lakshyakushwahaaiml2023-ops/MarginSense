import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

def compute_tumor_volume(label, spacing):
    """
    Computes tumor volume in cm^3.
    Volume = voxel count * voxel volume (mm^3) / 1000.0
    """
    voxel_count = np.sum(label > 0)
    voxel_volume_mm3 = np.prod(spacing)
    volume_cm3 = (voxel_count * voxel_volume_mm3) / 1000.0
    return float(volume_cm3)

def compute_hemisphere(label, brain_mask=None):
    """
    Determines tumor hemisphere (Left, Right, or Bilateral) based on the centroid x-coordinate.
    x is the last axis (axis 2) in standard (z, y, x) indexing.
    """
    tumor_indices = np.argwhere(label > 0)
    if len(tumor_indices) == 0:
        return "Bilateral"
    
    tumor_centroid_x = np.mean(tumor_indices[:, 2])
    
    if brain_mask is not None and np.any(brain_mask):
        brain_indices = np.argwhere(brain_mask)
        midline = np.mean(brain_indices[:, 2])
        brain_width = np.max(brain_indices[:, 2]) - np.min(brain_indices[:, 2])
    else:
        midline = 64.0
        brain_width = 128.0
        
    # If centroid is within 5% of brain width around midline, call it Bilateral
    threshold = 0.05 * brain_width
    if abs(tumor_centroid_x - midline) <= threshold:
        return "Bilateral"
    elif tumor_centroid_x < midline:
        return "Left"
    else:
        return "Right"

def compute_tumor_location(label, brain_mask=None):
    """
    Maps the tumor centroid to an approximate lobar region (Frontal, Parietal, Temporal, Occipital, Insular)
    using a simple coordinate-based heuristic relative to the brain bounding box.
    NOTE: This is a clearly-labeled geometric approximation, not a proper anatomical atlas registration.
    """
    tumor_indices = np.argwhere(label > 0)
    if len(tumor_indices) == 0:
        return "Insular"
        
    cz, cy, cx = np.mean(tumor_indices, axis=0)
    
    # Get brain bounding box
    if brain_mask is not None and np.any(brain_mask):
        brain_indices = np.argwhere(brain_mask)
        z_min, y_min, x_min = np.min(brain_indices, axis=0)
        z_max, y_max, x_max = np.max(brain_indices, axis=0)
    else:
        z_min, y_min, x_min = 0, 0, 0
        z_max, y_max, x_max = 128, 128, 128
        
    # Compute normalized coordinates within brain bounding box [0, 1]
    nz = (cz - z_min) / max((z_max - z_min), 1.0)
    ny = (cy - y_min) / max((y_max - y_min), 1.0)
    nx = (cx - x_min) / max((x_max - x_min), 1.0)
    
    # Heuristic mapping based on normalized coordinates:
    # Anterior-posterior (y): low values are anterior, high values are posterior
    # Superior-inferior (z): low values are inferior, high values are superior
    # Insular: deep central region
    
    # Deep central region (Insular)
    if 0.35 < nx < 0.65 and 0.4 < ny < 0.6 and 0.3 < nz < 0.7:
        return "Insular"
    # Anterior portion (Frontal)
    elif ny < 0.4:
        return "Frontal"
    # Posterior portion (Occipital)
    elif ny > 0.75:
        return "Occipital"
    # Superior middle (Parietal)
    elif nz > 0.55:
        return "Parietal"
    # Inferior middle (Temporal)
    else:
        return "Temporal"

def compute_ventricle_distance(label, tissue_map, spacing):
    """
    Computes the minimum distance from the tumor mask boundary to CSF/ventricle voxels (tissue_map == 3) in mm.
    """
    csf_indices = np.argwhere(tissue_map == 3)
    tumor_indices = np.argwhere(label > 0)
    
    if len(csf_indices) == 0 or len(tumor_indices) == 0:
        return 999.0  # Fallback large value if no ventricles/tumor found
        
    # Scale coordinates by voxel spacing for physical distance (mm)
    csf_physical = csf_indices * spacing
    tumor_physical = tumor_indices * spacing
    
    # Use KD-Tree for fast minimum distance lookup
    tree = cKDTree(csf_physical)
    distances, _ = tree.query(tumor_physical, k=1)
    
    return float(np.min(distances))

def compute_sphericity(label):
    """
    Computes the sphericity of the tumor mask:
    sphericity = (pi^(1/3) * (6 * V)^(2/3)) / SurfaceArea
    V = volume (in voxel count)
    SurfaceArea = boundary voxel count (estimated via binary erosion subtraction)
    """
    tumor_mask = label > 0
    V = np.sum(tumor_mask)
    if V == 0:
        return 0.0
        
    eroded = binary_erosion(tumor_mask)
    SA = np.sum(tumor_mask ^ eroded)
    
    if SA == 0:
        return 1.0
        
    sphericity = (np.pi**(1/3) * (6.0 * V)**(2/3)) / SA
    # Clamp to [0, 1] range to account for digital discretization artifacts
    return float(np.clip(sphericity, 0.0, 1.0))

def compute_all_imaging_features(label, spacing, tissue_map, brain_mask=None):
    """
    Computes all 5 derived features and returns a dictionary.
    """
    if brain_mask is None:
        # Infer brain mask from tissue map (non-zero regions)
        brain_mask = tissue_map > 0
        
    volume = compute_tumor_volume(label, spacing)
    hemisphere = compute_hemisphere(label, brain_mask)
    location = compute_tumor_location(label, brain_mask)
    ventricle_dist = compute_ventricle_distance(label, tissue_map, spacing)
    sphericity = compute_sphericity(label)
    
    return {
        "tumor_volume_cm3": volume,
        "hemisphere": hemisphere,
        "tumor_location": location,
        "ventricle_dist_mm": ventricle_dist,
        "sphericity": sphericity
    }
