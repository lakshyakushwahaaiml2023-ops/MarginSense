"""
export_meshes.py — Preprocess a patient into:
  - Tumor core mesh (.glb, aligned to template brain)
  - Standard margin mesh (.glb, aligned to template brain)
  - Predicted margin mesh (.glb, aligned to template brain)
  - 64³ density and uncertainty float32 arrays (serialized to data.js & JSON cache)
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.ndimage import zoom, binary_fill_holes
from skimage.measure import marching_cubes

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[Warning] trimesh not found — mesh will not be simplified.")


# ── helpers ──────────────────────────────────────────────────────────────────

def create_synthetic_patient(patient_id):
    """Re-generates anisotropic synthetic patient data."""
    os.makedirs("data/processed", exist_ok=True)
    out_file = f"data/processed/{patient_id}.npz"
    if os.path.exists(out_file):
        return
    shape = (128, 128, 128)
    image   = np.zeros((4, *shape), dtype=np.float32)
    label   = np.zeros(shape, dtype=np.int8)
    recurrence  = np.zeros(shape, dtype=np.int8)
    tissue_map  = np.zeros(shape, dtype=np.int8)
    bz, by, bx  = np.ogrid[:128, :128, :128]

    brain_mask = ((bx-64)/55)**2 + ((by-64)/55)**2 + ((bz-64)/45)**2 <= 1.0
    image[0, brain_mask] = 0.5 + 0.1*np.random.randn(brain_mask.sum())
    image[2, brain_mask] = 0.8 + 0.1*np.random.randn(brain_mask.sum())

    cx, cy, cz = 64, 64, 64
    rx, ry, rz = 12, 8, 8
    dist_tumor = ((bx-cx)/rx)**2 + ((by-cy)/ry)**2 + ((bz-cz)/rz)**2
    label[dist_tumor <= 1.0] = 1
    label[(label==0) & (((bx-cx)/(rx+4))**2 + ((by-cy)/(ry+4))**2 + ((bz-cz)/(rz+4))**2 <= 1.0)] = 2

    rx_rec, ry_rec, rz_rec = rx+12, ry+3, rz+3
    recurrence[((bx-cx)/rx_rec)**2 + ((by-cy)/ry_rec)**2 + ((bz-cz)/rz_rec)**2 <= 1.0] = 1

    wm = (np.abs(by-cy) <= 12) & (np.abs(bz-cz) <= 12)
    tissue_map[brain_mask] = 2
    tissue_map[brain_mask & wm] = 1
    tissue_map[brain_mask & (np.sqrt((bx-64)**2+(by-64)**2+(bz-64)**2) <= 16)] = 3
    tissue_map[label > 0] = 0

    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    np.savez_compressed(out_file, image=image, label=label,
                        spacing=spacing, recurrence=recurrence, tissue_map=tissue_map)


def build_mesh_glb(volume_bool: np.ndarray, spacing: np.ndarray,
                   target_faces: int = 40_000,
                   brain_bool: np.ndarray = None) -> bytes:
    """Run marching cubes on a boolean mask, align to template brain, decimate, return GLB bytes."""
    # Smooth binary fill to remove internal cavities
    vol = binary_fill_holes(volume_bool).astype(np.uint8)
    if vol.sum() == 0:
        return b""

    verts, faces, normals, _ = marching_cubes(vol, level=0.5, spacing=tuple(spacing))

    # ── ALIGNMENT TO STATIC TEMPLATE ─────────────────────────────────────────
    # Marching cubes outputs vertices as [z, y, x]. Map to standard [x, y, z]:
    verts_xyz = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)

    if brain_bool is not None:
        # 1. Compute patient brain bounds in [x, y, z] space
        brain_indices = np.argwhere(brain_bool)
        brain_xyz = np.stack([brain_indices[:, 2], brain_indices[:, 1], brain_indices[:, 0]], axis=1) * spacing
        min_brain = brain_xyz.min(axis=0)
        max_brain = brain_xyz.max(axis=0)
        centroid_patient = (min_brain + max_brain) / 2.0
        size_patient = max_brain - min_brain

        # 2. Bounding Box & Centroid of savir2010/Aurna template brain model:
        # Bounds: [[-0.7558, -0.8362, -1.0018], [0.7545, 0.8357, 0.9985]]
        # Centroid: [0.0055, 0.0689, -0.0351]
        centroid_template = np.array([0.0055, 0.0689, -0.0351])
        size_template = np.array([1.5103, 1.6719, 2.0002])

        # 3. Calculate uniform scaling factor
        scale_factor = np.mean(size_template / size_patient)
        # Shift to center, scale, and shift to template location
        verts_xyz = (verts_xyz - centroid_patient) * scale_factor + centroid_template

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces)
    mesh.fix_normals()

    if not TRIMESH_AVAILABLE:
        return mesh.export(file_type='glb')

    # Decimate if too large
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception as e:
            print(f"   [Warn] Decimation failed ({e}), keeping original mesh.")

    print(f"   Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh.export(file_type='glb')


def downsample_volume(volume: np.ndarray, target: int = 64) -> list:
    """Downsample a 3D float volume to target³ and return as a flat Python list (uint8 0-255)."""
    factor = target / np.array(volume.shape, dtype=float)
    small  = zoom(volume.astype(np.float32), zoom=factor, order=1)

    # Normalize to 0-1
    vmin, vmax = small.min(), small.max()
    if vmax > vmin:
        small = (small - vmin) / (vmax - vmin)

    # Quantize to uint8 for JSON efficiency
    small_u8 = (small * 255).clip(0, 255).astype(np.uint8)
    return small_u8.flatten().tolist()


# ── main ─────────────────────────────────────────────────────────────────────

def export_patient(patient_id: str):
    """Full export pipeline for a patient."""
    print(f"\n[*] Exporting meshes and volume textures for: {patient_id}")
    is_synthetic = patient_id.startswith("synthetic_")

    # Load processed data
    npz_path = f"data/processed/{patient_id}.npz"
    if not os.path.exists(npz_path):
        if is_synthetic:
            create_synthetic_patient(patient_id)
        else:
            print(f"[Error] No processed data found for {patient_id}")
            sys.exit(1)

    data       = np.load(npz_path)
    label      = data['label']          # (128,128,128) int8
    spacing    = data['spacing']        # (3,) mm
    image      = data['image']

    # Load uniform margin result
    uniform_path = f"outputs/{patient_id}_baseline_uniform.npz"
    uniform_margin = None
    if os.path.exists(uniform_path):
        uniform_margin = np.load(uniform_path)['dilated_mask'].astype(bool)

    # Load ensemble prediction + uncertainty
    ensemble_path = f"outputs/{patient_id}_prediction_ensemble.npz"
    mean_density  = None
    std_density   = None
    if os.path.exists(ensemble_path):
        ens = np.load(ensemble_path)
        mean_density = ens['mean_density']
        std_density  = ens['std_density']

    # Output directory
    model_dir = "src/viz/models"
    os.makedirs(model_dir, exist_ok=True)

    # Compute patient brain mask for alignment
    brain_bool = (image[0] > 0.1)  # T1 channel non-zero → in brain
    
    # Compute alignment scale and patient brain mask centroid
    centroid_template = np.array([0.0055, 0.0689, -0.0351])
    size_template = np.array([1.5103, 1.6719, 2.0002])
    brain_indices = np.argwhere(brain_bool)
    if len(brain_indices) > 0:
        brain_xyz = np.stack([brain_indices[:, 2], brain_indices[:, 1], brain_indices[:, 0]], axis=1) * spacing
        min_brain = brain_xyz.min(axis=0)
        max_brain = brain_xyz.max(axis=0)
        centroid_patient = (min_brain + max_brain) / 2.0
        size_patient = max_brain - min_brain
        scale_factor = np.mean(size_template / size_patient)
    else:
        centroid_patient = np.zeros(3)
        scale_factor = 1.0

    # 1. Tumor core surface
    print("[*] Extracting tumor core mesh...")
    tumor_bool = (label == 1)
    if tumor_bool.sum() > 50:
        glb_bytes = build_mesh_glb(tumor_bool, spacing, target_faces=15_000, brain_bool=brain_bool)
        tumor_path = os.path.join(model_dir, f"tumor_{patient_id}.glb")
        with open(tumor_path, 'wb') as f:
            f.write(glb_bytes)
        print(f"   Saved: {tumor_path}  ({len(glb_bytes)//1024} KB)")
    else:
        tumor_path = None
        print("   Tumor mask too small — skipped.")

    # 2. Standard clinical margin surface (Uniform Margin)
    if uniform_margin is not None:
        print("[*] Extracting standard clinical margin mesh...")
        margin_bool = uniform_margin & ~tumor_bool
        if margin_bool.sum() > 50:
            glb_bytes = build_mesh_glb(margin_bool, spacing, target_faces=40_000, brain_bool=brain_bool)
            margin_path = os.path.join(model_dir, f"margin_{patient_id}.glb")
            with open(margin_path, 'wb') as f:
                f.write(glb_bytes)
            print(f"   Saved: {margin_path}  ({len(glb_bytes)//1024} KB)")
        else:
            margin_path = None
    else:
        margin_path = None
        print("   No uniform margin output found — margin mesh skipped.")

    # 3. MarginSense Predicted Infiltration Margin surface (density > 0.35)
    predicted_margin_path = None
    if mean_density is not None:
        print("[*] Extracting MarginSense predicted margin mesh...")
        pred_bool = (mean_density > 0.35) & ~tumor_bool
        if pred_bool.sum() > 50:
            glb_bytes = build_mesh_glb(pred_bool, spacing, target_faces=40_000, brain_bool=brain_bool)
            predicted_margin_path = os.path.join(model_dir, f"predicted_margin_{patient_id}.glb")
            with open(predicted_margin_path, 'wb') as f:
                f.write(glb_bytes)
            print(f"   Saved: {predicted_margin_path}  ({len(glb_bytes)//1024} KB)")

    # 4. Volumetric textures (64³)
    print("[*] Building 64³ density and uncertainty volumes...")
    temporal_slices = []
    if mean_density is not None:
        for t_idx in range(6):
            t = t_idx / 5.0
            vol = mean_density * t + (label==1).astype(np.float32) * (1 - t)
            temporal_slices.append(downsample_volume(vol.clip(0,1), 64))
    else:
        for t_idx in range(6):
            tumor_vol = (label==1).astype(np.float32)
            temporal_slices.append(downsample_volume(tumor_vol, 64))

    density_64  = downsample_volume(mean_density if mean_density is not None else (label==1).astype(np.float32), 64)
    uncertainty_64 = downsample_volume(std_density  if std_density  is not None else np.zeros_like(label, dtype=np.float32), 64)

    # 5. Build data.js and patient cache JSON for the frontend
    print("[*] Serializing volume data to src/viz/data.js...")
    payload = {
        "patient_id":       patient_id,
        "shape":            [64, 64, 64],
        "spacing":          spacing.tolist(),
        "has_tumor":        tumor_path is not None,
        "has_margin":       margin_path is not None,
        "has_predicted_margin": predicted_margin_path is not None,
        "density":          density_64,
        "uncertainty":      uncertainty_64,
        "temporal_densities": temporal_slices,
        "alignment_scale":    float(scale_factor),
        "alignment_centroid": centroid_patient.tolist(),
    }

    data_js_path = "src/viz/data.js"
    with open(data_js_path, "w") as f:
        f.write("window.patientData = ")
        json.dump(payload, f, separators=(',', ':'))
        f.write(";")

    cache_path = os.path.join(model_dir, f"data_{patient_id}.json")
    with open(cache_path, "w") as f:
        json.dump(payload, f, separators=(',', ':'))

    print(f"[+] Done — data.js written ({os.path.getsize(data_js_path)//1024} KB)")
    print(f"[+] Cache written to {cache_path} ({os.path.getsize(cache_path)//1024} KB)")
    print(f"[+] Meshes in {model_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_id", nargs="?", default="synthetic_patient_2")
    args = parser.parse_args()
    export_patient(args.patient_id)
