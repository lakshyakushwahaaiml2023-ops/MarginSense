import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def find_file_by_suffix(directory, suffixes):
    """Finds a file in the directory that ends with one of the suffixes."""
    for f in os.listdir(directory):
        for suffix in suffixes:
            if f.lower().endswith(suffix.lower()):
                return os.path.join(directory, f)
    return None

def normalize_volume(volume):
    """Applies z-score normalization to non-zero (brain) voxels only."""
    mask = volume > 0
    if not np.any(mask):
        return volume.astype(np.float32)
    mean = volume[mask].mean()
    std = volume[mask].std()
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - mean) / (std + 1e-8)
    return normalized

def resample_volume(volume, target_shape=(128, 128, 128), order=1):
    """Resamples volume to target shape using scipy zoom."""
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom_factors, order=order)

def estimate_tissue_types(image, label):
    """
    Estimates healthy brain tissue types (White Matter, Gray Matter, CSF) from T1 and T2 channels.
    0 = Background/Tumor, 1 = White Matter (WM), 2 = Gray Matter (GM), 3 = CSF
    """
    # image shape (4, 128, 128, 128)
    t1 = image[0]
    t2 = image[2]
    flair = image[3]
    
    # Brain mask: non-zero in FLAIR/T1/T2, excluding any tumor labels (label > 0)
    brain_mask = (flair > 0) & (label == 0)
    
    tissue_map = np.zeros_like(label, dtype=np.int8)
    if not np.any(brain_mask):
        return tissue_map
        
    normal_t1 = t1[brain_mask]
    if len(normal_t1) < 100:
        return tissue_map
        
    # Standard T1 intensity relationships: WM > GM > CSF
    p33, p66 = np.percentile(normal_t1, [33, 66])
    
    tissue_map[brain_mask & (t1 > p66)] = 1  # White Matter (WM)
    tissue_map[brain_mask & (t1 >= p33) & (t1 <= p66)] = 2  # Gray Matter (GM)
    tissue_map[brain_mask & (t1 < p33)] = 3  # CSF
    
    return tissue_map

def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Each subfolder in data/raw is a patient
    patient_dirs = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) 
                    if os.path.isdir(os.path.join(raw_dir, d))]
    
    if not patient_dirs:
        print("[-] No raw patient folders found in data/raw.")
        print("    Please place BraTS dataset subfolders (containing t1, t1ce, t2, flair, and seg NIfTI files) into data/raw.")
        return

    print(f"[*] Found {len(patient_dirs)} patient folders in {raw_dir}.")
    
    modality_suffixes = {
        't1': ["_t1.nii", "_t1.nii.gz"],
        't1ce': ["_t1ce.nii", "_t1ce.nii.gz"],
        't2': ["_t2.nii", "_t2.nii.gz"],
        'flair': ["_flair.nii", "_flair.nii.gz"],
        'seg': ["_seg.nii", "_seg.nii.gz"]
    }
    
    recurrence_suffixes = ["_recurrence.nii", "_recurrence.nii.gz", "_rec.nii", "_rec.nii.gz"]
    
    loaded_count = 0
    
    for p_dir in patient_dirs:
        patient_id = os.path.basename(p_dir)
        print(f"\n--- Processing Patient: {patient_id} ---")
        
        # Locate files for each modality and the segmentation mask
        paths = {}
        missing = False
        for mod, suffixes in modality_suffixes.items():
            path = find_file_by_suffix(p_dir, suffixes)
            if not path:
                print(f"  [Error] Missing NIfTI file for modality: '{mod}' in {p_dir}")
                missing = True
                break
            paths[mod] = path
            
        if missing:
            print(f"  [Warning] Skipping patient {patient_id} due to missing modalities.")
            continue
            
        try:
            # Load volumes and calculate spacing
            volumes = {}
            shapes = {}
            orig_zooms = None
            for mod, path in paths.items():
                img = nib.load(path)
                data = img.get_fdata()
                volumes[mod] = data
                shapes[mod] = data.shape
                if mod == 't1':
                    orig_zooms = img.header.get_zooms()[:3]
                print(f"  Loaded {mod}: shape {data.shape}, dtype {data.dtype}")
                
            # We assume all modal volumes have same dimensions for a single patient
            first_shape = list(shapes.values())[0]
            for mod, shape in shapes.items():
                if shape != first_shape:
                    print(f"  [Warning] Modality {mod} shape {shape} mismatch with standard shape {first_shape}.")
            
            # Calculate resampled voxel spacing (for target shape 128x128x128)
            if orig_zooms is None or len(orig_zooms) < 3:
                orig_zooms = (1.0, 1.0, 1.0)
            resampled_spacing = np.array([
                orig_zooms[0] * first_shape[0] / 128.0,
                orig_zooms[1] * first_shape[1] / 128.0,
                orig_zooms[2] * first_shape[2] / 128.0
            ], dtype=np.float32)
            
            # Check for optional recurrence mask
            rec_path = find_file_by_suffix(p_dir, recurrence_suffixes)
            if rec_path:
                print(f"  Found post-treatment recurrence mask: {os.path.basename(rec_path)}")
                rec_img = nib.load(rec_path)
                rec_data = rec_img.get_fdata()
                # Resample recurrence mask (order=0)
                recurrence_resampled = resample_volume(rec_data, target_shape=(128, 128, 128), order=0)
                recurrence_resampled = np.round(recurrence_resampled).astype(np.int8)
            else:
                print("  No recurrence mask found. Defaulting recurrence target to active tumor (seg > 0).")
                recurrence_resampled = None
            
            # Resample and Normalize
            resampled_modalities = []
            for mod in ['t1', 't1ce', 't2', 'flair']:
                # Resample with order=1 (bilinear)
                resampled = resample_volume(volumes[mod], target_shape=(128, 128, 128), order=1)
                # Normalize intensity values within brain region
                normalized = normalize_volume(resampled)
                resampled_modalities.append(normalized)
                
            # Resample label segmentation mask with order=0 (nearest neighbor) to preserve integer labels
            seg_resampled = resample_volume(volumes['seg'], target_shape=(128, 128, 128), order=0)
            seg_resampled = np.round(seg_resampled).astype(np.int8)  # Ensure integer values
            
            # If recurrence mask was not loaded, default to the resampled tumor mask (seg > 0)
            if recurrence_resampled is None:
                recurrence_resampled = (seg_resampled > 0).astype(np.int8)
            
            # Stack modalities into (4, 128, 128, 128)
            stacked_image = np.stack(resampled_modalities, axis=0)
            
            # Estimate tissue types from resampled image and label
            tissue_map = estimate_tissue_types(stacked_image, seg_resampled)
            
            # Save compressed
            out_file = os.path.join(processed_dir, f"{patient_id}.npz")
            np.savez_compressed(
                out_file, 
                image=stacked_image, 
                label=seg_resampled, 
                spacing=resampled_spacing,
                recurrence=recurrence_resampled,
                tissue_map=tissue_map
            )
            
            print(f"  [Success] Saved preprocessed data to {out_file}")
            print(f"            Stacked image shape: {stacked_image.shape}")
            print(f"            Label mask shape:    {seg_resampled.shape}")
            print(f"            Recurrence shape:    {recurrence_resampled.shape}")
            print(f"            Tissue map shape:    {tissue_map.shape} (WM: {np.sum(tissue_map==1)}, GM: {np.sum(tissue_map==2)}, CSF: {np.sum(tissue_map==3)})")
            print(f"            Resampled spacing:   {resampled_spacing}")
            
            loaded_count += 1
            
        except Exception as e:
            print(f"  [Error] Failed to process patient {patient_id}: {str(e)}")
            
    print(f"\n[+] Preprocessing Complete. Successfully loaded and cached {loaded_count}/{len(patient_dirs)} patients.")

if __name__ == "__main__":
    main()
