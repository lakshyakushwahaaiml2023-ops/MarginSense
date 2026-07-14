import os
import sys
import time
import argparse
import numpy as np
from scipy.ndimage import distance_transform_edt

def create_synthetic_data():
    """Generates synthetic 128x128x128 data with a spherical tumor in the center."""
    print("[*] Generating synthetic data for testing...")
    shape = (128, 128, 128)
    label = np.zeros(shape, dtype=np.int8)
    
    # Create a sphere in the center of radius 15 voxels
    cx, cy, cz = 64, 64, 64
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    label[dist_from_center <= 15] = 1 # Tumor core
    
    # Anisotropic spacing
    spacing = np.array([1.0, 1.0, 1.2], dtype=np.float32)
    return label, spacing

def main():
    parser = argparse.ArgumentParser(description="Clinical Standard Uniform 1.5cm Margin Baseline")
    parser.add_argument("patient_id", type=str, nargs="?", default="synthetic_patient",
                        help="Patient ID to process. Ignored if --synthetic is set.")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test data.")
    args = parser.parse_args()
    
    if args.synthetic:
        patient_id = "synthetic_patient"
        label, spacing = create_synthetic_data()
    else:
        patient_id = args.patient_id
        processed_path = f"data/processed/{patient_id}.npz"
        if not os.path.exists(processed_path):
            print(f"[Error] Processed data not found at {processed_path}.")
            print("Please run preprocessing first, or use the --synthetic flag to test.")
            sys.exit(1)
            
        print(f"[*] Loading processed data for patient {patient_id}...")
        data = np.load(processed_path)
        label = data['label']
        if 'spacing' in data:
            spacing = data['spacing']
        else:
            print("[Warning] 'spacing' not found in processed file. Defaulting to isotropic 1.0mm spacing.")
            spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            
    print(f"[*] Input segmentation shape: {label.shape}")
    print(f"[*] Voxel spacing (mm): {spacing}")
    
    # Define tumor mask (any label > 0 is considered tumor)
    tumor_mask = (label > 0).astype(np.int8)
    tumor_volume_voxels = np.sum(tumor_mask)
    print(f"[*] Active tumor volume: {tumor_volume_voxels} voxels")
    
    if tumor_volume_voxels == 0:
        print("[Warning] No tumor cells detected in the mask. Dilation will be empty.")
        dilated_mask = np.zeros_like(tumor_mask)
        elapsed_time = 0.0
    else:
        # Start wall-clock timer
        start_time = time.perf_counter()
        
        # Calculate distances from tumor boundary.
        # EDT expects background (0) to compute distance to foreground (1), 
        # so we compute EDT on the inverted tumor mask.
        distances = distance_transform_edt(1 - tumor_mask, sampling=spacing)
        
        # 1.5 cm = 15.0 mm uniform margin
        margin_mm = 15.0
        dilated_mask = (distances <= margin_mm).astype(np.int8)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
    print(f"[+] Dilation complete.")
    print(f"    Target volume (with 1.5cm margin): {np.sum(dilated_mask)} voxels")
    print(f"    Execution Time: {elapsed_time:.4f} seconds")
    
    # Save output volume
    os.makedirs("outputs", exist_ok=True)
    out_file = f"outputs/{patient_id}_baseline_uniform.npz"
    np.savez_compressed(
        out_file, 
        dilated_mask=dilated_mask, 
        original_mask=tumor_mask, 
        elapsed_time=elapsed_time,
        spacing=spacing
    )
    print(f"[+] Saved baseline uniform margin to {out_file}")

if __name__ == "__main__":
    main()
