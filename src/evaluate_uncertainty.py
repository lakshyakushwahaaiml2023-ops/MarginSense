import os
import sys
import time
import argparse
import numpy as np
import torch

# Import our MarginSense network
from src.models.amortized_pinn import MarginSenseNet, load_covariate_vector

def main():
    parser = argparse.ArgumentParser(description="MarginSense Ensemble Uncertainty Quantification")
    parser.add_argument("patient_id", type=str, nargs="?", default="synthetic_patient_2",
                        help="Patient ID to evaluate. Ignored if --synthetic is set.")
    parser.add_argument("--synthetic", action="store_true", help="Evaluate the synthetic patient 2.")
    args = parser.parse_args()
    
    # 1. Device Setup
    if not torch.cuda.is_available():
        print("[CRITICAL] CUDA is not available! Evaluation must run on local GPU.")
        sys.exit(1)
        
    device = torch.device('cuda')
    print(f"[*] Running on local GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Set Patient ID and Load Data
    patient_id = "synthetic_patient_2" if args.synthetic else args.patient_id
    processed_path = f"data/processed/{patient_id}.npz"
    
    if not os.path.exists(processed_path):
        print(f"[Error] Preprocessed data not found at {processed_path}.")
        print("Please run preprocessing first.")
        sys.exit(1)
        
    print(f"[*] Loading preprocessed data for patient: {patient_id}...")
    data = np.load(processed_path)
    image = data['image']
    label = data['label']
    spacing = data['spacing'] if 'spacing' in data else np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    # Formulate 5-channel encoder input
    volume_in = np.concatenate([image, np.expand_dims(label, axis=0)], axis=0)
    volume_tensor = torch.tensor(volume_in, dtype=torch.float32, device=device).unsqueeze(0)
    
    # 3. Load Ensemble Models
    ensemble_checkpoints = []
    for i in range(5):
        ckpt_path = f"outputs/marginsense_ensemble_{i}.pt"
        if not os.path.exists(ckpt_path):
            print(f"[Error] Ensemble checkpoint not found at {ckpt_path}.")
            print("Please run ensemble training first.")
            sys.exit(1)
        ensemble_checkpoints.append(ckpt_path)
        
    print(f"[*] Loading 5 ensemble models...")
    models = []
    for ckpt_path in ensemble_checkpoints:
        model = MarginSenseNet(embedding_dim=64, hidden_dim=64).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        models.append(model)
        
    print("[*] Successfully loaded all 5 ensemble networks.")
    
    # 4. Perform Ensemble Grid Evaluation
    print("[*] Running ensemble grid evaluation (2.1 million points, t = 1)...")
    start_time = time.perf_counter()
    
    # Generate full coordinate grid
    x = np.linspace(0, 1, 128)
    y = np.linspace(0, 1, 128)
    z = np.linspace(0, 1, 128)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    coords_grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones_like(xx.ravel())], axis=1)
    
    # Arrays to hold final predictions
    mean_density = np.zeros(coords_grid.shape[0], dtype=np.float32)
    std_density = np.zeros(coords_grid.shape[0], dtype=np.float32)
    
    cov_vec = load_covariate_vector(patient_id, npz_data=data)
    cov_tensor = torch.tensor(cov_vec, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Run CNN encoder on the volume for all 5 models to get embeddings
    embeddings = []
    with torch.no_grad():
        for idx, model in enumerate(models):
            z_embed, _, _ = model.forward_encoder(volume_tensor, cov_tensor)
            embeddings.append(z_embed.float())
            if idx == 0:
                from src.models.amortized_pinn import save_patient_latent
                save_patient_latent(patient_id, z_embed)
            
    # Evaluation loop in chunks to prevent CUDA OOM
    chunk_size = 131072
    with torch.no_grad():
        for i in range(0, len(coords_grid), chunk_size):
            chunk = coords_grid[i:i+chunk_size]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)
            
            # Predict density using each of the 5 models
            chunk_preds = []
            for m_idx, model in enumerate(models):
                z_embed = embeddings[m_idx]
                z_expanded = z_embed.expand(chunk_tensor.size(0), -1)
                pred = model.coordinate_mlp(chunk_tensor, z_expanded)
                chunk_preds.append(pred.cpu().numpy().squeeze())
                
            # Stack along model axis: shape (5, chunk_size)
            chunk_preds_stacked = np.stack(chunk_preds, axis=0)
            
            # Compute mean and standard deviation per voxel
            mean_vals = np.mean(chunk_preds_stacked, axis=0)
            std_vals = np.std(chunk_preds_stacked, axis=0)
            
            mean_density[i:i+chunk_size] = mean_vals
            std_density[i:i+chunk_size] = std_vals
            
    elapsed_time = time.perf_counter() - start_time
    
    # Reshape back to 3D grid
    mean_density_3d = mean_density.reshape((128, 128, 128))
    std_density_3d = std_density.reshape((128, 128, 128))
    
    print(f"\n[+] Ensemble grid evaluation complete.")
    print(f"    Total Wall-Clock Inference Time: {elapsed_time:.4f} seconds")
    
    # 5. Output metrics and diagnostic logs
    tumor_mask = label > 0
    healthy_mask = (label == 0) & (mean_density_3d > 0.05) # normal tissue where model projects some growth
    
    print("\n--- Spatial Uncertainty Metrics ---")
    if np.any(tumor_mask):
        mean_std_tumor = np.mean(std_density_3d[tumor_mask])
        print(f"    Mean Uncertainty inside Tumor Core: {mean_std_tumor:.4f}")
    else:
        print("    Mean Uncertainty inside Tumor Core: N/A")
        
    if np.any(healthy_mask):
        mean_std_healthy = np.mean(std_density_3d[healthy_mask])
        print(f"    Mean Uncertainty in Infiltration Zone: {mean_std_healthy:.4f}")
    else:
        print("    Mean Uncertainty in Infiltration Zone: N/A")
        
    max_uncertainty = np.max(std_density_3d)
    print(f"    Maximum Per-Voxel Uncertainty:       {max_uncertainty:.4f}")
    
    # 6. Save results
    os.makedirs("outputs", exist_ok=True)
    out_file = f"outputs/{patient_id}_prediction_ensemble.npz"
    np.savez_compressed(
        out_file,
        mean_density=mean_density_3d,
        std_density=std_density_3d,
        spacing=spacing,
        inference_time=elapsed_time
    )
    print(f"\n[+] Saved ensemble mean and uncertainty volumes to {out_file}")

if __name__ == "__main__":
    main()
