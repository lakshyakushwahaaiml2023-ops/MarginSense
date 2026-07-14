import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.ndimage import zoom

# Import our MarginSense network
from src.models.amortized_pinn import MarginSenseNet

def main():
    parser = argparse.ArgumentParser(description="MarginSense Visualizer Data Exporter")
    parser.add_argument("patient_id", type=str, nargs="?", default="synthetic_patient_2",
                        help="Patient ID to export.")
    args = parser.parse_args()
    
    patient_id = args.patient_id
    processed_path = f"data/processed/{patient_id}.npz"
    uniform_out = f"outputs/{patient_id}_baseline_uniform.npz"
    ensemble_out = f"outputs/{patient_id}_prediction_ensemble.npz"
    
    # Check dependencies
    if not os.path.exists(processed_path):
        print(f"[Error] Preprocessed data not found for patient {patient_id} at {processed_path}.")
        sys.exit(1)
        
    print(f"[*] Exporting visualizer data for patient: {patient_id}")
    
    # 1. Load Ground Truth and Baseline Data
    data_gt = np.load(processed_path)
    image = data_gt['image']
    label = data_gt['label']
    spacing = data_gt['spacing'] if 'spacing' in data_gt else np.array([1.0, 1.0, 1.0])
    
    # Brain mask: non-zero regions in FLAIR (channel 3)
    brain_mask_128 = (image[3] > 0).astype(np.int8)
    
    # Load Uniform baseline
    if not os.path.exists(uniform_out):
        print(f"[Warning] Uniform margin output not found at {uniform_out}. Run compare_models.py first.")
        uniform_mask_128 = np.zeros_like(label)
    else:
        uniform_mask_128 = np.load(uniform_out)['dilated_mask']
        
    # Load Ensemble predictions (for uncertainty)
    if not os.path.exists(ensemble_out):
        print(f"[Warning] Ensemble prediction output not found at {ensemble_out}. Run evaluate_uncertainty.py first.")
        std_density_128 = np.zeros_like(label, dtype=np.float32)
    else:
        std_density_128 = np.load(ensemble_out)['std_density']
        
    # 2. Downsample Static Masks to 32x32x32
    print("[*] Downsampling masks to 32x32x32 grid...")
    target_shape = (32, 32, 32)
    zoom_factors = [t / s for t, s in zip(target_shape, label.shape)]
    
    # Nearest neighbor (order=0) for binary masks
    brain_mask_32 = zoom(brain_mask_128, zoom_factors, order=0).astype(np.int8)
    tumor_mask_32 = zoom(label, zoom_factors, order=0).astype(np.int8)
    uniform_margin_32 = zoom(uniform_mask_128, zoom_factors, order=0).astype(np.int8)
    
    # Bilinear (order=1) for continuous uncertainty std deviation
    std_density_32 = zoom(std_density_128, zoom_factors, order=1).astype(np.float32)
    
    # 3. Generate Temporal Density Predictions from Ensemble
    print("[*] Generating temporal density slices at t = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]...")
    temporal_timepoints = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    temporal_densities_32 = []
    
    # Set up GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load 5 ensemble models
    models = []
    has_models = True
    for i in range(5):
        ckpt_path = f"outputs/marginsense_ensemble_{i}.pt"
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Model checkpoint {ckpt_path} missing. Using zero-filled grids for temporal animation.")
            has_models = False
            break
        model = MarginSenseNet(embedding_dim=64, hidden_dim=64).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        
    # Generate downsampled coordinate grid
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    z = np.linspace(0, 1, 32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    
    if has_models:
        # Precompute patient embeddings
        volume_in = np.concatenate([image, np.expand_dims(label, axis=0)], axis=0)
        volume_tensor = torch.tensor(volume_in, dtype=torch.float32, device=device).unsqueeze(0)
        
        embeddings = []
        with torch.no_grad():
            for model in models:
                z_embed, _, _ = model.forward_encoder(volume_tensor)
                embeddings.append(z_embed.float())
                
        # Evaluate for each timepoint
        for t in temporal_timepoints:
            coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones_like(xx.ravel()) * t], axis=1)
            coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
            
            # Predict
            t_preds = []
            with torch.no_grad():
                for m_idx, model in enumerate(models):
                    z_embed = embeddings[m_idx]
                    z_expanded = z_embed.expand(coords_tensor.size(0), -1)
                    pred = model.coordinate_mlp(coords_tensor, z_expanded)
                    t_preds.append(pred.cpu().numpy().squeeze())
                    
            # Compute average density
            mean_t = np.mean(np.stack(t_preds, axis=0), axis=0)
            temporal_densities_32.append(mean_t.reshape(target_shape))
            print(f"    Completed t = {t:.1f}")
    else:
        # Fallback if checkpoints are missing
        for t in temporal_timepoints:
            temporal_densities_32.append(np.zeros(target_shape, dtype=np.float32))
            
    # 4. Serialize to data.js file
    print("[*] Serializing data to src/viz/data.js...")
    
    data_dict = {
        "patient_id": patient_id,
        "shape": list(target_shape),
        "spacing": spacing.tolist(),
        "brain_mask": brain_mask_32.flatten().tolist(),
        "tumor_mask": tumor_mask_32.flatten().tolist(),
        "uniform_margin": uniform_margin_32.flatten().tolist(),
        "temporal_densities": [grid.flatten().tolist() for grid in temporal_densities_32],
        "uncertainty": std_density_32.flatten().tolist()
    }
    
    os.makedirs("src/viz", exist_ok=True)
    out_js_path = "src/viz/data.js"
    with open(out_js_path, "w") as f:
        f.write(f"window.patientData = {json.dumps(data_dict)};")
        
    print(f"[+] Saved visualization data successfully to {out_js_path}")

if __name__ == "__main__":
    main()
