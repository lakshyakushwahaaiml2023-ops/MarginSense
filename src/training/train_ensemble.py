import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import our MarginSense network and synthetic helper
from src.models.amortized_pinn import MarginSenseNet
from src.training.train import create_synthetic_patient_data, get_data_points, get_initial_points, get_collocation_points

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_data_points_cached(batch_size, rec_indices, bg_indices, device):
    """Stratified sampling using pre-computed indices (completely bypasses np.argwhere CPU bottleneck)."""
    half_batch = batch_size // 2
    
    # Sample recurrence indices
    if len(rec_indices) > 0:
        idx_r = np.random.choice(len(rec_indices), half_batch, replace=(len(rec_indices) < half_batch))
        sampled_r = rec_indices[idx_r]
    else:
        sampled_r = np.random.randint(0, 128, size=(half_batch, 3))
        
    # Sample background indices
    idx_b = np.random.choice(len(bg_indices), half_batch, replace=(len(bg_indices) < half_batch))
    sampled_b = bg_indices[idx_b]
    
    sampled_indices = np.concatenate([sampled_r, sampled_b], axis=0)
    target_c = np.concatenate([np.ones((half_batch, 1)), np.zeros((half_batch, 1))], axis=0)
    
    # Convert index coordinates to [0, 1] domain
    coords_xyz = sampled_indices / 128.0
    coords_t = np.ones((batch_size, 1)) # t = 1
    
    coords = np.concatenate([coords_xyz, coords_t], axis=1)
    
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    target_c_tensor = torch.tensor(target_c, dtype=torch.float32, device=device)
    
    return coords_tensor, target_c_tensor

def train_single_model(model_idx, seed, epochs, lr, cached_patients, device, lambda_data=1.0, lambda_ic=1.0, lambda_pde=0.1):
    """Trains a single amortized PINN model with a specific seed."""
    print(f"\n==================================================")
    print(f"[*] Starting training for Ensemble Model #{model_idx} (Seed: {seed})")
    print(f"==================================================")
    
    set_seed(seed)
    
    model = MarginSenseNet(embedding_dim=64, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    
    # TensorBoard setup for this specific model
    writer = SummaryWriter(log_dir=f"outputs/runs/ensemble_model_{model_idx}")
    
    start_time = time.perf_counter()
    
    # Copy/shuffle index references
    shuffled_indices = list(range(len(cached_patients)))
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        
        epoch_loss = 0.0
        epoch_loss_data = 0.0
        epoch_loss_ic = 0.0
        epoch_loss_pde = 0.0
        
        # Shuffle index order per epoch
        random.shuffle(shuffled_indices)
        
        model.train()
        for p_idx in shuffled_indices:
            p_data = cached_patients[p_idx]
            
            # Encoder input and covariates already in memory
            volume_tensor = p_data['volume_tensor']
            cov_tensor = p_data['cov_tensor']
            tissue_map_tensor = p_data['tissue_map_tensor']
            
            optimizer.zero_grad()
            
            # Encoder forward (mixed precision)
            with torch.amp.autocast('cuda'):
                z_embed, D_0, rho_0 = model.forward_encoder(volume_tensor, covariates=cov_tensor)
                
            z_embed = z_embed.float()
            D_0 = D_0.float()
            rho_0 = rho_0.float()
            
            # Tumor centroid for IC
            centroid = p_data['centroid']
            
            # Sample coordinate points using the pre-computed indices (extremely fast)
            coords_data, target_data = get_data_points_cached(
                batch_size=4096, 
                rec_indices=p_data['rec_indices'], 
                bg_indices=p_data['bg_indices'], 
                device=device
            )
            coords_ic, target_ic = get_initial_points(batch_size=2048, centroid=centroid, device=device)
            coords_pde = get_collocation_points(batch_size=4096, device=device)
            
            # Coordinate network evaluations
            z_embed_expanded_data = z_embed.expand(coords_data.size(0), -1)
            pred_data = model.forward_coordinate(coords_data, z_embed_expanded_data)
            loss_data = torch.mean((pred_data - target_data) ** 2)
            
            z_embed_expanded_ic = z_embed.expand(coords_ic.size(0), -1)
            pred_ic = model.forward_coordinate(coords_ic, z_embed_expanded_ic)
            loss_ic = torch.mean((pred_ic - target_ic) ** 2)
            
            z_embed_expanded_pde = z_embed.expand(coords_pde.size(0), -1)
            pred_pde = model.forward_coordinate(coords_pde, z_embed_expanded_pde)
            
            # Spatial derivatives via Autograd
            grads = torch.autograd.grad(pred_pde.sum(), coords_pde, create_graph=True)[0]
            dc_dx, dc_dy, dc_dz, dc_dt = grads[:, 0], grads[:, 1], grads[:, 2], grads[:, 3]
            
            # GPU healthy tissue lookup for spatially-varying diffusion
            grid_xyz = coords_pde[:, :3] * 128.0
            grid_indices = torch.clamp(grid_xyz, 0, 127).long()
            tissue_vals = tissue_map_tensor[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
            
            # Map tissue type to local diffusion multiplier (WM = 0.15, GM = 0.03, CSF = 0.0, Necrotic/ED = 0.075)
            diffusion_weights = torch.zeros(coords_pde.size(0), device=device)
            diffusion_weights[tissue_vals == 1] = 0.15
            diffusion_weights[tissue_vals == 2] = 0.03
            diffusion_weights[tissue_vals == 3] = 0.0
            diffusion_weights[tissue_vals == 0] = 0.075
            
            # Scale global D_0 by tissue weights
            D_local = D_0.squeeze() * diffusion_weights
            
            # Populate D(x) per voxel: shape (B, 3, 3)
            # Default isotropic tensor: d_scalar(x) * I
            I_matrix = torch.eye(3, device=device).unsqueeze(0)  # Shape (1, 3, 3)
            D_tensor = D_local.unsqueeze(1).unsqueeze(2) * I_matrix  # Shape (B, 3, 3)
            
            # Compute flux j = D(x) @ grad_c: shape (B, 3)
            grad_c = torch.stack([dc_dx, dc_dy, dc_dz], dim=1)
            flux = torch.bmm(D_tensor, grad_c.unsqueeze(2)).squeeze(2)  # Shape (B, 3)
            jx = flux[:, 0]
            jy = flux[:, 1]
            jz = flux[:, 2]
            
            # Divergence of flux: ∇·j = ∂jx/∂x + ∂jy/∂y + ∂jz/∂z
            djx_dx = torch.autograd.grad(jx.sum(), coords_pde, create_graph=True)[0][:, 0]
            djy_dy = torch.autograd.grad(jy.sum(), coords_pde, create_graph=True)[0][:, 1]
            djz_dz = torch.autograd.grad(jz.sum(), coords_pde, create_graph=True)[0][:, 2]
            
            div_flux = djx_dx + djy_dy + djz_dz
            
            # PDE residual using divergence of flux
            pde_residual = dc_dt - div_flux - rho_0.squeeze() * pred_pde.squeeze() * (1.0 - pred_pde.squeeze())
            loss_pde = torch.mean(pde_residual ** 2)
            
            # Total Loss
            loss = lambda_data * loss_data + lambda_ic * loss_ic + lambda_pde * loss_pde
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_loss_data += loss_data.item()
            epoch_loss_ic += loss_ic.item()
            epoch_loss_pde += loss_pde.item()
            
        num_patients = len(cached_patients)
        avg_loss = epoch_loss / num_patients
        avg_data = epoch_loss_data / num_patients
        avg_ic = epoch_loss_ic / num_patients
        avg_pde = epoch_loss_pde / num_patients
        
        # Log TensorBoard scalars
        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/data', avg_data, epoch)
        writer.add_scalar('Loss/ic', avg_ic, epoch)
        writer.add_scalar('Loss/pde', avg_pde, epoch)
        
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            epoch_time = time.perf_counter() - epoch_start
            print(f"  Model {model_idx} | Epoch {epoch:3d}/{epochs:3d} | Loss: {avg_loss:.5f} [Data: {avg_data:.5f}, PDE: {avg_pde:.5f}] | Time: {epoch_time:.2f}s")
            
    writer.close()
    
    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    checkpoint_path = f"outputs/marginsense_ensemble_{model_idx}.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'seed': seed
    }, checkpoint_path)
    print(f"[+] Saved checkpoint to {checkpoint_path}")
    
    if model_idx == 0:
        # Save final patient latent vectors for all training cases (from Model 0)
        print("  [*] Saving final patient latent vectors to lookup table from Model 0...")
        from src.models.amortized_pinn import save_patient_latent
        model.eval()
        with torch.no_grad():
            for p_data in cached_patients:
                p_id = p_data['patient_id']
                p_z, _, _ = model.forward_encoder(p_data['volume_tensor'], covariates=p_data['cov_tensor'])
                save_patient_latent(p_id, p_z)
    
    total_time = time.perf_counter() - start_time
    print(f"[+] Finished training Model {model_idx} in {total_time:.2f}s")
    return checkpoint_path

import random

def main():
    parser = argparse.ArgumentParser(description="MarginSense Ensemble Training Loop")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic multi-patient files and verify training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs per model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()
    
    # 1. GPU Check
    if not torch.cuda.is_available():
        print("[CRITICAL] CUDA is not available! Model training must run on local GPU.")
        sys.exit(1)
        
    device = torch.device('cuda')
    print(f"[*] Running on local GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Synthetic Data Setup (if requested)
    if args.synthetic:
        print("[*] Setting up synthetic patient dataset...")
        create_synthetic_patient_data("synthetic_patient_1", (60, 60, 60), 12)
        create_synthetic_patient_data("synthetic_patient_2", (64, 68, 60), 15)
        create_synthetic_patient_data("synthetic_patient_3", (70, 55, 65), 13)
        
    # 3. Load Patient Datasets
    processed_dir = "data/processed"
    patient_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) 
                     if f.startswith("brats_") and f.endswith(".npz")]
    if not patient_files:
        patient_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith(".npz")]
    
    if not patient_files:
        print("[-] No preprocessed patient files found in data/processed.")
        print("    Please run preprocess.py first, or use the --synthetic flag.")
        sys.exit(1)
        
    print(f"[*] Pre-loading and caching all {len(patient_files)} patient datasets in memory...")
    cached_patients = []
    
    for idx, p_file in enumerate(patient_files):
        p_id = os.path.basename(p_file).replace(".npz", "")
        # Load and decompress in CPU once
        data = np.load(p_file)
        image = data['image']
        label = data['label']
        recurrence = data['recurrence']
        tissue_map = data['tissue_map']
        
        # Pre-compute static index arrays
        rec_indices = np.argwhere(recurrence > 0)
        bg_indices = np.argwhere(recurrence == 0)
        
        tumor_indices = np.argwhere(label > 0)
        centroid = np.mean(tumor_indices, axis=0) / 128.0 if len(tumor_indices) > 0 else np.array([0.5, 0.5, 0.5])
        
        # 5-channel encoder input
        volume_in = np.concatenate([image, np.expand_dims(label, axis=0)], axis=0)
        volume_tensor = torch.tensor(volume_in, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Load covariates
        from src.models.amortized_pinn import load_covariate_vector
        cov_vec = load_covariate_vector(p_id, npz_data=data)
        cov_tensor = torch.tensor(cov_vec, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Store as GPU tensor
        tissue_map_tensor = torch.tensor(tissue_map, dtype=torch.int8, device=device)
        
        cached_patients.append({
            'patient_id': p_id,
            'rec_indices': rec_indices,
            'bg_indices': bg_indices,
            'centroid': centroid,
            'volume_tensor': volume_tensor,
            'cov_tensor': cov_tensor,
            'tissue_map_tensor': tissue_map_tensor
        })
        if (idx + 1) % 20 == 0 or (idx + 1) == len(patient_files):
            print(f"    Loaded {idx+1}/{len(patient_files)} patients...")
            
    # Seeds for the 5 ensemble models
    seeds = [42, 43, 44, 45, 46]
    
    start_ensemble_time = time.perf_counter()
    
    # Train 5 models
    for idx, seed in enumerate(seeds):
        train_single_model(
            model_idx=idx,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            cached_patients=cached_patients,
            device=device
        )
        
    total_ensemble_time = time.perf_counter() - start_ensemble_time
    print(f"\n[+] Ensemble Training Complete! Trained 5 models in {total_ensemble_time:.2f} seconds.")

if __name__ == "__main__":
    main()
