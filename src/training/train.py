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

# Import our MarginSense network
from src.models.amortized_pinn import MarginSenseNet

def create_synthetic_patient_data(patient_id, center, radius):
    """Generates synthetic patient files with anisotropic growth along X-axis tracts."""
    os.makedirs("data/processed", exist_ok=True)
    out_file = f"data/processed/{patient_id}.npz"
    print(f"[*] Generating synthetic patient data at {out_file}...")
    
    shape = (128, 128, 128)
    image = np.zeros((4, 128, 128, 128), dtype=np.float32)
    label = np.zeros(shape, dtype=np.int8)
    recurrence = np.zeros(shape, dtype=np.int8)
    tissue_map = np.zeros(shape, dtype=np.int8)
    
    # Create simple ellipsoidal brain mask
    bz, by, bx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_brain = ((bx - 64)/55)**2 + ((by - 64)/55)**2 + ((bz - 64)/45)**2
    brain_mask = dist_brain <= 1.0
    
    # Add dummy intensities in brain area for MRI channels (T1, T1ce, T2, FLAIR)
    image[0, brain_mask] = 0.5 + 0.1 * np.random.randn(*image[0, brain_mask].shape) # T1
    image[1, brain_mask] = 0.2 + 0.05 * np.random.randn(*image[1, brain_mask].shape) # T1ce
    image[2, brain_mask] = 0.8 + 0.1 * np.random.randn(*image[2, brain_mask].shape) # T2
    image[3, brain_mask] = 0.6 + 0.1 * np.random.randn(*image[3, brain_mask].shape) # FLAIR
    
    # Anisotropic Ellipsoidal Tumor Core (elongated along X)
    cx, cy, cz = center
    rx_tumor = 1.6 * radius
    ry_tumor = radius
    rz_tumor = radius
    dist_tumor = ((bx - cx)/rx_tumor)**2 + ((by - cy)/ry_tumor)**2 + ((bz - cz)/rz_tumor)**2
    label[dist_tumor <= 1.0] = 1 # Active tumor core
    
    # Edema shell
    dist_edema = ((bx - cx)/(rx_tumor + 4))**2 + ((by - cy)/(ry_tumor + 4))**2 + ((bz - cz)/(rz_tumor + 4))**2
    label[(label == 0) & (dist_edema <= 1.0)] = 2
    
    # Recurrence target (grows extensively along the white matter tract on the X-axis)
    rx_rec = rx_tumor + 12.0 # large expansion along X
    ry_rec = ry_tumor + 3.0  # small expansion along Y
    rz_rec = rz_tumor + 3.0  # small expansion along Z
    dist_rec = ((bx - cx)/rx_rec)**2 + ((by - cy)/ry_rec)**2 + ((bz - cz)/rz_rec)**2
    recurrence[dist_rec <= 1.0] = 1
    
    # Tissue types inside brain
    # Simulate a central white matter corridor along X-axis
    wm_corridor = (np.abs(by - cy) <= 12) & (np.abs(bz - cz) <= 12)
    tissue_map[brain_mask] = 2 # Default to Gray Matter (slow diffusion)
    tissue_map[brain_mask & wm_corridor] = 1 # White Matter (fast diffusion corridor)
    
    # CSF inner ring
    dist_center = np.sqrt((bx - 64)**2 + (by - 64)**2 + (bz - 64)**2)
    tissue_map[brain_mask & (dist_center <= 16)] = 3
    
    # Tumor overrides tissue map
    tissue_map[label > 0] = 0
    
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    np.savez_compressed(
        out_file,
        image=image,
        label=label,
        spacing=spacing,
        recurrence=recurrence,
        tissue_map=tissue_map
    )

def get_data_points(batch_size, label, recurrence, device):
    """Stratified sampling of points at t=1: 50% tumor recurrence, 50% background."""
    rec_indices = np.argwhere(recurrence > 0)
    bg_indices = np.argwhere(recurrence == 0)
    
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

def get_initial_points(batch_size, centroid, device, sigma=0.05):
    """Generates points at t=0 with Gaussian initial condition target."""
    coords = torch.rand(batch_size, 4, device=device)
    coords[:, 3] = 0.0 # t = 0
    
    centroid_tensor = torch.tensor(centroid, device=device, dtype=torch.float32)
    dist_sq = torch.sum((coords[:, :3] - centroid_tensor) ** 2, dim=1)
    target_c = torch.exp(-dist_sq / (2.0 * sigma**2)).unsqueeze(1)
    
    return coords, target_c

def get_collocation_points(batch_size, device):
    """Generates random points in the spatio-temporal domain [0, 1]^3 x [0, 1]."""
    coords = torch.rand(batch_size, 4, device=device)
    coords.requires_grad = True
    return coords

def main():
    parser = argparse.ArgumentParser(description="MarginSense Amortized PINN Training Loop")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic multi-patient files and verify training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()
    
    # 1. GPU Check
    if not torch.cuda.is_available():
        print("[CRITICAL] CUDA is not available! Model training must run on local GPU.")
        sys.exit(1)
        
    device = torch.device('cuda')
    print(f"[*] Running on local GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] CUDA Device confirmed: {device}")
    
    # 2. Synthetic Data Setup (if requested)
    if args.synthetic:
        print("[*] Setting up synthetic patient dataset...")
        # Create 3 synthetic patients with different tumors
        create_synthetic_patient_data("synthetic_patient_1", (60, 60, 60), 12)
        create_synthetic_patient_data("synthetic_patient_2", (64, 68, 60), 15)
        create_synthetic_patient_data("synthetic_patient_3", (70, 55, 65), 13)
        
    # 3. Load Patient Datasets
    processed_dir = "data/processed"
    patient_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith(".npz")]
    
    if not patient_files:
        print("[-] No preprocessed patient files found in data/processed.")
        print("    Please run preprocess.py first, or use the --synthetic flag.")
        sys.exit(1)
        
    print(f"[*] Loaded {len(patient_files)} patient file(s) for training.")
    
    # 4. Model and Logging Setup
    model = MarginSenseNet(embedding_dim=64, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Use AMP scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Logs setup
    os.makedirs("outputs", exist_ok=True)
    csv_file_path = "outputs/train_logs.csv"
    writer = SummaryWriter(log_dir="outputs/runs")
    
    # Initialize CSV header
    with open(csv_file_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['epoch', 'loss', 'loss_data', 'loss_ic', 'loss_pde', 'D_mean', 'rho_mean', 'time_elapsed'])
        
    start_time = time.perf_counter()
    
    # Loss scaling weights
    lambda_data = 1.0
    lambda_ic = 1.0
    lambda_pde = 0.1
    
    print("[*] Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        
        # Accumulate metrics across all patients in this epoch
        epoch_loss = 0.0
        epoch_loss_data = 0.0
        epoch_loss_ic = 0.0
        epoch_loss_pde = 0.0
        epoch_D = []
        epoch_rho = []
        
        # Shuffle patient files per epoch
        np.random.shuffle(patient_files)
        
        model.train()
        for p_file in patient_files:
            # Load patient cached data
            data = np.load(p_file)
            image = data['image']           # Shape (4, 128, 128, 128)
            label = data['label']           # Shape (128, 128, 128)
            recurrence = data['recurrence'] # Shape (128, 128, 128)
            tissue_map = data['tissue_map'] # Shape (128, 128, 128)
            
            # Formulate 5-channel encoder input: stacked image channels + label
            volume_in = np.concatenate([image, np.expand_dims(label, axis=0)], axis=0) # shape (5, 128, 128, 128)
            volume_tensor = torch.tensor(volume_in, dtype=torch.float32, device=device).unsqueeze(0) # shape (1, 5, 128, 128, 128)
            
            # Load covariates (11 features total: 6 manual + 5 auto-computed)
            # RETRAIN_REQUIRED: Training must include the 11-dimensional covariate vector
            from src.models.amortized_pinn import load_covariate_vector
            patient_id = os.path.basename(p_file).replace(".npz", "")
            cov_vec = load_covariate_vector(patient_id, npz_data=data)
            cov_tensor = torch.tensor(cov_vec, dtype=torch.float32, device=device).unsqueeze(0)

            # Healthy tissue map tensor for GPU lookup
            tissue_map_tensor = torch.tensor(tissue_map, dtype=torch.int8, device=device)
            
            optimizer.zero_grad()
            
            # --- 1. Mixed Precision CNN Encoder Pass ---
            with torch.amp.autocast('cuda'):
                z_embed, D_0, rho_0 = model.forward_encoder(volume_tensor, covariates=cov_tensor)
                
            # Cast latent values to float32 for coordinate network and double-grad autograd stability
            z_embed = z_embed.float()
            D_0 = D_0.float()
            rho_0 = rho_0.float()
            
            epoch_D.append(D_0.item())
            epoch_rho.append(rho_0.item())
            
            # Calculate tumor centroid for IC seed
            tumor_indices = np.argwhere(label > 0)
            centroid = np.mean(tumor_indices, axis=0) / 128.0 if len(tumor_indices) > 0 else np.array([0.5, 0.5, 0.5])
            
            # Sample coordinate points on the fly
            coords_data, target_data = get_data_points(batch_size=2048, label=label, recurrence=recurrence, device=device)
            coords_ic, target_ic = get_initial_points(batch_size=1024, centroid=centroid, device=device)
            coords_pde = get_collocation_points(batch_size=2048, device=device)
            
            # --- 2. Float32 Coordinate Network forward pass ---
            z_embed_expanded_data = z_embed.expand(coords_data.size(0), -1)
            pred_data = model.forward_coordinate(coords_data, z_embed_expanded_data)
            loss_data = torch.mean((pred_data - target_data) ** 2)
            
            z_embed_expanded_ic = z_embed.expand(coords_ic.size(0), -1)
            pred_ic = model.forward_coordinate(coords_ic, z_embed_expanded_ic)
            loss_ic = torch.mean((pred_ic - target_ic) ** 2)
            
            z_embed_expanded_pde = z_embed.expand(coords_pde.size(0), -1)
            pred_pde = model.forward_coordinate(coords_pde, z_embed_expanded_pde)
            
            # Compute spatial derivatives via Autograd
            grads = torch.autograd.grad(pred_pde.sum(), coords_pde, create_graph=True)[0]
            dc_dx = grads[:, 0]
            dc_dy = grads[:, 1]
            dc_dz = grads[:, 2]
            dc_dt = grads[:, 3]
            
            # Look up physical tissue types on GPU
            grid_xyz = coords_pde[:, :3] * 128.0
            grid_indices = torch.clamp(grid_xyz, 0, 127).long()
            tissue_vals = tissue_map_tensor[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
            
            # Map tissue type to local diffusion multiplier (WM = 0.15, GM = 0.03, CSF = 0.0, Necrotic/ED = 0.075)
            # grounded in clinical literature (Giese et al., Swanson glioma model)
            diffusion_weights = torch.zeros(coords_pde.size(0), device=device)
            diffusion_weights[tissue_vals == 1] = 0.15
            diffusion_weights[tissue_vals == 2] = 0.03
            diffusion_weights[tissue_vals == 3] = 0.0
            diffusion_weights[tissue_vals == 0] = 0.075
            
            # Scale global D_0 by tissue weights
            D_local = D_0.squeeze() * diffusion_weights
            
            # --- Generic Full Tensor Diffusion: ∇·(D(x)∇c) ---
            # grad_c has shape (B, 3) where columns are [dc_dx, dc_dy, dc_dz]
            grad_c = torch.stack([dc_dx, dc_dy, dc_dz], dim=1)
            
            # Populate D(x) per voxel: shape (B, 3, 3)
            # Default isotropic tensor: d_scalar(x) * I
            I_matrix = torch.eye(3, device=device).unsqueeze(0)  # Shape (1, 3, 3)
            D_tensor = D_local.unsqueeze(1).unsqueeze(2) * I_matrix  # Shape (B, 3, 3)
            
            # --- DTI TENSOR EXTENSION POINT ---
            # Placeholder code path for supplying a true DTI-derived tensor per voxel.
            # When DTI data is available, this block can be enabled to load a 3x3 diffusion tensor
            # mapping anisotropic orientation (e.g. from fractional anisotropy / main eigenvector direction)
            # rather than falling back to the isotropic d_scalar(x) * I.
            dti_data_available = False  # Set to True when DTI data is integrated
            if dti_data_available:
                # Example placeholder: D_tensor = load_dti_tensor_at_coords(coords_pde)
                # D_tensor should be a batch of symmetric positive-definite 3x3 tensors (B, 3, 3)
                pass
            # ----------------------------------
            
            # Compute flux j = D(x) @ grad_c: shape (B, 3)
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
            
            # Combine losses
            loss = lambda_data * loss_data + lambda_ic * loss_ic + lambda_pde * loss_pde
            
            # Backward pass using gradient scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate logs
            epoch_loss += loss.item()
            epoch_loss_data += loss_data.item()
            epoch_loss_ic += loss_ic.item()
            epoch_loss_pde += loss_pde.item()
            
        # Average epoch values
        num_patients = len(patient_files)
        avg_loss = epoch_loss / num_patients
        avg_data = epoch_loss_data / num_patients
        avg_ic = epoch_loss_ic / num_patients
        avg_pde = epoch_loss_pde / num_patients
        mean_D = np.mean(epoch_D)
        mean_rho = np.mean(epoch_rho)
        
        epoch_time = time.perf_counter() - epoch_start
        total_time = time.perf_counter() - start_time
        
        # Log to TensorBoard
        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/data', avg_data, epoch)
        writer.add_scalar('Loss/ic', avg_ic, epoch)
        writer.add_scalar('Loss/pde', avg_pde, epoch)
        writer.add_scalar('Physics/D_mean', mean_D, epoch)
        writer.add_scalar('Physics/rho_mean', mean_rho, epoch)
        
        # Log to CSV
        with open(csv_file_path, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, avg_loss, avg_data, avg_ic, avg_pde, mean_D, mean_rho, total_time])
            
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d}/{args.epochs:4d} | "
                  f"Loss: {avg_loss:.5f} [Data: {avg_data:.5f}, IC: {avg_ic:.5f}, PDE: {avg_pde:.5f}] | "
                  f"D: {mean_D:.6f} | rho: {mean_rho:.5f} | "
                  f"Time: {epoch_time:.2f}s")
            
    writer.close()
    
    # Save checkpoint
    checkpoint_path = "outputs/marginsense_checkpoint.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'D_mean': mean_D,
        'rho_mean': mean_rho
    }, checkpoint_path)
    print(f"\n[+] Saved model checkpoint to {checkpoint_path}")

    # Save final patient latent vectors for all training cases
    print("\n[*] Saving final patient latent vectors to lookup table...")
    from src.models.amortized_pinn import save_patient_latent
    model.eval()
    with torch.no_grad():
        for p_file in patient_files:
            p_id = os.path.basename(p_file).replace(".npz", "")
            p_data = np.load(p_file)
            p_img = p_data['image']
            p_lbl = p_data['label']
            p_vol_in = np.concatenate([p_img, np.expand_dims(p_lbl, axis=0)], axis=0)
            p_vol_tensor = torch.tensor(p_vol_in, dtype=torch.float32, device=device).unsqueeze(0)
            p_cov_vec = load_covariate_vector(p_id, npz_data=p_data)
            p_cov_tensor = torch.tensor(p_cov_vec, dtype=torch.float32, device=device).unsqueeze(0)
            
            p_z, _, _ = model.forward_encoder(p_vol_tensor, covariates=p_cov_tensor)
            save_patient_latent(p_id, p_z)
    
    # 5. Single forward pass inference verification
    print("\n--- Running Single Forward Pass Inference Verification ---")
    test_file = patient_files[0]
    patient_id = os.path.basename(test_file).replace(".npz", "")
    print(f"[*] Testing inference on patient: {patient_id}")
    
    # Load patient volume
    data = np.load(test_file)
    image = data['image']
    label = data['label']
    spacing = data['spacing'] if 'spacing' in data else np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    volume_in = np.concatenate([image, np.expand_dims(label, axis=0)], axis=0)
    volume_tensor = torch.tensor(volume_in, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Measure inference time
    model.eval()
    inf_start = time.perf_counter()
    with torch.no_grad():
        # Get patient-specific embedding and growth parameters
        cov_vec = load_covariate_vector(patient_id, npz_data=data)
        cov_tensor = torch.tensor(cov_vec, dtype=torch.float32, device=device).unsqueeze(0)
        z_embed, D_est, rho_est = model.forward_encoder(volume_tensor, covariates=cov_tensor)
        
        # Save patient latent
        from src.models.amortized_pinn import save_patient_latent
        save_patient_latent(patient_id, z_embed)
        
        # Evaluate full 128x128x128 grid at t = 1
        grid_density = np.zeros((128, 128, 128), dtype=np.float32)
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 1, 128)
        z = np.linspace(0, 1, 128)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        coords_grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones_like(xx.ravel())], axis=1)
        
        # Evaluation in chunks (prevents OOM on local GPU)
        z_expanded = z_embed.float().expand(131072, -1)
        chunk_size = 131072
        for i in range(0, len(coords_grid), chunk_size):
            chunk = coords_grid[i:i+chunk_size]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)
            # If the last batch is smaller than chunk_size, match dimensions
            if chunk_tensor.size(0) != z_expanded.size(0):
                z_expanded = z_embed.float().expand(chunk_tensor.size(0), -1)
            pred = model.coordinate_mlp(chunk_tensor, z_expanded)
            grid_density.ravel()[i:i+chunk_size] = pred.cpu().numpy().squeeze()
            
    inf_end = time.perf_counter()
    inf_time = inf_end - inf_start
    
    print(f"[+] Inference complete for patient: {patient_id}")
    print(f"    Estimated physical D: {D_est.item():.6f}")
    print(f"    Estimated physical rho: {rho_est.item():.6f}")
    print(f"    Total Inference Grid Evaluation Time: {inf_time:.4f} seconds")
    
    # Save inference output
    pred_out_file = f"outputs/{patient_id}_prediction.npz"
    np.savez_compressed(
        pred_out_file,
        density=grid_density,
        spacing=spacing,
        D_est=D_est.item(),
        rho_est=rho_est.item(),
        inference_time=inf_time
    )
    print(f"[+] Saved predicted density grid to {pred_out_file}")

if __name__ == "__main__":
    main()
