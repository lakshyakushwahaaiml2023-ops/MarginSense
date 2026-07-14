import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    
    # Isotropic spacing
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return label, spacing

class VanillaPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Input is (x, y, z, t)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Trainable physics parameters (initialized to reasonable starting values in log-space)
        self.log_D = nn.Parameter(torch.tensor(-4.6))    # log(0.01) -> D ~ 0.01
        self.log_rho = nn.Parameter(torch.tensor(-1.2))  # log(0.3) -> rho ~ 0.3
        
    def forward(self, coords):
        return self.net(coords)
        
    @property
    def D(self):
        return torch.exp(self.log_D)
        
    @property
    def rho(self):
        return torch.exp(self.log_rho)

def get_collocation_points(batch_size, device):
    """Generates random points in the spatio-temporal domain [0, 1]^3 x [0, 1]."""
    coords = torch.rand(batch_size, 4, device=device)
    coords.requires_grad = True
    return coords

def get_initial_points(batch_size, centroid, device, sigma=0.05):
    """Generates points at t=0 with Gaussian initial condition target."""
    coords = torch.rand(batch_size, 4, device=device)
    coords[:, 3] = 0.0 # t = 0
    
    # Compute Gaussian density target centered at tumor centroid
    centroid_tensor = torch.tensor(centroid, device=device, dtype=torch.float32)
    dist_sq = torch.sum((coords[:, :3] - centroid_tensor) ** 2, dim=1)
    target_c = torch.exp(-dist_sq / (2.0 * sigma**2)).unsqueeze(1)
    
    return coords, target_c

def get_data_points(batch_size, label, device):
    """Stratified sampling of points at t=1: 50% tumor, 50% background."""
    tumor_indices = np.argwhere(label > 0)
    bg_indices = np.argwhere(label == 0)
    
    half_batch = batch_size // 2
    
    # Sample tumor indices
    if len(tumor_indices) > 0:
        idx_t = np.random.choice(len(tumor_indices), half_batch, replace=(len(tumor_indices) < half_batch))
        sampled_t = tumor_indices[idx_t]
    else:
        # Fallback if no tumor (should not happen in practice)
        sampled_t = np.random.randint(0, 128, size=(half_batch, 3))
        
    # Sample background indices
    idx_b = np.random.choice(len(bg_indices), half_batch, replace=(len(bg_indices) < half_batch))
    sampled_b = bg_indices[idx_b]
    
    sampled_indices = np.concatenate([sampled_t, sampled_b], axis=0) # shape (batch_size, 3)
    target_c = np.concatenate([np.ones((half_batch, 1)), np.zeros((half_batch, 1))], axis=0)
    
    # Convert index coordinates to [0, 1] domain
    coords_xyz = sampled_indices / 128.0
    coords_t = np.ones((batch_size, 1)) # t = 1
    
    coords = np.concatenate([coords_xyz, coords_t], axis=1)
    
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    target_c_tensor = torch.tensor(target_c, dtype=torch.float32, device=device)
    
    return coords_tensor, target_c_tensor

def main():
    parser = argparse.ArgumentParser(description="Vanilla Patient-Specific PINN Fitting Baseline")
    parser.add_argument("patient_id", type=str, nargs="?", default="synthetic_patient",
                        help="Patient ID to process. Ignored if --synthetic is set.")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test data.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of gradient descent iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()
    
    # 1. Device Setup
    if not torch.cuda.is_available():
        print("[CRITICAL] CUDA is not available! As per requirements, training must run on local GPU.")
        print("Stopping execution.")
        sys.exit(1)
        
    device = torch.device('cuda')
    print(f"[*] Running on local GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] CUDA Device confirmed: {device}")
    
    # 2. Data Loading
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
        spacing = data['spacing'] if 'spacing' in data else np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # 3. Centroid calculation for IC seed
    tumor_indices = np.argwhere(label > 0)
    if len(tumor_indices) > 0:
        centroid = np.mean(tumor_indices, axis=0) / 128.0
    else:
        centroid = np.array([0.5, 0.5, 0.5])
    print(f"[*] Tumor centroid: {centroid}")
    
    # 4. Model & Optimizer Setup
    model = VanillaPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    print(f"[*] Starting vanilla PINN optimization for {args.epochs} epochs...")
    start_time = time.perf_counter()
    
    lambda_data = 1.0
    lambda_ic = 1.0
    lambda_pde = 0.1
    
    # Monitor GPU memory usage
    initial_gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"[*] Initial GPU memory allocated: {initial_gpu_mem:.2f} MB")
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        # 5a. Data Loss (t = 1)
        coords_data, target_data = get_data_points(batch_size=4096, label=label, device=device)
        pred_data = model(coords_data)
        loss_data = torch.mean((pred_data - target_data) ** 2)
        
        # 5b. Initial Condition Loss (t = 0)
        coords_ic, target_ic = get_initial_points(batch_size=2048, centroid=centroid, device=device)
        pred_ic = model(coords_ic)
        loss_ic = torch.mean((pred_ic - target_ic) ** 2)
        
        # 5c. PDE Collocation Loss (t in [0, 1])
        coords_pde = get_collocation_points(batch_size=4096, device=device)
        pred_pde = model(coords_pde)
        
        # Calculate gradients using Autograd
        grads = torch.autograd.grad(pred_pde.sum(), coords_pde, create_graph=True)[0]
        dc_dx = grads[:, 0]
        dc_dy = grads[:, 1]
        dc_dz = grads[:, 2]
        dc_dt = grads[:, 3]
        
        # Second derivatives for spatial Laplacian
        d2c_dx2 = torch.autograd.grad(dc_dx.sum(), coords_pde, create_graph=True)[0][:, 0]
        d2c_dy2 = torch.autograd.grad(dc_dy.sum(), coords_pde, create_graph=True)[0][:, 1]
        d2c_dz2 = torch.autograd.grad(dc_dz.sum(), coords_pde, create_graph=True)[0][:, 2]
        laplacian = d2c_dx2 + d2c_dy2 + d2c_dz2
        
        # Fisher-Kolmogorov reaction diffusion PDE
        # dc/dt = D * laplacian + rho * c * (1 - c)
        D_val = model.D
        rho_val = model.rho
        pde_residual = dc_dt - D_val * laplacian - rho_val * pred_pde.squeeze() * (1.0 - pred_pde.squeeze())
        loss_pde = torch.mean(pde_residual ** 2)
        
        # Total loss
        loss = lambda_data * loss_data + lambda_ic * loss_ic + lambda_pde * loss_pde
        
        loss.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}/{args.epochs:4d} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Data: {loss_data.item():.6f} | "
                  f"IC: {loss_ic.item():.6f} | "
                  f"PDE: {loss_pde.item():.6f} | "
                  f"D: {D_val.item():.5f} | "
                  f"rho: {rho_val.item():.5f}")
            
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    final_gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"\n[+] Optimization Complete!")
    print(f"    Total Training Time: {elapsed_time:.2f} seconds")
    print(f"    Final GPU memory allocated: {final_gpu_mem:.2f} MB")
    print(f"    Estimated D: {model.D.item():.6f}")
    print(f"    Estimated rho: {model.rho.item():.6f}")
    
    # 6. Evaluate full 3D grid at t = 1
    print("[*] Generating final density volume on full 128x128x128 grid...")
    grid_density = np.zeros((128, 128, 128), dtype=np.float32)
    
    # Coordinate grid generation
    x = np.linspace(0, 1, 128)
    y = np.linspace(0, 1, 128)
    z = np.linspace(0, 1, 128)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    # Reshape grid coordinates and append t=1
    coords_grid = np.stack([
        xx.ravel(), 
        yy.ravel(), 
        zz.ravel(), 
        np.ones_like(xx.ravel())
    ], axis=1)
    
    # Memory-safe batch evaluation (chunking to prevent CUDA OOM)
    chunk_size = 131072
    model.eval()
    with torch.no_grad():
        for i in range(0, len(coords_grid), chunk_size):
            chunk = coords_grid[i:i+chunk_size]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)
            pred = model(chunk_tensor)
            grid_density.ravel()[i:i+chunk_size] = pred.cpu().numpy().squeeze()
            
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    out_file = f"outputs/{patient_id}_baseline_pinn.npz"
    np.savez_compressed(
        out_file,
        density=grid_density,
        elapsed_time=elapsed_time,
        final_loss=loss.item(),
        D=model.D.item(),
        rho=model.rho.item(),
        spacing=spacing
    )
    print(f"[+] Saved baseline PINN results to {out_file}")

if __name__ == "__main__":
    main()
