import torch
import torch.nn as nn

class Encoder3D(nn.Module):
    """
    3D CNN Encoder that ingests the patient's 5-channel 3D volume (4 MRI modalities + 1 segmentation)
    and outputs a patient latent representation z.
    """
    def __init__(self, in_channels=5, embedding_dim=64):
        super().__init__()
        
        self.conv = nn.Sequential(
            # Input: (B, 5, 128, 128, 128)
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1), # (B, 16, 64, 64, 64)
            nn.BatchNorm3d(16),
            nn.ReLU(),
            
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),          # (B, 32, 32, 32, 32)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),          # (B, 64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),          # (B, 64, 8, 8, 8)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool3d(1)                                         # (B, 64, 1, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Latent patient embedding projection
        self.fc_embed = nn.Linear(64, embedding_dim)
        
        # Project covariates (shape Bx11) to 64 dims
        # RETRAIN_REQUIRED: Changed input dimensions from 6 to 11
        self.cov_proj = nn.Linear(11, 64)
        nn.init.zeros_(self.cov_proj.weight)
        nn.init.zeros_(self.cov_proj.bias)
        
    def forward(self, x, covariates=None):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        
        if covariates is not None:
            features = features + self.cov_proj(covariates)
            
        z_embed = self.fc_embed(features)
        return z_embed


class PhysicsDecoder(nn.Module):
    """
    Decodes physical parameters D and rho from the explicit patient latent vector z.
    """
    def __init__(self, embedding_dim=64):
        super().__init__()
        # A small decoder network
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU()
        )
        self.fc_D = nn.Linear(16, 1)
        self.fc_rho = nn.Linear(16, 1)
        
        # Initialize biases to standard initial physical values: D0 ~ 0.01 (log(-4.6)), rho0 ~ 0.3 (log(-1.2))
        self.fc_D.bias.data.fill_(-4.6)
        self.fc_rho.bias.data.fill_(-1.2)
        
    def forward(self, z):
        feat = self.net(z)
        D_val = torch.exp(self.fc_D(feat))
        rho_val = torch.exp(self.fc_rho(feat))
        return D_val, rho_val


class FiLMLayer(nn.Module):
    """
    Feature Linear Modulation (FiLM) Layer.
    Applies patient-conditioned scale and shift to the layer's pre-activations.
    """
    def __init__(self, in_features, out_features, embedding_dim=64):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Projection from patient embedding to modulation parameters
        self.film_gamma = nn.Linear(embedding_dim, out_features)
        self.film_beta = nn.Linear(embedding_dim, out_features)
        
        # Initialize weights for identity mapping at startup
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias) # Scale starts at 1
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)   # Shift starts at 0
        
        self.activation = nn.Tanh()
        
    def forward(self, x, z_embed):
        # x: (N, in_features), z_embed: (N, embedding_dim)
        pre_act = self.linear(x)
        
        gamma = self.film_gamma(z_embed)
        beta = self.film_beta(z_embed)
        
        return self.activation(gamma * pre_act + beta)


class CoordinateMLP(nn.Module):
    """
    Coordinate-based MLP mapping spatio-temporal inputs (x, y, z, t) -> density c.
    Each hidden layer is modulated via FiLM based on the patient's latent embedding.
    """
    def __init__(self, embedding_dim=64, hidden_dim=64, num_layers=4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        # Input layer: 4 coordinates (x, y, z, t)
        self.layers.append(FiLMLayer(4, hidden_dim, embedding_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(FiLMLayer(hidden_dim, hidden_dim, embedding_dim))
            
        # Output layer maps to 1 density value in [0, 1]
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, coords, z_embed):
        # coords: (N, 4), z_embed: (N, embedding_dim)
        h = coords
        for layer in self.layers:
            h = layer(h, z_embed)
        
        out = self.out_layer(h)
        return self.sigmoid(out)


class MarginSenseNet(nn.Module):
    """
    Integrated MarginSense Network combining the 3D CNN encoder, physics decoder, and the FiLM coordinate MLP.
    """
    def __init__(self, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.encoder = Encoder3D(in_channels=5, embedding_dim=embedding_dim)
        self.physics_decoder = PhysicsDecoder(embedding_dim=embedding_dim)
        self.coordinate_mlp = CoordinateMLP(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        
    def forward_encoder(self, volume, covariates=None):
        """Runs the 3D CNN to extract patient latent representation z."""
        z = self.encoder(volume, covariates)
        D, rho = self.physics_decoder(z)
        return z, D, rho
        
    def forward_coordinate(self, coords, z_embed):
        """Evaluates the cell density at specific coordinates conditioned on the patient embedding."""
        return self.coordinate_mlp(coords, z_embed)
        
    def forward(self, volume, coords, covariates=None):
        """Combined forward pass: extracts embedding z, decodes physics parameters, and evaluates density."""
        z, D, rho = self.forward_encoder(volume, covariates)
        
        # Expand embedding to match the number of coordinates
        z_expanded = z.expand(coords.size(0), -1)
        
        density = self.forward_coordinate(coords, z_expanded)
        return density, D, rho, z


def save_patient_latent(patient_id, z):
    """
    Saves the 64-dimensional patient latent vector z to outputs/patient_latents.json.
    z can be a PyTorch tensor, numpy array, or a list of floats.
    """
    import os
    import json
    import torch
    
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    if hasattr(z, "tolist"):
        z = z.tolist()
    if isinstance(z, list) and len(z) > 0 and isinstance(z[0], list):
        # If it's a batch, take the first element
        z = z[0]
        
    os.makedirs("outputs", exist_ok=True)
    latents_path = "outputs/patient_latents.json"
    
    latents = {}
    if os.path.exists(latents_path):
        try:
            with open(latents_path, "r") as f:
                latents = json.load(f)
        except Exception:
            pass
            
    latents[patient_id] = z
    
    try:
        with open(latents_path, "w") as f:
            json.dump(latents, f, indent=4)
        print(f"[+] Saved latent vector for patient '{patient_id}' to {latents_path}")
    except Exception as e:
        print(f"[Warning] Failed to save latent vector for patient '{patient_id}': {e}")


def load_covariate_vector(patient_id, npz_data=None):
    """Loads patient covariates JSON if present, otherwise uses literature defaults,
    computes/loads 5 derived imaging features, and returns a standardized 11-element numeric list.
    """
    import os
    import json
    import numpy as np
    
    cov_path = f"data/processed/{patient_id}_covariates.json"
    cov = None
    if os.path.exists(cov_path):
        try:
            with open(cov_path, "r") as f:
                cov = json.load(f)
        except Exception:
            pass
            
    if cov is None:
        # Generate default covariates
        cov = {
            "age": 58,
            "kps": 80,
            "idh_status": "Wild-type",
            "mgmt_status": "Unmethylated",
            "resection_extent": "GTR",
            "laterality": "Left"
        }
        
    try:
        age = float(cov.get("age", 58))
    except Exception:
        age = 58.0
        
    try:
        kps = float(cov.get("kps", 80))
    except Exception:
        kps = 80.0
        
    idh = 1.0 if cov.get("idh_status") == "Mutant" else 0.0
    mgmt = 1.0 if cov.get("mgmt_status") == "Methylated" else 0.0
    
    resection_val = cov.get("resection_extent", "GTR")
    if resection_val in ["GTR", "Gross Total", "Gross Total Resection (GTR)"]:
        resection = 1.0
    elif resection_val in ["STR", "Subtotal", "Subtotal Resection (STR)"]:
        resection = 0.5
    elif resection_val in ["Biopsy", "Biopsy Only"]:
        resection = 0.25
    else:
        resection = 0.0
        
    laterality = 0.0 if cov.get("laterality") == "Left" else (0.5 if cov.get("laterality") == "Right" else 1.0)
    
    # --- Derived imaging features ---
    derived_path = f"data/processed/{patient_id}_derived_features.json"
    derived = None
    if os.path.exists(derived_path):
        try:
            with open(derived_path, "r") as f:
                derived = json.load(f)
        except Exception:
            pass
            
    if derived is None and npz_data is not None:
        from src.compute_features import compute_all_imaging_features
        try:
            label = npz_data.get('label')
            spacing = npz_data.get('spacing', np.array([1.0, 1.0, 1.0]))
            tissue_map = npz_data.get('tissue_map')
            derived = compute_all_imaging_features(label, spacing, tissue_map)
        except Exception:
            pass
            
    if derived is None:
        npz_path = f"data/processed/{patient_id}.npz"
        if os.path.exists(npz_path):
            try:
                npz_file = np.load(npz_path)
                from src.compute_features import compute_all_imaging_features
                label = npz_file['label']
                spacing = npz_file['spacing']
                tissue_map = npz_file['tissue_map']
                derived = compute_all_imaging_features(label, spacing, tissue_map)
            except Exception:
                pass
                
    if derived is None:
        # Defaults if no data/processed file exists yet (e.g. at start of preprocessing)
        derived = {
            "tumor_volume_cm3": 10.0,
            "hemisphere": "Left",
            "tumor_location": "Temporal",
            "ventricle_dist_mm": 15.0,
            "sphericity": 0.75
        }
        
    # Standardize values to range [0, 1] approximately:
    # 1. Age (divided by 100)
    age_norm = age / 100.0
    # 2. KPS (divided by 100)
    kps_norm = kps / 100.0
    # 3. Tumor volume (log scale, divided by 5)
    vol = float(derived.get("tumor_volume_cm3", 10.0))
    vol_norm = float(np.log1p(vol) / 5.0)
    # 4. Hemisphere
    hemi_str = derived.get("hemisphere", "Left")
    hemi = 0.0 if hemi_str == "Left" else (0.5 if hemi_str == "Right" else 1.0)
    # 5. Tumor location (lobar region)
    loc_str = derived.get("tumor_location", "Temporal")
    loc_map = {"Frontal": 0.0, "Parietal": 0.25, "Temporal": 0.5, "Occipital": 0.75, "Insular": 1.0}
    loc = loc_map.get(loc_str, 0.5)
    # 6. Distance from ventricles (clamped at 50mm, divided by 50)
    dist = float(derived.get("ventricle_dist_mm", 15.0))
    dist_norm = float(np.clip(dist, 0.0, 50.0) / 50.0)
    # 7. Sphericity [0, 1]
    sph = float(derived.get("sphericity", 0.75))
    sph_norm = float(np.clip(sph, 0.0, 1.0))
    
    # 11 features total: [age_norm, kps_norm, idh, mgmt, resection, laterality, vol_norm, hemi, loc, dist_norm, sph_norm]
    # RETRAIN_REQUIRED: The dimension of this returned vector must match the neural network input size (11)
    return [age_norm, kps_norm, idh, mgmt, resection, laterality, vol_norm, hemi, loc, dist_norm, sph_norm]
