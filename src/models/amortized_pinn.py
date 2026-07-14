import torch
import torch.nn as nn

class Encoder3D(nn.Module):
    """
    3D CNN Encoder that ingests the patient's 5-channel 3D volume (4 MRI modalities + 1 segmentation)
    and outputs a patient embedding and global physical growth parameters (D, rho).
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
        
        # Physical parameters projection (in log-space for positivity enforcement)
        self.fc_log_D = nn.Linear(64, 1)
        self.fc_log_rho = nn.Linear(64, 1)
        
        # Initialize biases to standard initial physical values: D0 ~ 0.01 (log(-4.6)), rho0 ~ 0.3 (log(-1.2))
        self.fc_log_D.bias.data.fill_(-4.6)
        self.fc_log_rho.bias.data.fill_(-1.2)
        
    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        
        z_embed = self.fc_embed(features)
        
        # Enforce positive physics parameters via exponential mapping
        D_val = torch.exp(self.fc_log_D(features))
        rho_val = torch.exp(self.fc_log_rho(features))
        
        return z_embed, D_val, rho_val


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
    Integrated MarginSense Network combining the 3D CNN encoder and the FiLM coordinate MLP.
    """
    def __init__(self, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.encoder = Encoder3D(in_channels=5, embedding_dim=embedding_dim)
        self.coordinate_mlp = CoordinateMLP(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        
    def forward_encoder(self, volume):
        """Runs the 3D CNN to extract patient features and physical variables."""
        return self.encoder(volume)
        
    def forward_coordinate(self, coords, z_embed):
        """Evaluates the cell density at specific coordinates conditioned on the patient embedding."""
        return self.coordinate_mlp(coords, z_embed)
        
    def forward(self, volume, coords):
        """Combined forward pass: extracts embedding and evaluates at coords."""
        z_embed, D, rho = self.forward_encoder(volume)
        
        # Expand embedding to match the number of coordinates
        z_expanded = z_embed.expand(coords.size(0), -1)
        
        density = self.forward_coordinate(coords, z_expanded)
        return density, D, rho
