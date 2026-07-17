# MarginSense: Technical Architecture and Model Specification

This document provides a rigorous mathematical and structural breakdown of the MarginSense system—a physics-informed, uncertainty-aware, patient-specific radiotherapy margin optimization framework for Glioblastoma (GBM).

---

## 1. High-Level System Architecture

MarginSense shifts the paradigm of radiotherapy planning from standard population-averaged, uniform safety margins (1.5 cm) to individualized, biophysically-grounded target volumes. The pipeline integrates:
1. **Multi-modal MRI Processing**: Fuses 4 structural MRI modalities ($T_1$, $T_{1ce}$, $T_2$, FLAIR) and GTV (Gross Tumor Volume) segmentation.
2. **Clinical Covariate Integration**: Incorporates clinical factors (Age, KPS, IDH status, MGMT methylation, resection extent, tumor sphericity, etc.) to condition the spatial spread.
3. **Amortized (Meta-Learned) Physics-Informed Neural Network (PINN)**: Infers patient-specific cell density maps in a single feedforward pass (no per-patient retraining required).
4. **Epistemic Uncertainty Quantification (UQ)**: Utilizes a 5-model deep ensemble to output a per-voxel confidence field.
5. **Margin Optimization (UCB & Spared OARs)**: Sweeps coverage vs. healthy-tissue toxicity curves, applies Upper Confidence Bound (UCB) safety contours, and explicitly excludes Cerebrospinal Fluid (CSF) to protect organs at risk.

```mermaid
graph TD
    A[MRI Modalities + GTV Mask] --> B(3D CNN Encoder)
    C[Clinical Covariates] --> B
    B -->|Patient Embedding z| D(Physics Decoder)
    B -->|Patient Embedding z| E(Coordinate-Based MLP)
    D -->|Estimated global D_0 & rho_0| F(Physics-Informed Loss)
    G[Spatio-Temporal Coords x, y, z, t] --> E
    E -->|Predicted Cell Density c| F
    F -->|Fisher-KPP PDE Residual| H[Backpropagation]
    
    subgraph Ensemble Inference (5 Seeded Models)
        E1[Model 1] & E2[Model 2] & E3[Model 3] & E4[Model 4] & E5[Model 5] -->|Feedforward| I[Per-Voxel Mean & Std Dev]
    end
    
    I -->|mean + z * std| J[Upper Confidence Bound Contour]
    J -->|Exclude CSF / Ventricles| K[Optimized Safety Margin]
```

---

## 2. Neural Network Components

The core model, defined in [amortized_pinn.py](file:///d:/Lakshya/Hackathons/Internal-Hackathon/Demos/MarginSense/src/models/amortized_pinn.py), consists of three coupled sub-networks: `Encoder3D`, `PhysicsDecoder`, and `CoordinateMLP`, coordinated via `MarginSenseNet`.

### 2.1 3D CNN Encoder (`Encoder3D`)
Ingests a $128 \times 128 \times 128$ 5-channel voxel grid (4 MRI channels + 1 GTV mask) and extracts a patient-specific spatial signature:
- **Convolutional Trunk**: 4 blocks of 3D convolutions with stride 2 and padding 1:
  - $\text{Conv3D}(5 \to 16) \to \text{BatchNorm3D} \to \text{ReLU}$
  - $\text{Conv3D}(16 \to 32) \to \text{BatchNorm3D} \to \text{ReLU}$
  - $\text{Conv3D}(32 \to 64) \to \text{BatchNorm3D} \to \text{ReLU}$
  - $\text{Conv3D}(64 \to 64) \to \text{BatchNorm3D} \to \text{ReLU}$
- **Pooling**: 3D Adaptive Average Pooling collapses spatial dimensions to $1 \times 1 \times 1$, producing a 64-dimensional feature vector.
- **Covariate Injection**: Ingests an 11-dimensional standardized covariate vector (Age, KPS, IDH, MGMT, Resection, Laterality, Normalized Volume, Hemisphere, Lobar Location, Ventricle Distance, Sphericity). A linear layer projects these covariates into 64 dimensions, which is directly added to the pooled CNN features:
  \[
  \mathbf{f}_{patient} = \mathbf{f}_{CNN} + \mathbf{W}_{cov} \mathbf{x}_{cov} + \mathbf{b}_{cov}
  \]
  where $\mathbf{W}_{cov}$ is initialized to zero to allow the model to learn a stable image-only baseline before utilizing covariates.
- **Latent Projection**: Outputs a 64-dimensional latent embedding $\mathbf{z}_{embed}$.

### 2.2 Physics Decoder (`PhysicsDecoder`)
Maps the patient embedding $\mathbf{z}_{embed}$ to global scalar biophysical parameters: diffusion rate $D_0$ and proliferation rate $\rho_0$.
- **Layers**: $\text{Linear}(64 \to 16) \to \text{ReLU} \to \text{Linear}(16 \to 1 \text{ for } D_0, 1 \text{ for } \rho_0)$.
- **Positivity Enforcement**: To ensure parameter physical validity ($D_0, \rho_0 > 0$), outputs are projected via an exponential layer:
  \[
  D_0 = \exp(f_{D}(\mathbf{z})) \quad \text{and} \quad \rho_0 = \exp(f_{\rho}(\mathbf{z}))
  \]
- **Initialization**: Biases are filled at $b_{D} = -4.6$ and $b_{\rho} = -1.2$ so that at initialization, parameters are anchored at literature-typical baseline values ($D_0 \approx 0.01\text{ mm}^2/\text{day}$, $\rho_0 \approx 0.3/\text{day}$).

### 2.3 Feature-wise Linear Modulation (`FiLMLayer`)
To achieve amortized generalization across patient profiles without per-patient neural network training, MarginSense conditions the coordinate evaluation network using FiLM:
- For a given layer with input $\mathbf{x} \in \mathbb{R}^{d_{in}}$:
  \[
  \mathbf{y} = \text{Tanh}(\gamma(\mathbf{z}_{embed}) \odot (\mathbf{W}\mathbf{x} + \mathbf{b}) + \beta(\mathbf{z}_{embed}))
  \]
  where:
  - $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$ and $\mathbf{b} \in \mathbb{R}^{d_{out}}$ are standard weights and biases.
  - $\gamma(\mathbf{z}_{embed}) \in \mathbb{R}^{d_{out}}$ represents the scaling vector, projected from $\mathbf{z}_{embed}$.
  - $\beta(\mathbf{z}_{embed}) \in \mathbb{R}^{d_{out}}$ represents the shift vector, projected from $\mathbf{z}_{embed}$.
  - $\odot$ denotes the Hadamard (element-wise) product.
- **Weight Initialization**: Initialized at identity ($\gamma$ weights $= 0$, biases $= 1$; $\beta$ weights $= 0$, biases $= 0$) so the MLP behaves as a patient-agnostic coordinate network at training start.

### 2.4 Coordinate MLP (`CoordinateMLP`)
A coordinate-based multi-layer perceptron mapping spatial coordinates and time $(x,y,z,t) \in [0,1]^4$ to predicted tumor cell density $c \in [0, 1]$.
- **Structure**: 4 hidden layers (64 units each), with each layer structured as a `FiLMLayer` conditioned on $\mathbf{z}_{embed}$.
- **Output**: Sigmoid activation ensures $c \in [0, 1]$.

---

## 3. Governing Biophysical Equations & Loss Formulation

MarginSense is trained using a multi-objective loss function combining data observation with partial differential equation (PDE) regularization, detailed in [train.py](file:///d:/Lakshya/Hackathons/Internal-Hackathon/Demos/MarginSense/src/training/train.py).

### 3.1 Governing Equation (Fisher-KPP)
Glioma spread is modeled using the Fisher-Kolmogorov-Petrovsky-Piscounov (Fisher-KPP) reaction-diffusion model:
\[
\frac{\partial c}{\partial t} = \nabla \cdot \left( \mathbf{D}(\mathbf{x}) \nabla c \right) + \rho_0 c (1 - c)
\]
where $c(\mathbf{x}, t)$ is the cell density, $\mathbf{D}(\mathbf{x})$ is the spatially-heterogeneous diffusion tensor, and $\rho_0$ is the proliferation rate.

### 3.2 Spatially-Varying Diffusion Tensor
Diffusion speeds depend on tissue microanatomy. We define the tissue-weighted diffusion coefficient $D(\mathbf{x})$:
\[
D(\mathbf{x}) = D_0 \cdot w_{tissue}(\mathbf{x})
\]
where $w_{tissue}(\mathbf{x})$ acts as a diffusion multiplier mapped from anatomical segmentation maps (Z-score thresholds of structural MRI):
- **White Matter (WM)** ($w_{tissue} = 0.15$): Fast propagation along structural fibers.
- **Gray Matter (GM)** ($w_{tissue} = 0.03$): Slow propagation.
- **Necrotic Core / Edema** ($w_{tissue} = 0.075$): Intermediate propagation.
- **Cerebrospinal Fluid (CSF)** ($w_{tissue} = 0.0$): Zero diffusion (physical boundary).

In the default implementation, the diffusion tensor is isotropic:
\[
\mathbf{D}(\mathbf{x}) = D(\mathbf{x})\mathbf{I}_{3\times3}
\]
*Note: The code features a built-in extension point for anisotropic tensors derived from DTI (Diffusion Tensor Imaging) fractional anisotropy maps.*

### 3.3 Physics Loss via Automated Differentiation
To compute the divergence of the flux, $\nabla \cdot \mathbf{j}$ where $\mathbf{j} = \mathbf{D}(\mathbf{x}) \nabla c$, a double-pass PyTorch Autograd scheme is implemented:
1. **First-Order Gradients**:
   \[
   \nabla c = \left( \frac{\partial c}{\partial x}, \frac{\partial c}{\partial y}, \frac{\partial c}{\partial z} \right) \quad \text{and} \quad \frac{\partial c}{\partial t}
   \]
2. **Flux Calculation**:
   \[
   \mathbf{j} = D(\mathbf{x}) \nabla c = \left( j_x, j_y, j_z \right)
   \]
3. **Second-Order Divergence**:
   \[
   \nabla \cdot \mathbf{j} = \frac{\partial j_x}{\partial x} + \frac{\partial j_y}{\partial y} + \frac{\partial j_z}{\partial z}
   \]
4. **PDE Residual**:
   \[
   \mathcal{R}_{pde}(\mathbf{x}, t) = \frac{\partial c}{\partial t} - \nabla \cdot \mathbf{j} - \rho_0 c (1 - c)
   \]

### 3.4 Multi-Objective Training Loss
The network is optimized over a training set of size $N$ using a composite loss function:
\[
\mathcal{L} = \lambda_{data} \mathcal{L}_{data} + \lambda_{ic} \mathcal{L}_{ic} + \lambda_{pde} \mathcal{L}_{pde}
\]
1. **Data Loss ($\mathcal{L}_{data}$)**: MSE at $t=1$ evaluating predicted cell density against the clinical recurrence mask (using stratified sampling: 50% recurrence region, 50% background):
   \[
   \mathcal{L}_{data} = \frac{1}{N_{data}} \sum_{i=1}^{N_{data}} (c(\mathbf{x}_i, 1) - y_i)^2
   \]
2. **Initial Condition Loss ($\mathcal{L}_{ic}$)**: Enforces a localized Gaussian distribution of density centered at the visible tumor centroid $\mathbf{x}_{centroid}$ at $t=0$:
   \[
   \mathcal{L}_{ic} = \frac{1}{N_{ic}} \sum_{j=1}^{N_{ic}} (c(\mathbf{x}_j, 0) - c_0(\mathbf{x}_j))^2 \quad \text{where} \quad c_0(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_{centroid}\|^2}{2\sigma^2}\right)
   \]
3. **PDE Loss ($\mathcal{L}_{pde}$)**: Evaluated at random spatio-temporal collocation points $\mathbf{x}_k, t_k \in [0, 1]^4$ to regularize predictions to behave physically:
   \[
   \mathcal{L}_{pde} = \frac{1}{N_{col}} \sum_{k=1}^{N_{col}} (\mathcal{R}_{pde}(\mathbf{x}_k, t_k))^2
   \]
- **Loss Weights**: Coordinated at $\lambda_{data} = 1.0$, $\lambda_{ic} = 1.0$, and $\lambda_{pde} = 0.1$.

---

## 4. Epistemic Uncertainty & Safety Contouring

Standard PINN models produce point predictions, masking regions where predictions are ungrounded due to data sparsity. MarginSense implements a Deep Ensemble to quantify epistemic uncertainty and define safe treatment margins, detailed in [train_ensemble.py](file:///d:/Lakshya/Hackathons/Internal-Hackathon/Demos/MarginSense/src/training/train_ensemble.py) and [evaluate_uncertainty.py](file:///d:/Lakshya/Hackathons/Internal-Hackathon/Demos/MarginSense/src/evaluate_uncertainty.py).

### 4.1 Ensemble Architecture
- **Configuration**: $M = 5$ identical `MarginSenseNet` networks trained with distinct random seeds (Seeds: 42, 43, 44, 45, 46).
- **Voxel-wise Statistics**: For any coord $\mathbf{x}$ at $t=1$:
  - **Ensemble Mean** ($\mu(\mathbf{x})$): Represents the expected infiltration probability.
    \[
    \mu(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} c_m(\mathbf{x}, 1)
    \]
  - **Ensemble Standard Deviation** ($\sigma(\mathbf{x})$): Represents the epistemic model uncertainty.
    \[
    \sigma(\mathbf{x}) = \sqrt{ \frac{1}{M} \sum_{m=1}^{M} \left(c_m(\mathbf{x}, 1) - \mu(\mathbf{x})\right)^2 }
    \]

### 4.2 Upper Confidence Bound (UCB) Safety Margin
To expand treatment boundaries in areas where the model is highly uncertain, MarginSense calculates an adjusted probability field using a UCB formulation:
\[
c_{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + z \cdot \sigma(\mathbf{x})
\]
where $z$ is the clinical safety caution multiplier (adjustable via the user interface slider, default $z = 1.0$).
- At $z = 0$, the margin is formed using the pure mean prediction.
- At $z > 0$, the treatment margin expands dynamically into regions where the ensemble models disagree, ensuring a robust safety buffer.

---

## 5. Margin Optimization Module

Instead of utilizing an arbitrary threshold, MarginSense optimizes the target contour by balancing tumor recurrence coverage against the volume of irradiated healthy brain tissue (implemented in [server.py](file:///d:/Lakshya/Hackathons/Internal-Hackathon/Demos/MarginSense/src/viz/server.py#L988-L1097)):

### 5.1 Coverage vs. Toxicity Sweep
1. Sweeps threshold values $\tau \in [0, 1]$ in $0.02$ increments.
2. For each $\tau$, defines a candidate binary contour:
   \[
   \Omega_{treat}(\tau) = \{ \mathbf{x} \mid c_{UCB}(\mathbf{x}) \ge \tau \text{ and } \text{Tissue}(\mathbf{x}) \ne \text{CSF} \}
   \]
3. Computes:
   - **Recurrence Coverage**:
     \[
     \text{Coverage}(\tau) = \frac{|\Omega_{treat}(\tau) \cap \Omega_{recurrence}|}{|\Omega_{recurrence}|}
     \]
   - **Treated Healthy Volume ($V_{healthy}$)**:
     \[
     V_{healthy}(\tau) = |\Omega_{treat}(\tau) \cap \Omega_{healthy\_tissue}| \times \text{Voxel Volume (cm}^3\text{)}
     \]

### 5.2 Optimal Contour Selection
The optimal threshold $\tau^*$ is chosen by maximizing the clinical payoff objective function:
\[
\tau^* = \arg\max_{\tau} \left( \text{Coverage}(\tau) - \lambda \cdot \frac{V_{healthy}(\tau)}{\text{Normalizing Constant}} \right)
\]
where $\lambda$ represents the clinical priority weight (adjustable via the UI slider, defaulted to calibrate MarginSense toxicity with standard-margin baseline volumes).

---

## 6. Factual Explainability Engine

To ensure clinical accountability, MarginSense extracts deterministic, template-filled spatial descriptions instead of using probabilistic generative LLMs (see [server.py](file:///d:/Lakshya/Hackathons/Internal-Hackathon/Demos/MarginSense/src/viz/server.py#L1100-L1114+)):

1. **Anatomical Sector Decomposition**: Computes patient vectors centered at the tumor centroid and divides the surrounding region into 6 half-space directional sectors: Superior, Inferior, Anterior, Posterior, Medial, and Lateral.
2. **Localized Metrics Extraction**: For each sector, computes:
   - **Infiltration Extension**: The $95^{\text{th}}$-percentile distance that predicted density ($c > 0.20$) extends past the visible GTV boundary.
   - **Anatomical Substrate**: The local white matter fraction compared to the global average.
   - **Biophysical Properties**: Local mean diffusion coefficient $D(\mathbf{x})$ based on tissue maps.
3. **Tumor Elongation via PCA**: Conducts Principal Component Analysis (PCA) on the visible tumor mask coordinates to determine spatial orientation, alignment scores, and sphericity ($\lambda_{min} / \lambda_{max}$).
4. **Deterministic Synthesis**: Merges metrics into a pre-defined f-string template referencing clinical publications, providing a deterministic explanation of *why* the tumor is predicted to infiltrate along specific paths.

---

## 7. Model Verification & Performance Evaluation Metrics

To compare MarginSenseNet with standard uniform margins and vanilla non-amortized PINN baselines, the evaluation engine ([compare_models.py](file:///d:/Lakshya/Hackathons/Internal-Hackathon/Demos/MarginSense/src/compare_models.py)) reports the following benchmarks:

- **Recurrence Coverage (%)**: Percentage of recurrence voxels contained inside the treated region.
- **Treated Healthy Tissue ($\text{cm}^3$)**: Cumulative volume of healthy brain tissue included in the treatment contour.
- **95th-Percentile Hausdorff Distance (HD95, mm)**: Standard BraTS metric assessing maximum boundary-to-boundary distance while eliminating outliers.
- **Surface Dice @ 2mm**: Percentage of the treatment surface that lies within 2mm of the ground-truth recurrence boundary.
- **Average Symmetric Surface Distance (ASD, mm)**: Average distance from the predicted margin boundary to the recurrence boundary.
- **Physics Residual MSE**: Verification of how closely the coordinate network conforms to the Fisher-KPP PDE (residual error calculated via finite differences).
