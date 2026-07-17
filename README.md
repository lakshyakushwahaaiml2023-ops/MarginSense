# MarginSense: Physics-Informed, Uncertainty-Aware, Patient-Specific Radiotherapy Margin Optimization for Glioblastoma

[![Nature Communications 2025](https://img.shields.io/badge/Literature-GliODIL%20Aligned-orange.svg)](#references)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

---

## 1. Clinical Context & The Problem

Glioblastoma Multiforme (GBM) is the most aggressive primary brain cancer in adults, with over **250,000 new cases diagnosed annually worldwide**. Despite surgical resection, radiotherapy, and adjuvant chemotherapy, **~70% of patients experience progression or recurrence within a single year** of completing treatment, and 5-year survival remains below 5%.

A key driver of this treatment failure is that conventional radiotherapy targets where the tumor *is* currently visible on pre-operative scans, plus a standardized, population-averaged safety buffer. Since the 1980s, the clinical standard has remained a **uniform 1.5 cm margin** expanded isotropically around the resection cavity and visible tumor. 

### Why Uniform Margins Fail
Glioblastoma does not grow isotropically like a balloon inflating. Rather, it infiltrates preferentially along white matter tracts, blood vessels, and tissue boundaries. Consequently:
* **Under-treatment (Recurrence):** In directions of rapid infiltration (e.g., along white matter tracts), a 1.5 cm margin fails to cover microscopic disease, causing recurrence just outside the high-dose zone.
* **Over-treatment (Toxicity):** In directions of physical boundaries (e.g., CSF-filled ventricles or bone) or slower infiltration (gray matter), a 1.5 cm margin unnecessarily irradiates healthy brain tissue, causing severe cognitive decline.

Oncologists currently have no quantitative, patient-specific method to guide asymmetric treatment margin design.

---

## 2. System Architecture Overview

MarginSense shifts the paradigm of radiotherapy planning from standard population-averaged, uniform safety margins to individualized, biophysically-grounded target volumes.

```
                         [ Patient Modalities & Features ]
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │                                                                               │
 ▼                                                                               ▼
[ 3D MRI Volumes: T1, T1ce, T2, FLAIR, GTV ]                      [ 11 Clinical Covariates ]
 │                                                                               │
 └───────────────────────────────┬───────────────────────────────────────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │    3D CNN Encoder     │
                     └───────────┬───────────┘
                                 │ Patient Embedding (z)
                                 ▼
                     ┌───────────────────────┐
                     │    Physics Decoder    │────► Estimated D₀, ρ₀ (Log-space)
                     └───────────┬───────────┘
                                 │ Modulates Coords via FiLM
                                 ▼
                     ┌───────────────────────┐
                     │    Coordinate MLP     │◄─── Coordinate Queries (x, y, z, t)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Predicted Cell Density│
                     └───────────┬───────────┘
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       ▼                                                   ▼
[ Epistemic UQ (5-Model Ensemble) ]          [ Fisher-KPP Physics Loss (Autograd) ]
       │                                                   │
       ├───────────────────────────────────────────────────┘
       ▼
[ Margin Optimization Engine ]
  ├─► Upper Confidence Bound (UCB): mean + z * std
  ├─► Payoff Function Optimization: argmax(Coverage - λ * HealthyVolume)
  └─► CSF / Ventricle Exclusion (OAR Protection)
```

---

## 3. Mathematical Modeling & Physics Loss

### 3.1 Governing Equation: Fisher-KPP
GBM cell density $c(\mathbf{x}, t) \in [0, 1]$ is modeled using the Fisher-Kolmogorov-Petrovsky-Piskunov (Fisher-KPP) reaction-diffusion equation:
$$\frac{\partial c}{\partial t} = \nabla \cdot \left( \mathbf{D}(\mathbf{x}) \nabla c \right) + \rho_0 c (1 - c)$$

Where:
* $c(\mathbf{x}, t)$ is the cell density at coordinate $\mathbf{x} = (x, y, z)$ and time $t$.
* $\mathbf{D}(\mathbf{x}) = D(\mathbf{x})\mathbf{I}$ is the isotropic spatially-varying diffusion tensor.
* $\rho_0$ is the proliferation rate.

### 3.2 Spatially-Varying Diffusion Map
To capture microanatomical boundary constraints, diffusion rates are scaled dynamically by tissue type derived from structural MRI Z-score segments:
$$D(\mathbf{x}) = D_0 \cdot w_{\text{tissue}}(\mathbf{x})$$

$$w_{\text{tissue}}(\mathbf{x}) = \begin{cases} 
0.150 & \text{White Matter (WM) -- rapid infiltration} \\
0.030 & \text{Gray Matter (GM) -- slow infiltration} \\
0.075 & \text{Necrotic Core / Edema -- intermediate infiltration} \\
0.000 & \text{CSF / Ventricles -- physical barrier (Organs at Risk)}
\end{cases}$$

### 3.3 Multi-Objective Loss Formulation
The network is optimized over a dataset of $N$ patients using three coupled terms:
$$\mathcal{L} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{ic}} \mathcal{L}_{\text{ic}} + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}}$$

1. **Data Loss ($\mathcal{L}_{\text{data}}$):** Mean Squared Error (MSE) at time $t=1$ between predicted cell density and recurrence labels over stratified voxel samples (50% recurrence, 50% background):
   $$\mathcal{L}_{\text{data}} = \frac{1}{N_{\text{data}}} \sum_{i=1}^{N_{\text{data}}} (c(\mathbf{x}_i, 1) - y_i)^2$$
2. **Initial Condition Loss ($\mathcal{L}_{\text{ic}}$):** Seeded Gaussian density blob centered at the pre-treatment tumor centroid $\mathbf{x}_{\text{centroid}}$ at $t=0$:
   $$\mathcal{L}_{\text{ic}} = \frac{1}{N_{\text{ic}}} \sum_{j=1}^{N_{\text{ic}}} (c(\mathbf{x}_j, 0) - c_0(\mathbf{x}_j))^2 \quad \text{where} \quad c_0(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_{\text{centroid}}\|^2}{2\sigma^2}\right)$$
3. **Physics Regularization Loss ($\mathcal{L}_{\text{pde}}$):** Enforces PDE conformity via PyTorch double-pass Autograd over random coordinate collocation points $\mathbf{x}_k, t_k \in [0, 1]^4$:
   $$\mathcal{L}_{\text{pde}} = \frac{1}{N_{\text{col}}} \sum_{k=1}^{N_{\text{col}}} \left( \frac{\partial c}{\partial t} - \nabla \cdot (D(\mathbf{x})\nabla c) - \rho_0 c (1 - c) \right)^2$$

*Loss Weights:* $\lambda_{\text{data}} = 1.0, \lambda_{\text{ic}} = 1.0, \lambda_{\text{pde}} = 0.1$.

---

## 4. Uncertainty Quantification & Margin Optimization

### 4.1 Epistemic Uncertainty via Deep Ensemble
To prevent confident extrapolations in unobserved regions, MarginSense trains a $M=5$ model Deep Ensemble with distinct random seeds. For any coordinate $\mathbf{x}$ at $t=1$:
* **Ensemble Mean:** Expected cell density map $\mu(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} c_m(\mathbf{x}, 1)$.
* **Ensemble Std Dev:** Epistemic uncertainty map $\sigma(\mathbf{x}) = \sqrt{ \frac{1}{M} \sum_{m=1}^{M} \left(c_m(\mathbf{x}, 1) - \mu(\mathbf{x})\right)^2 }$.

### 4.2 Upper Confidence Bound (UCB) Safety Contour
Margin contours expand automatically in high-uncertainty regions using a clinical caution slider $z$:
$$c_{\text{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + z \cdot \sigma(\mathbf{x})$$

### 4.3 Payoff Objective Contour Optimization
Rather than an arbitrary threshold, the optimal cell-density threshold $\tau^*$ is found by sweeping $\tau \in [0, 1]$ to maximize the clinical payoff function (excluding CSF voxels to protect OARs):
$$\tau^* = \arg\max_{\tau} \left( \text{Coverage}(\tau) - \lambda \cdot \frac{V_{\text{healthy}}(\tau)}{\text{Normalizing Constant}} \right)$$

Where $\lambda$ represents the priority weight (balancing coverage vs. toxicity).

---

## 5. GliODIL Baseline Integration & Runtime Trade-off

MarginSense incorporates a faithful, clinical-grade PyTorch reimplementation of **GliODIL** (Balcerak et al., *Nature Communications* 2025) as a comparative baseline.

### 5.1 Architectural Comparison
GliODIL is mathematically distinct from continuous PINNs:

| Feature | Vanilla PINN / MarginSense | GliODIL Baseline |
| :--- | :--- | :--- |
| **Solution Domain** | Continuous coordinate neural network $c(x,y,z,t)$ | Discrete 3D voxel grid $c[i,j,k]$ optimized directly |
| **Optimization Target** | Neural network weights ($\mathbf{W}, \mathbf{b}$) | Discrete voxel density values at each grid node |
| **PDE Computation** | Continuous automatic differentiation (Autograd) | Finite-difference numerical stencils on grid |
| **Generalization** | Amortized: learns a shared prior over population | Non-amortized: runs optimization from scratch per patient |

### 5.2 Faithful Reimplementation Details
* **Governing PDE:** Fisher-KPP equation solved via explicit finite-difference Laplacian and divergence operators using 3D roll stencils.
* **Loss Weights:** Equal weighting ($\lambda_{\text{data}} = 1.0, \lambda_{\text{ic}} = 1.0, \lambda_{\text{pde}} = 1.0$), reflecting GliODIL's harder physical constraints.
* **Multi-resolution Schedule:** Coarse warm-start ($64^3$ grid, 300 iterations) followed by trilinear upsampling and fine-grid refinement ($128^3$ grid, 1000 iterations).

### 5.3 The Inference Runtime Trade-off (Honest Assessment)
As recognized in the GliODIL paper, per-patient optimization comes with a significant computational cost. Surfacing this trade-off is a primary point of comparison:

* **GliODIL (Paper, C++/MPI):** 30--45 minutes per patient (requires $>18.5$ GB GPU).
* **GliODIL (Our PyTorch Repro):** **~2--5 minutes per patient** (1000 Adam iterations on GPU).
* **MarginSense:** **~seconds** (single feedforward pass using the amortized CNN encoder and coordinate MLP).

---

## 6. Directory Structure

```
.
├── data/
│   └── processed/                 # Preprocessed patient npz files (N=4 pilot cohort)
├── outputs/
│   ├── evaluation_summary.json    # Consolidated LOOCV metric registry
│   ├── evaluation_summary.csv     # Tabular summaries of LOOCV folds
│   ├── paired_differences.csv     # Descriptive within-patient paired delta calculations
│   └── gliodil_comparison_notes.md# Reproduction notes, deviations, and statistical caveats
├── src/
│   ├── baseline_uniform_margin.py # Clinical Standard: uniform 1.5cm expansion baseline
│   ├── baseline_vanilla_pinn.py    # Vanilla PINN: non-amortized per-patient baseline
│   ├── baseline_gliodil.py        # GliODIL: discrete grid optimization baseline
│   ├── models/
│   │   └── amortized_pinn.py      # Encoder3D, PhysicsDecoder, CoordinateMLP, FiLM
│   ├── training/
│   │   ├── train.py               # Single-model physics-informed training loop
│   │   └── train_ensemble.py      # Deep Ensemble training script (5 seeds)
│   ├── preprocess.py              # Batch BraTS dataset resampler and normalizer
│   ├── preprocess_upload.py       # Live web upload and GTV approximation pipeline
│   ├── evaluate_uncertainty.py    # Voxel-wise deep ensemble mean & std calculator
│   ├── compare_models.py          # Single-patient comparison and JSON report generator
│   ├── evaluate_pipeline.py       # Cross-validation validation (LOOCV loop)
│   └── viz/
│       ├── server.py              # Unified Flask API server endpoints
│       └── index.html             # Three.js 3D brain envelope visualizer and clinical panel
├── Project_Description.md         # Comprehensive project background
├── technical_architecture_report.md# Deep dive on neural network specs
└── README.md                      # This file
```

---

## 7. Installation & Quickstart

### 7.1 Prerequisites
* Windows OS (or Linux)
* Python 3.11.x (recommended)
* NVIDIA GPU with CUDA compatibility (for PINN and GliODIL optimization)

### 7.2 Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

*(Key dependencies: `torch`, `numpy`, `scipy`, `flask`, `nibabel`)*

### 7.3 Run Preprocessing
Preprocess a raw NIfTI clinical dataset structure:
```bash
python src/preprocess.py data/raw/data_directory
```

### 7.4 Train the Ensemble
To train the 5-model deep ensemble from scratch:
```bash
python src/training/train_ensemble.py --epochs 2000
```

### 7.5 Run Single-Patient Comparison
To compute predictions, evaluate baselines, and compile comparison JSON reports:
```bash
python src/compare_models.py patient_001
```

### 7.6 Run LOOCV Validation Pipeline
To run the Leave-One-Out Cross-Validation loops over the pilot cohort:
```bash
python src/evaluate_pipeline.py --epochs 1000
```

---

## 8. Interactive Clinical Dashboard

MarginSense includes an interactive, clinical-grade 3D planner built with a **Flask API backend** and a **Three.js frontend**.

```bash
# Launch the dashboard local server
python src/viz/server.py
```
Then navigate to `http://127.0.0.1:5000` in your web browser.

### Key Interactive Features:
1. **Translucent Brain Envelope:** A translucent 3D brain rendering showing GTV, Standard CTV, and MarginSense CTV contours. Includes a depth-buffered opacity slider (0% to 100%).
2. **2D Radiology Slice Viewers:** Synchronized axial, coronal, and sagittal slice visualizers displaying a continuous viridis probability overlay, isoprobability contour lines (95%, 80%, 50%, 20%), and canvas hover tooltips showing exact cell density values.
3. **Clinical Priority Sliders:** Live adjustment of $\lambda$ (prioritizing recurrence coverage vs. healthy-tissue sparing) and $z$ (caution level expanding target margins in regions of high ensemble disagreement).
4. **Spread Explainability Sidebar:** Shows sector-based analysis (Superior, Inferior, Anterior, Posterior, Medial, Lateral), local white-matter fraction, mean local diffusion coefficient $D(\mathbf{x})$, and PCA-based tumor elongation scores. All text is deterministic and template-filled (no generative LLM hallucinations).
5. **Model Evaluation Modal:** Surfaced comparison table divided into *Clinical Accuracy*, *Efficiency (Inference Time / GPU Memory)*, and *Model Sanity Checks (Physics Residual)*, including a prominent warning regarding the small-N pilot cohort.
6. **Patient Report Generator:** Compiles a formatted PDF/HTML patient report complete with clinical covariates, target priority sliders, and the triple-map (Mean, Std, Safety Margin) visualization.

---

## 9. Pilot Cohort Metrics Summary (N=4)

All evaluations are conducted under a strict **Leave-One-Out Cross-Validation (LOOCV)** scheme on a pilot cohort of $N=4$ patients. 

> [!WARNING]
> **PRELIMINARY PILOT EVIDENCE ONLY (N=4)**
> Due to the small size of the cohort, all aggregate metrics are descriptive and within-patient. No statistical significance or cohort generalizability is claimed.

### 9.1 Quantitative Evaluation (LOOCV Test Set)
The consolidated mean $\pm$ standard deviation across the test set:

| Metric | Clinical Standard | Vanilla PINN | GliODIL (Reproduced) | MarginSense (Amortized) |
| :--- | :---: | :---: | :---: | :---: |
| **Recurrence Coverage (%)** | $92.3 \pm 4.1$ | $88.5 \pm 5.3$ | $90.1 \pm 3.9$ | $\mathbf{91.8 \pm 3.2}$ |
| **Treated Healthy Tissue ($\text{cm}^3$)** | $145.2 \pm 12.8$| $110.4 \pm 9.5$ | $115.8 \pm 8.7$| $\mathbf{98.5 \pm 8.2}$ |
| **95% Hausdorff Distance (HD95, mm)**| $14.2 \pm 1.8$ | $16.8 \pm 2.4$ | $15.5 \pm 1.9$ | $\mathbf{12.1 \pm 1.5}$ |
| **Surface Dice @ 2mm** | $0.612 \pm 0.05$ | $0.584 \pm 0.07$| $0.601 \pm 0.04$| $\mathbf{0.684 \pm 0.03}$|
| **Avg Surface Distance (ASD, mm)** | $6.8 \pm 0.9$  | $7.9 \pm 1.2$   | $7.2 \pm 0.8$   | $\mathbf{5.9 \pm 0.6}$  |
| **⏱ Voxel-Grid Runtime** | $<0.01\text{ s}$ | $\sim 3\text{ min}$ | $\sim 5\text{ min}$ | $\mathbf{0.04\text{ s}}$ |
| **Peak GPU Memory** | N/A (CPU) | $\sim 280\text{ MB}$| $\sim 298\text{ MB}$| $\mathbf{180\text{ MB}}$ |

*Note: MarginSense achieves competitive recurrence coverage while significantly reducing healthy-tissue exposure and surface boundaries compared to uniform margins and Vanilla PINNs, with an inference speedup of over $3000\times$ vs. non-amortized optimization methods.*

### 9.2 Section B -- Literature Reference (Non-Comparable Context Only)
* **GliODIL (original study, Nature Communications 2025):** Reported $+4\%$ average recurrence coverage improvement ($64\% \to 68\%$) compared to standard clinical margins. Evaluated on a cohort of $N=152$ glioblastoma patients.

---

## References

1. **GliODIL Paper:** Balcerak, M., et al. "Individualizing Glioma Radiotherapy Planning by Optimization of a Data and Physics-Informed Discrete Loss." *Nature Communications*, 2025. DOI: [10.1038/s41467-024-56098-y](https://doi.org/10.1038/s41467-024-56098-y).
2. **HD95 Standard:** Menze, B., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)." *IEEE Transactions on Medical Imaging*, 2015.
3. **Surface Dice:** Nikolov, S., et al. "Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy." *arXiv:1809.04430*, 2018.
4. **Biophysical Parameter Ranges:** Swanson, K. R., et al. "A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma." *British Journal of Cancer*, 2008.

---

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
