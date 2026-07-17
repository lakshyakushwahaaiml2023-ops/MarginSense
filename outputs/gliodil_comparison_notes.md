# GliODIL Comparison Notes

> [!WARNING]
> **N=4 — Preliminary, Exploratory Evidence Only**
> All numeric results in this comparison involve 4 patients. No statistical significance
> can be claimed at this sample size. All comparisons are **descriptive** and **within-patient**.
> No p-values are reported. No claim that any method "beats" another in a statistically
> validated sense is warranted.

---

## 1. What GliODIL Is

**GliODIL** (Balcerak et al., *Nature Communications*, 2025, DOI: 10.1038/s41467-024-56098-y)
is the current published state-of-the-art for patient-specific glioma radiotherapy margin planning.

It uses the **ODIL** (Optimizing a Discrete Loss) framework to directly optimize a discrete
3D voxel field representing tumor cell concentration, constrained by the Fisher-KPP
reaction-diffusion PDE, against multi-modal MRI (and optionally PET) imaging data.

**Key published result:** ~64% → ~68% recurrence coverage improvement over standard
uniform margin, measured on **152 real glioblastoma patients** (their own cohort).

---

## 2. Public Code Status

GliODIL's official code repository is public:
**https://github.com/m1balcerak/GliODIL**

However, direct integration was **not feasible** for this environment because:
- Requires `libmpich-dev`, MPICH (MPI) C bindings, and Debian system packages
- Requires the `cselab/odil` C++ multigrid solver library
- Requires >18.5 GB GPU VRAM per patient, 30–45 min per optimization run
- Not portable to a Windows Python environment

**Approach taken:** A faithful **Python/PyTorch reimplementation** of GliODIL's core
discrete loss formulation, derived from the paper's Methods section.

---

## 3. What Was Reproduced Faithfully

The following choices match the paper directly (marked `# [FROM PAPER]` in code):

| Component | Paper specification | Our implementation |
|---|---|---|
| Solution representation | Discrete 3D voxel field c[i,j,k] as optimization variable | `nn.Parameter` voxel grid, same |
| PDE type | Fisher-KPP: ∂c/∂t = ∇·(D(x)∇c) + ρc(1-c) | Identical formulation |
| PDE discretization | Finite-difference operators on grid (not Autograd through a net) | FD stencils via `torch.roll` |
| Tissue-weighted diffusion | D(x) = D₀ · w_tissue(x); WM faster than GM | WM:1.0, GM:0.1, CSF:0.0 |
| Multi-resolution schedule | Multigrid coarse→fine decomposition | 64³ warm-start → 128³ fine-tune |
| D₀ initialization | ~0.1 mm²/day (log(-2.3)) | Same |
| ρ₀ initialization | ~0.1 /day (log(-2.3)) | Same |
| Loss weights | λ_data=1.0, λ_ic=1.0, λ_pde=1.0 | Same (equal weighting) |
| IC (initial condition) | Gaussian blob at tumor centroid at t=0 | Same |
| CSF exclusion | CSF/ventricles hard-excluded from treatment contour | Same (tissue_map == 3) |

### Key Architectural Distinction Preserved

The reimplementation correctly captures GliODIL's fundamental architectural difference
from a vanilla PINN:

| | Vanilla PINN (existing baseline) | GliODIL (this reproduction) |
|---|---|---|
| Solution | Continuous neural network c(x,y,z,t) | Discrete voxel field c[i,j,k] |
| PDE loss | Autograd through MLP | Finite-difference stencil on grid |
| Optimizes | Network weights | Voxel field values directly |
| Physics constraint strength | λ_pde=0.1 (soft) | λ_pde=1.0 (harder) |

This distinction matters for the comparison to be honest. The reimplementation is
**not** "vanilla PINN with a new name."

---

## 4. Approximations and Deviations

Lines marked `# [APPROXIMATION]` in code:

| Choice | Paper specification | Our approximation | Rationale |
|---|---|---|---|
| **Optimizer** | ODIL framework (MPI multigrid, L-BFGS via C++ solver) | Adam + 2-level multigrid in PyTorch | ODIL requires C++/MPI; Adam is widely available |
| **Iterations** | 30–45 min on >18.5 GB GPU (full multigrid) | 300 coarse + 1000 fine Adam steps (~2–5 min) | Hardware/portability constraint |
| **Multigrid levels** | Full multigrid (many levels) | 2 levels only (64³ → 128³) | Simplification; captures main benefit |
| **Boundary conditions** | Neumann (zero-flux) BCs | Periodic roll + boundary masking | Negligible for brain-interior voxels |
| **Time discretization** | Forward Euler (Δt in physical time) | Normalized Δt=1 (single time step) | Paper normalizes [0,1] time interval |
| **IC via backward Euler** | Explicit from paper's IC construction | Approximate back-step from t=1 | Paper's full IC requires additional data |
| **Sigmoid vs clamping** | Hard clamping c ∈ [0,1] | Sigmoid activation | Differentiable; numerically stable |

Lines marked `# [DEVIATION]` in code:

| Choice | Reason |
|---|---|
| D₀ init = log(0.1) vs. pipeline default log(0.01) | Matches paper's literature-based initialization |
| ρ₀ init = log(0.1) vs. pipeline default log(0.3) | Matches paper's initialization |
| No PET data used | Dataset contains MRI only; paper optionally uses FET-PET |

---

## 5. What This Comparison Can and Cannot Claim

### ✅ Valid claims from this comparison (N=4, same patients, same metrics)

- **Runtime comparison is fully valid and unambiguous:** GliODIL optimization takes
  ~minutes per patient; MarginSense takes ~seconds per patient (single forward pass).
  This is a legitimate, honest factual contrast that requires no accuracy claim.

- **Method behavior on our test set:** Descriptive per-patient metric differences
  (e.g., "GliODIL achieves X% coverage vs. MarginSense Y% on patient Z") are honest
  observations — presented as paired within-patient observations only.

- **Architectural contrast:** The comparison honestly demonstrates the difference
  between discrete field optimization vs. amortized neural network inference —
  both methodologically and in runtime.

### ❌ What this comparison CANNOT claim

- **No statistical significance.** N=4 is below any meaningful threshold for hypothesis
  testing. No p-values are reported or appropriate.

- **No claim that MarginSense "beats" GliODIL paper numbers.** GliODIL's ~68%
  recurrence coverage figure is from their 152-patient cohort. Our reproduction runs on
  4 different patients with different preprocessing and simplified optimization.
  These numbers are in a separate non-comparable reference section and must not be
  merged with our N=4 results.

- **No generalization.** Results on N=4 patients do not generalize to any population.

- **No validation of our reproduction's fidelity.** Without the authors' dataset and
  full ODIL implementation, we cannot verify that our reproduction achieves the same
  accuracy as the original GliODIL. Our reimplementation may underperform due to fewer
  iterations and simplified multigrid.

---

## 6. Runtime Comparison (Honest Assessment)

The paper explicitly acknowledges this trade-off. GliODIL is a per-patient optimization
method — this is its design choice, not a limitation to be hidden.

| Method | Runtime per patient | Mechanism |
|---|---|---|
| Clinical Standard | ~milliseconds | Morphological dilation (CPU) |
| Vanilla PINN | ~minutes (500 epochs) | Per-patient neural network fitting |
| **GliODIL (Reproduced)** | **~2–5 minutes (1000 Adam steps)** | **Per-patient discrete field optimization** |
| GliODIL (paper, full) | **30–45 minutes** | Full ODIL multigrid with MPI |
| **MarginSense** | **~seconds** | **Single forward pass (amortized)** |

Runtime is logged prominently in all outputs and the dashboard's Efficiency section.
MarginSense's single-forward-pass runtime advantage is a factual, unambiguous
differentiator that does not depend on any accuracy claim.

---

## 7. Citation

> Balcerak, M., et al. "Individualizing Glioma Radiotherapy Planning by Optimization
> of a Data and Physics-Informed Discrete Loss." *Nature Communications*, 2025.
> DOI: 10.1038/s41467-024-56098-y
> GitHub: https://github.com/m1balcerak/GliODIL
