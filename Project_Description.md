# MarginSense
### Fast, Uncertainty-Aware, Patient-Specific Radiotherapy Margins for Glioblastoma

---

## 1. The Problem

Glioblastoma (GBM) is the most aggressive primary brain cancer in adults. Roughly **250,000 new cases are diagnosed worldwide every year** (about 12,000 of them in the US alone). Despite surgery, radiotherapy, and chemotherapy, about **70% of patients progress within a single year** of finishing treatment, and five-year survival remains below 5%.

A major reason recurrence is so common: radiotherapy doesn't actually target where the tumor will grow — it targets where the tumor *already is*, plus a fixed safety buffer. Since the 1970s–80s, the clinical standard has been to draw a **uniform 1.5 cm margin** around whatever is visible on a pre-treatment MRI scan and irradiate that. This number comes from population-level averages, not from anything specific to the patient in front of the oncologist.

The problem is that glioblastoma doesn't grow like a balloon inflating evenly outward — it infiltrates along white matter tracts, blood vessels, and tissue planes, spreading faster in some directions than others. A uniform circular margin will over-treat healthy brain tissue in some directions (causing unnecessary neurological damage) while under-treating the actual direction of spread in others (leading to a recurrence just outside the treated field — which is exactly what happens in a large share of cases). Oncologists currently have no patient-specific, quantitative way to know which of these two errors they're making for any individual patient.

---

## 2. Existing Solutions — and Where They Fall Short

This is an active research area, not an unexplored one, so it's important to be precise about what already exists:

- **Reaction–diffusion PDE models** (Fisher-Kolmogorov equation and variants) have long been used to describe tumor cell density spreading through tissue, and recent work has used Physics-Informed Neural Networks (PINNs) to fit these models to a patient's MRI and infer patient-specific growth parameters.
- **GliODIL** (published in *Nature Communications*, 2025) is the current state-of-the-art system. Tested on 152 real glioblastoma patients with pre- and post-treatment scans, it improved average recurrence coverage from about 64% to 68% (and from 70% to 73% when PET imaging was available) compared to the standard uniform margin — roughly doubling the share of patients who saw a meaningful benefit (from ~21% to ~35%).
- **PREDICT-GBM**, a newer multi-center benchmarking platform, is now pooling several competing models (including U-Net-based and PINN-based approaches) to evaluate them side by side on public data.

The gap: GliODIL's authors explicitly moved *away* from a pure PINN formulation because, in their own words, PINNs are too computationally slow for realistic clinical workflows, and a PINN trained for one patient has to be retrained from scratch for the next — there is no shared, reusable model across patients. Separately, none of the current systems give clinicians a *confidence* map — they output a single boundary, when in reality the confidence in that boundary varies by region and by how much imaging data is available.

---

## 3. Our Idea

**MarginSense** is a PINN-based system built specifically to close those two gaps: speed/generalization, and honest uncertainty.

**a) Amortized (meta-learned) PINN, not per-patient optimization.**
Instead of training a fresh network for every new patient, we train a single network across a population of patients, conditioning it on each patient's imaging features (via a lightweight hypernetwork/embedding layer). Once trained, predicting a new patient's infiltration map is a single fast forward pass rather than a new optimization run — directly answering the speed and generalization limitation that pushed prior work away from PINNs.

**b) A confidence map, not just a boundary.**
Using a 5-model deep ensemble around the amortized PINN, MarginSense outputs a spatial uncertainty heatmap alongside the predicted infiltration boundary — showing oncologists not just "the estimated edge" but "how confident are we, region by region." This is closer to how a clinician actually needs to reason about a margin decision: where can we safely shrink the margin, and where should we be cautious even if the model's point estimate looks confident?

**c) Physics grounded, but not physics-rigid.**
The Fisher-Kolmogorov PDE residual is used as a soft physics-loss constraint (as in prior work), but weighted per-voxel by tissue type from a segmentation mask (white matter vs. gray matter vs. CSF), giving a cheap, principled approximation of anisotropic spread without requiring diffusion tensor imaging, which isn't available for every patient.

**d) Benchmarked the same way the field already benchmarks itself.**
We validate on the same publicly available GliODIL/BraTS-style longitudinal dataset (pre-treatment scan + post-treatment recurrence follow-up) and report the identical metric — recurrence coverage vs. the standard 1.5 cm margin — so our result is directly comparable to published numbers rather than a standalone claim.

---

## 4. What This Solves (vs. Current Issues)

| Current limitation | MarginSense's answer |
|---|---|
| PINNs too slow for clinical use (per-patient training) | Amortized model — one training run, fast inference per new patient |
| No generalization across patients | Shared network conditioned on patient-specific features |
| Single hard boundary, no confidence signal | Per-region uncertainty heatmap alongside the prediction |
| Rigid isotropic diffusion assumption | Tissue-weighted diffusion using routinely available segmentation, not just a single global diffusion constant |
| No clinical covariate integration | Patient age, KPS score, IDH/MGMT status, resection extent, and laterality encoded into every inference |

---

## 5. System Architecture & Implemented Components

### 5.1 Model Architecture (`src/models/amortized_pinn.py`)

The core model (`MarginSenseNet`) has three integrated sub-networks:

- **`Encoder3D`** — A 3D CNN with four strided convolutional blocks (16→32→64→64 channels) followed by adaptive average pooling. It ingests a **5-channel** 3D volume (T1, T1ce, T2, FLAIR + tumor segmentation mask) at 128³ resolution and outputs:
  - A **64-dimensional patient embedding** (latent code `z`)
  - **Patient-specific physical growth parameters** (diffusion coefficient *D* and proliferation rate *ρ*, estimated in log-space for positivity enforcement)
  - An optional **6-element clinical covariate vector** (age, KPS, IDH status, MGMT methylation, resection extent, laterality) projected into the feature space via a zero-initialized linear layer that activates only when covariate data is provided

- **`FiLMLayer`** — Feature-wise Linear Modulation layers that implement patient-conditioned affine transformations (γ·x + β) at every hidden layer of the coordinate network, where γ and β are computed from the patient embedding *z*. Initialized at identity (γ=1, β=0) so the network starts near a patient-agnostic baseline and gradually specializes.

- **`CoordinateMLP`** — A 4-layer FiLM-modulated MLP that maps spatio-temporal coordinates (x, y, z, t) ∈ [0,1]⁴ to a scalar cell density prediction c ∈ [0,1]. Each hidden layer (64 units, Tanh activation) is independently conditioned on the patient embedding, so the same network evaluates differently for different patients without retraining.

### 5.2 Training (`src/training/train.py`)

The amortized training loop implements:

- **Mixed-precision training** (PyTorch `torch.amp.autocast` + `GradScaler`) for GPU efficiency
- **Stratified data sampling**: 50% from post-treatment recurrence region (c=1 targets), 50% from background (c=0 targets) per mini-batch to handle class imbalance
- **Gaussian initial condition**: At t=0, cell density is seeded as a Gaussian blob centered on the pre-treatment tumor centroid
- **Three-term loss function**:
  - *L_data* — MSE against recurrence/non-recurrence labels at t=1
  - *L_ic* — MSE against Gaussian initial condition at t=0
  - *L_pde* — Fisher-Kolmogorov PDE residual computed via automatic differentiation (spatial Laplacian + logistic growth term), with **per-voxel tissue-weighted diffusion** (WM: ×1.0, GM: ×0.1, CSF: ×0.0, necrotic/edema: ×0.5) applied to the global diffusion parameter *D*
- **TensorBoard logging** (loss components, physics parameters *D* and *ρ* per epoch) and CSV export of all training metrics
- Multi-patient shuffling per epoch; the same model is optimized across all patients in one training run

### 5.3 Ensemble Training (`src/training/train_ensemble.py`)

Trains **5 independent MarginSenseNet models** with different random seeds (42–46), each with:
- Larger mini-batches (4096 data points, 2048 IC points, 4096 PDE collocation points vs. 2048/1024/2048 in single training)
- Per-model TensorBoard runs in `outputs/runs/ensemble_model_{i}/`
- Saved checkpoints at `outputs/marginsense_ensemble_{i}.pt`

The ensemble diversity (different random seeds → different loss landscapes → different local minima) provides epistemic uncertainty estimates without any architectural changes.

### 5.4 Preprocessing Pipeline (`src/preprocess.py`)

Batch preprocessing of BraTS-format raw NIfTI patient directories:

- **Multi-modality loading**: T1, T1ce, T2, FLAIR, and tumor segmentation (`.nii` / `.nii.gz`)
- **Voxel resampling**: All volumes resampled to 128³ using scipy's `zoom` (order=1 bilinear for images, order=0 nearest-neighbor for masks)
- **Z-score normalization** of each modality over brain-masked (non-zero) voxels only
- **Resampled spacing** calculated from original NIfTI header zoom factors and recorded in the output NPZ for volumetric metric computation (cm³)
- **Tissue type estimation** from T1 intensity percentiles: White Matter (top 33%), Gray Matter (middle), CSF (bottom 33%), with tumor overriding healthy labels
- **Optional recurrence mask**: If a post-treatment NIfTI (`_recurrence.nii[.gz]`) is present, it is loaded and resampled; otherwise the pre-treatment tumor mask is used as a training-time proxy
- Output: compressed `.npz` bundles at `data/processed/{patient_id}.npz` containing `image`, `label`, `spacing`, `recurrence`, and `tissue_map` arrays

### 5.5 NIfTI Upload & Live Preprocessing Pipeline (`src/preprocess_upload.py`)

End-to-end preprocessing pipeline for **new patient MRI uploads** (no SimpleITK/ANTs/FSL dependency):

- Accepts T1, T1ce, T2, FLAIR NIfTI files via multipart form upload
- **Automatic GTV approximation**: intensity thresholding on T1ce and FLAIR with morphological cleanup (binary fill holes, erosion, dilation, connected-component filtering) to approximate the Gross Tumor Volume without a manual segmentation
- **Bias field correction**: Gaussian smoothing-based inhomogeneity correction
- **Tissue segmentation**: T1 intensity percentile-based WM/GM/CSF classification on healthy brain tissue
- **Clinical covariate ingestion**: age, KPS score, IDH status, MGMT methylation, resection extent, and laterality parsed from form fields and saved as `{patient_id}_covariates.json`
- **Progress streaming**: thread-safe log accumulation exposed via Server-Sent Events (SSE) for real-time UI feedback during the multi-second preprocessing run

### 5.6 Baseline Models

**Uniform Margin Baseline (`src/baseline_uniform_margin.py`):**
- Reproduces the 1.5 cm clinical standard using Euclidean distance transform (scipy EDT) morphological dilation
- Outputs dilated binary mask and wall-clock time for direct comparison

**Vanilla Per-Patient PINN (`src/baseline_vanilla_pinn.py`):**
- Single-patient PINN with the same Fisher-Kolmogorov PDE loss but no amortization (encoder re-initialized and re-optimized from scratch per patient)
- Serves as the "PINN without amortization" ablation baseline

### 5.7 Ensemble Uncertainty Quantification (`src/evaluate_uncertainty.py`)

Full 128³ grid evaluation using all 5 ensemble models:

- Loads 5 checkpoints, runs the 3D CNN encoder once per model on the patient volume
- Evaluates the coordinate MLP in 131,072-voxel chunks to prevent CUDA OOM
- Computes **per-voxel mean density** and **per-voxel standard deviation** across the 5 models
- Logs diagnostic metrics: mean uncertainty inside the tumor core vs. the predicted infiltration zone vs. global maximum
- Outputs compressed `{patient_id}_prediction_ensemble.npz` containing `mean_density`, `std_density`, `spacing`, and `inference_time`

### 5.8 Comparative Evaluation Engine (`src/compare_models.py`)

Computes and prints a side-by-side ASCII table with three metrics for each method:

| Metric | Definition |
|--------|-----------|
| **Recurrence Coverage (%)** | Fraction of the post-treatment recurrence voxels captured inside the treatment volume |
| **Treated Healthy Tissue Volume (cm³)** | Volume of the treatment mask falling outside the original tumor — lower is better |
| **Total Target Volume (cm³)** | Absolute size of the irradiated volume |

Results are saved as a structured JSON report at `outputs/{patient_id}_comparison_report.json`.

### 5.9 3D Interactive Visualizer (`src/viz/`)

A full-stack web application (Flask + Three.js) with:

**Backend (`server.py`):**
- `GET /api/patients` — lists all available patients (raw directories + preprocessed NPZs + synthetic fallbacks)
- `POST /api/upload_patient` — accepts 4 NIfTI files + covariates, saves uploads, and starts background preprocessing thread; returns patient ID immediately
- `GET /api/upload_progress/<patient_id>` — **SSE stream** of live preprocessing log lines for real-time UI progress display
- `POST /api/start_pipeline` — triggers the full 4-stage live planning simulation (uniform margin → vanilla PINN → MarginSense ensemble → metric comparison → mesh export) in a background thread
- `GET /api/live_logs` — returns current pipeline log buffer and running state for polling
- `GET /api/patient_data/<patient_id>` — serves per-patient JSON data cache; runs the mesh/texture exporter on cache miss
- `GET /api/models/<patient_id>/<model_type>` — serves binary GLB mesh files (brain template, tumor, uniform margin, predicted margin) for Three.js
- `GET /api/comparison/<patient_id>` — serves the comparative metrics JSON report
- `GET /api/generate_report/<patient_id>` — assembles a **structured deterministic patient report** from pipeline outputs (comparison JSON + covariate JSON + ensemble NPZ), flagging which covariate fields used literature defaults vs. patient-supplied values
- `GET /api/gpu_status` — queries live GPU utilization and VRAM usage via `nvidia-smi`
- `GET /api/patient_covariates/<patient_id>` — serves stored clinical covariates

**Frontend (`index.html`):**
- 3D brain mesh rendered in Three.js with toggleable overlays (tumor core, uniform margin boundary, predicted anisotropic margin, uncertainty heatmap)
- Temporal density animation slider showing the Fisher-Kolmogorov infiltration wave at t ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- Side-by-side metrics panel comparing all three methods
- Live log console for the planning pipeline
- Patient upload form with covariate input fields

**Data Export (`export_json.py`, `export_meshes.py`):**
- Downsamples all volumes from 128³ to 32³ for web transfer efficiency
- Serializes brain mask, tumor mask, uniform margin, temporal density grids, and uncertainty map to `data.js` / per-patient JSON
- Generates GLB surface meshes via marching cubes for Three.js rendering

---

## 6. What This Solves (vs. Current Issues)

| Current limitation | MarginSense's answer |
|---|---|
| PINNs too slow for clinical use (per-patient training) | Amortized model — one training run, fast single forward-pass inference per new patient |
| No generalization across patients | Shared network conditioned on patient-specific imaging features |
| Single hard boundary, no confidence signal | Per-voxel uncertainty heatmap from 5-model deep ensemble |
| Rigid isotropic diffusion assumption | Tissue-type-weighted (WM/GM/CSF) anisotropic diffusion using routinely available segmentation |
| No clinical covariate integration | Age, KPS, IDH/MGMT status, resection extent, laterality conditioning via encoder covariate projection |
| No accessible clinical interface | Interactive 3D web visualizer with upload, live pipeline, and auto-generated patient reports |

---

## 7. Future Scope

- Incorporate diffusion tensor imaging (DTI) where available for true anisotropic white-matter-tract-aware spread, rather than the coarse tissue-type proxy used in the current version.
- Extend the amortized model with continual learning so it improves as more multi-center data becomes available, without full retraining.
- Couple the spatial infiltration model with a treatment-response timescale, updating predictions mid-course (during the multi-week radiotherapy schedule) as the tumor responds — moving from a one-time prediction to an adaptive planning tool.
- Pursue retrospective multi-center validation via platforms like PREDICT-GBM, which is already built for exactly this kind of model comparison, as a realistic path toward eventual prospective clinical evaluation.
- Replace the intensity-threshold GTV approximation in the upload pipeline with a validated automated segmentation model (e.g., a BraTS-trained nnU-Net) for real patient data.
- Add Monte Carlo Dropout as an alternative uncertainty quantification method alongside the deep ensemble, and compare calibration between both approaches.

---

## 8. Limitations (to state upfront, not hide)

- This is a research prototype trained/validated on public retrospective data — it is **not** validated for real clinical decision-making and would need years of further validation (multi-center retrospective studies, then prospective trials) before it could influence an actual treatment plan.
- Public longitudinal glioblastoma datasets are relatively small (low hundreds of patients), which limits how well an amortized model can generalize; performance on populations outside the training distribution is unverified.
- The tissue-weighted diffusion approximation is a proxy for true anisotropy, not a replacement for DTI-based modeling.
- The automatic GTV segmentation in the upload pipeline uses intensity thresholding — it is an approximation for demonstration purposes and is **not** a validated clinical segmentation method.
- Recurrence coverage improving on average does not guarantee improvement for every individual patient — as seen in prior work, some patients see large gains and others see none, and communicating that variability honestly (via the uncertainty map) is part of the design, not a flaw to be hidden.
- The ensemble-based uncertainty estimate captures model uncertainty (epistemic) but not measurement noise in the MRI acquisition (aleatoric), so the uncertainty map does not represent all sources of prediction variability.

---

## References
- GliODIL: Individualizing Glioma Radiotherapy Planning by Optimization of a Data and Physics-Informed Discrete Loss, *Nature Communications*, 2025.
- PREDICT-GBM: Platform for Robust Evaluation and Development of Individualized Computational Tumor Models in Glioblastoma, 2025.
- Single-snapshot PINN parameter estimation for glioblastoma reaction-diffusion modeling (Ezhov et al.).
- Feature-wise Transformations (FiLM): Visual Reasoning with a General Conditioning Layer, Perez et al., AAAI 2018.
- Deep Ensembles: A Loss Landscape Perspective, Fort et al., NeurIPS 2019.
