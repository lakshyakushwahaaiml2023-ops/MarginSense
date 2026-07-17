# MarginSense
### Physics-Informed, Uncertainty-Aware, Patient-Specific Radiotherapy Margin Optimization for Glioblastoma

---

## 1. The Problem

Glioblastoma (GBM) is the most aggressive primary brain cancer in adults. Roughly **250,000 new cases are diagnosed worldwide every year** (about 12,000 in the US alone). Despite surgery, radiotherapy, and chemotherapy, about **70% of patients progress within a single year** of finishing treatment, and five-year survival remains below 5%.

A major reason recurrence is so common: radiotherapy does not actually target where the tumor *will* grow -- it targets where the tumor already is, plus a fixed safety buffer. Since the 1970s-80s, the clinical standard has been to draw a **uniform 1.5 cm margin** around whatever is visible on a pre-treatment MRI scan and irradiate that zone. This number comes from population-level averages, not from anything specific to the patient in front of the oncologist.

The problem is that glioblastoma does not grow like a balloon inflating evenly outward -- it infiltrates along white matter tracts, blood vessels, and tissue planes, spreading faster in some directions than others. A uniform circular margin will simultaneously over-treat healthy brain tissue in some directions (causing unnecessary neurological damage) while under-treating the actual direction of spread in others (leading to recurrence just outside the treated field). Oncologists currently have no patient-specific, quantitative way to know which of these two errors they are making for any individual patient.

---

## 2. Existing Solutions and Where They Fall Short

This is an active research area, not an unexplored one:

- **Reaction-diffusion PDE models** (Fisher-Kolmogorov equation and variants) have long been used to describe tumor cell density spreading through tissue, and recent work has used Physics-Informed Neural Networks (PINNs) to fit these models to a patient MRI and infer patient-specific growth parameters.
- **GliODIL** (*Nature Communications*, 2025) is the current state-of-the-art, improving average recurrence coverage from ~64% to ~68% vs. standard margin on 152 real patients. Its authors explicitly moved *away* from pure PINNs because per-patient optimization is too slow for clinical use.
- **PREDICT-GBM** is now benchmarking competing models (U-Net, PINN) side-by-side on public data.

**The gap:** None of the current systems give clinicians (a) a *confidence map* -- they output a single boundary with no uncertainty quantification; (b) an *optimized* margin boundary that balances coverage vs. healthy-tissue toxicity for that specific patient; or (c) an explainable, factual account of *why* the predicted spread extends in a particular direction.

---

## 3. Our Idea

**MarginSense** is a full radiotherapy planning system built to close those gaps simultaneously. It does not stop at predicting infiltration probability -- it generates, optimizes, and explains the treatment margin.

**a) Amortized (meta-learned) PINN -- fast inference, no per-patient retraining.**
A single network trained across a population, conditioned on each patient imaging features. A new patient full infiltration map is a single forward pass, not a new training run.

**b) Ensemble uncertainty -- a confidence field, not just a boundary.**
Five independently seeded models form a deep ensemble. Their agreement (mean) and disagreement (std) produce a per-voxel confidence map. Regions of high uncertainty get a wider safety margin automatically.

**c) Continuous infiltration probability overlay -- not just a hard contour.**
The ensemble mean field is displayed as a continuous viridis-colored probability map (with isoprobability contour lines at 95%, 80%, 50%, 20%) directly on 2D slice views and as a point cloud in the 3D viewer, so clinicians see the full probability gradient rather than just a binary threshold.

**d) Margin Optimization Module -- replacing the fixed threshold.**
For each patient probability map, MarginSense sweeps threshold values 0 to 1 and computes a coverage-vs-healthy-tissue ROC curve. The optimal threshold is selected by maximizing coverage - lambda * normalized_healthy_volume, where lambda is a clinical priority slider (default calibrated to roughly match standard-margin toxicity). Organs at risk (CSF/ventricles from the tissue map) are excluded from the final contour.

**e) Recommended Safety Margin -- upper-confidence-bound rule.**
A third output map applies djusted = mean_prediction + z * uncertainty, then thresholds at the optimized threshold. The z slider (Safety Margin Caution Level) lets oncologists explicitly trade off model confidence against margin aggressiveness. This produces three distinct maps side by side: Prediction (where tumor likely is), Uncertainty (model disagreement), and Recommended Safety Margin (the boundary that grows in uncertain regions).

**f) Spread Explainability -- factual, template-filled, no LLM.**
For each patient, the system divides the tumor neighborhood into six anatomical sectors (anterior/posterior/medial/lateral/superior/inferior), computes how far the predicted infiltration probability extends beyond the visible tumor boundary per sector, and reports local white-matter fraction, diffusion coefficient D(x), and tumor elongation via PCA toward the dominant sector. All output is a fixed f-string template filled with computed values -- no free-form AI text generation. A fixed disclaimer is always appended.

**g) Comprehensive quantitative evaluation -- beyond Dice.**
MarginSense computes HD95 (95th-percentile Hausdorff, not raw max), Surface Dice at an explicit stated 2mm tolerance, Average Symmetric Surface Distance, Sensitivity/Specificity at the stated threshold, and Physics Residual MSE (Fisher-KPP PDE sanity check, labeled separately as *not* a clinical accuracy metric) -- all reported as mean +/- std across the test set, with a visible small-N disclaimer.

---

## 4. What This Solves (vs. Current Approaches)

| Current limitation | MarginSense answer |
|---|---|
| PINNs too slow for clinical use (per-patient retraining) | Amortized model -- one training run, fast single forward-pass per new patient |
| No generalization across patients | Shared network conditioned on patient-specific imaging features |
| Single hard boundary, no confidence signal | Per-voxel uncertainty heatmap from 5-model deep ensemble |
| Rigid isotropic diffusion assumption | Tissue-weighted diffusion (WM/GM/CSF) from routinely available segmentation |
| No clinical covariate integration | Age, KPS, IDH/MGMT status, resection extent, laterality encoded per inference |
| Fixed threshold regardless of patient anatomy | Patient-specific threshold optimization via lambda-weighted coverage-toxicity curve |
| No margin-level uncertainty propagation | UCB safety margin: contour expands automatically in high-uncertainty regions |
| Black-box spatial prediction, no explanation | Sector-based factual spread analysis using computed WM fraction, D(x), PCA elongation |
| Dice-only evaluation (known weak spot) | HD95, Surface Dice @ 2mm, ASD, Sensitivity/Specificity, Physics Residual -- mean +/- std |
| No accessible clinical interface | Interactive full-stack 3D web dashboard with upload, live pipeline, and auto-generated reports |

---

## 5. System Architecture and Implemented Components

### 5.1 Model Architecture (src/models/amortized_pinn.py)

The core model (MarginSenseNet) has three integrated sub-networks:

- **Encoder3D** -- A 3D CNN with four strided convolutional blocks (16->32->64->64 channels) followed by adaptive average pooling. It ingests a 5-channel 3D volume (T1, T1ce, T2, FLAIR + tumor segmentation mask) at 128 cubed resolution and outputs a 64-dimensional patient embedding (latent code z), patient-specific physical growth parameters (diffusion coefficient D and proliferation rate rho, estimated in log-space for positivity enforcement), and an optional 6-element clinical covariate vector (age, KPS, IDH status, MGMT methylation, resection extent, laterality) projected via a zero-initialized linear layer.

- **FiLMLayer** -- Feature-wise Linear Modulation layers implementing patient-conditioned affine transforms (gamma*x + beta) at every hidden layer. Initialized at identity so the network starts near a patient-agnostic baseline.

- **CoordinateMLP** -- A 4-layer FiLM-modulated MLP mapping spatio-temporal coordinates (x, y, z, t) in [0,1]^4 to a scalar cell density prediction c in [0,1]. Hidden layers: 64 units, Tanh activation, each independently conditioned on the patient embedding.

### 5.2 Training (src/training/train.py)

- Mixed-precision training (PyTorch torch.amp.autocast + GradScaler)
- Stratified data sampling: 50% recurrence region, 50% background per mini-batch
- Gaussian initial condition: cell density seeded as Gaussian blob at tumor centroid at t=0
- Three-term loss: L_data (MSE against recurrence labels at t=1), L_ic (MSE against Gaussian IC at t=0), L_pde (Fisher-Kolmogorov PDE residual via automatic differentiation with per-voxel tissue-weighted diffusion: WM x1.0, GM x0.1, CSF x0.0)
- TensorBoard logging and CSV export of all training metrics

### 5.3 Ensemble Training (src/training/train_ensemble.py)

Trains 5 independent MarginSenseNet models with different random seeds (42-46), with larger mini-batches and per-model TensorBoard runs. Ensemble diversity provides epistemic uncertainty estimates without architectural changes. Saved to outputs/marginsense_ensemble_{i}.pt.

### 5.4 Preprocessing Pipeline (src/preprocess.py)

Batch preprocessing of BraTS-format NIfTI directories: multi-modality loading (T1, T1ce, T2, FLAIR, segmentation), voxel resampling to 128 cubed, Z-score normalization over brain-masked voxels, tissue type estimation from T1 intensity percentiles (WM top 33%, GM middle, CSF bottom 33%), optional recurrence mask loading. Output: data/processed/{patient_id}.npz with image, label, spacing, recurrence, tissue_map arrays.

### 5.5 NIfTI Upload and Live Preprocessing Pipeline (src/preprocess_upload.py)

End-to-end preprocessing for new patient MRI uploads with no SimpleITK/ANTs/FSL dependency: accepts 4 NIfTI files via multipart form, automatic GTV approximation via intensity thresholding and morphological cleanup, Gaussian-smoothing bias field correction, clinical covariate ingestion, thread-safe SSE log streaming for real-time UI feedback.

### 5.6 Baseline Models

**Uniform Margin Baseline (src/baseline_uniform_margin.py):** Reproduces the 1.5 cm clinical standard using scipy Euclidean distance transform morphological dilation. Outputs dilated binary mask and wall-clock time.

**Vanilla Per-Patient PINN (src/baseline_vanilla_pinn.py):** Single-patient PINN with the same Fisher-Kolmogorov PDE loss but no amortization -- serves as the PINN-without-amortization ablation baseline.

### 5.7 Ensemble Uncertainty Quantification (src/evaluate_uncertainty.py)

Full 128 cubed grid evaluation across all 5 ensemble models: evaluates coordinate MLP in 131,072-voxel chunks to prevent CUDA OOM; computes per-voxel mean density and per-voxel standard deviation; outputs {patient_id}_prediction_ensemble.npz with mean_density, std_density, spacing, inference_time.

### 5.8 Comparative Evaluation Engine (src/compare_models.py)

Side-by-side comparison (ASCII + JSON) of all methods:
- Recurrence Coverage (%): fraction of post-treatment recurrence voxels captured inside the treatment volume
- Treated Healthy Tissue (cm3): treatment volume falling outside the original tumor
- Total Target Volume (cm3): absolute size of the irradiated volume
Results saved to outputs/{patient_id}_comparison_report.json.

### 5.9 Margin Optimization Module (/api/margin_sweep, /api/safety_metrics)

**Threshold sweep:** Sweeps threshold tau in [0.0, 1.0] in 0.02 increments, computing coverage and healthy-tissue volume at each tau. Returns a coverage-vs-healthy-tissue ROC curve displayed in the dashboard.

**Optimal threshold selection:** argmax(coverage - lambda * normalized_healthy_volume) where lambda is the clinical-priority slider (default ~50, calibrated to match standard-margin toxicity). CSF voxels (tissue_map == 3) are hard-excluded from the final contour.

**Safety Margin (UCB rule):** adjusted_voxel = mean_density + z * std_density, thresholded at the optimized tau. The z-slider (Safety Margin Caution Level, default z=1.0) is exposed live in the UI. At z=0: pure mean prediction. At z=2: boundary grows significantly into high-uncertainty regions.

### 5.10 Spread Explainability Module (/api/explain/<patient_id>)

Factual, template-filled spatial analysis -- no LLM, no free-form text generation:

1. Divides the region around the tumor centroid into 6 anatomical half-space sectors (superior/inferior/anterior/posterior/medial/lateral)
2. Per sector: 95th-percentile extension of infiltrated voxels (density > 0.20) beyond the 95th-percentile tumor radius
3. Per sector: local white-matter fraction vs. global brain average
4. Per sector: mean D(x) using tissue-specific diffusion constants (WM: 0.15 mm2/day, GM: 0.03 mm2/day -- literature-grounded Giese/Swanson model)
5. PCA elongation of tumor voxels toward dominant direction (alignment score, sphericity = lambda_min/lambda_max)
6. Global factors: estimated rho, IDH status, MGMT status with literature associations (Hegi 2005, Dang 2009)
7. Fixed f-string template -- output is fully deterministic and reproducible
8. Non-removable disclaimer: results reflect statistical association, not verified causal explanation

Displayed in the SPREAD EXPLAINABILITY sidebar panel and in Report Section 6.

### 5.11 Comprehensive Evaluation Module (/api/evaluation_metrics, src/evaluate_pipeline.py)

MarginSense features a complete, quantitative validation module consisting of two parts:
1. **Interactive Evaluation Dashboard API (`/api/evaluation_metrics`)**: Dynamic query endpoint returning mean ± std across cached test-set ensemble predictions.
2. **Leave-One-Out Cross-Validation (LOOCV) Pipeline (`src/evaluate_pipeline.py`)**: A command-line evaluation framework that splits the dataset, trains the amortized network from scratch on $N-1$ patients (ensuring zero overlap or data leakage), and validates on the held-out patient at every epoch to log train-vs-validation curves.

Metrics evaluated side-by-side (Clinical Standard vs. Vanilla PINN vs. MarginSense):
- **Recurrence Coverage (%)** — fraction of post-treatment recurrence captured.
- **Sensitivity / Specificity** — voxel-wise sensitivity and specificity computed at the optimized threshold.
- **Margin Volume and Healthy Tissue Irradiated (cm³)** — spatial toxicity metrics.
- **Hausdorff Distance (HD95, mm)** — 95th-percentile symmetric surface distance to exclude single-voxel noise, following the BraTS benchmark standard (Menze et al., IEEE TMI 2015).
- **Surface Dice @ 2mm** — surface-to-surface alignment index at a 2mm tolerance.
- **Average Surface Distance (ASD, mm)** — average boundary-to-boundary distance.
- **Efficiency Metrics** — inference/grid evaluation time (seconds) and peak GPU memory allocation (MB) monitored dynamically during the prediction forward passes.
- **Model Sanity Check** — spatial physics consistency reported as the Mean Squared PDE Residual (Fisher-KPP) over white and gray matter brain voxels (labeled separately from clinical metrics).

All comparative reports are exported as machine-readable JSON/CSV files and a presentation-ready Markdown table, displaying `mean ± std (min - max)` and a prominent small-N banner (N=4 patients) warning of preliminary, exploratory findings.

Methods compared: Clinical Standard (uniform 1.5cm), Vanilla PINN (per-patient), MarginSense (ensemble amortized).

### 5.12 3D Interactive Visualizer (src/viz/)

A full-stack web application (Flask + Three.js).

**Backend (server.py) -- complete API surface:**

| Endpoint | Purpose |
|----------|---------|
| GET /api/patients | Lists all available patients |
| POST /api/upload_patient | Accepts 4 NIfTI files + covariates; starts background preprocessing |
| GET /api/upload_progress/<id> | SSE stream of live preprocessing log lines |
| POST /api/start_pipeline | Triggers 4-stage planning in a background thread |
| GET /api/live_logs | Current pipeline log buffer and running state |
| GET /api/patient_data/<id> | Per-patient JSON data cache |
| GET /api/models/<id>/<type> | Binary GLB mesh files for Three.js |
| GET /api/comparison/<id> | Comparative metrics JSON |
| GET /api/margin_sweep/<id> | Coverage-vs-healthy ROC sweep data |
| GET /api/safety_metrics/<id> | Safety margin UCB metrics at current z/lambda |
| GET /api/generate_report/<id> | Structured deterministic patient report (7 sections) |
| GET /api/explain/<id> | Sector-based spread explainability analysis |
| GET /api/evaluation_metrics | Comprehensive test-set evaluation (HD95, Surface Dice, ASD, etc.) |
| GET /api/gpu_status | Live GPU utilization and VRAM via nvidia-smi |
| GET /api/patient_covariates/<id> | Stored clinical covariates |

**Frontend (index.html) -- dashboard features:**

- 3D brain mesh in Three.js with a **translucent shell** (30% opacity default, depthWrite:false + renderOrder layering so tumor and margins remain visible through it). Live **opacity slider** (0% = hidden, 30% = default, 100% = opaque) next to the Brain Envelope toggle.
- Toggleable overlays: GTV, Standard CTV, MarginSense CTV (0.35), Optimized CTV, Recommended Safety Margin (purple)
- **Continuous probability overlay** on 2D slice views: viridis colormap, color-scale legend, hover tooltip showing exact infiltration probability at cursor
- **Isoprobability contour lines** at 95%, 80%, 50%, 20% (toggled separately from the threshold contour)
- **Continuous 3D probability cloud** (point cloud colored by infiltration probability, toggleable)
- Temporal density animation slider: Fisher-Kolmogorov infiltration wave at t in {0.0 to 1.0}
- **Margin Optimization panel**: lambda slider, z slider, optimal threshold display, ROC chart
- **Three maps side by side in report**: Prediction (ensemble mean), Uncertainty (ensemble std), Recommended Safety Margin
- **SPREAD EXPLAINABILITY panel**: sector bar chart (6 directions, dominant highlighted in color), template summary text, non-removable disclaimer
- **Model Evaluation modal**: three-section table (Clinical Accuracy / Efficiency / Model Sanity Check), mean +/- std and [min-max], per-patient collapsible detail, amber small-N warning
- **Patient Report (7 sections)**: Patient/Tumor Profile, Infiltration Probability (3 maps), Comparative Metrics, Contour Overlay, Discussion Points, Spread Explainability, Disclaimer
- Live GPU monitor chart, training loss chart, patient latent space projection (UMAP-style)
- 2D radiology slice viewer with axial/coronal/sagittal views and canvas hover tooltips
- Patient upload form with covariate fields and real-time SSE progress display

---

## 6. Key Technical Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| depthWrite:false on brain shell + renderOrder | Standard Three.js solution for translucent shell over opaque inner meshes -- avoids z-fighting without transmission/refraction (glassmorphism avoided) |
| HD95 not raw Hausdorff | Raw max Hausdorff is dominated by a single stray voxel -- HD95 is the BraTS benchmark standard (Menze et al., IEEE TMI 2015) |
| Surface Dice at explicitly stated 2mm | Tolerance changes the number significantly; must always be stated alongside |
| Physics Residual labeled separately | It is a model sanity check, not a clinical accuracy metric -- mixing them would mislead |
| No LLM in explainability | Template-filled with computed values: fully deterministic, reproducible, auditable |
| scipy EDT for surface metrics, not medpy | medpy not installed; scipy distance_transform_edt produces identical results |
| Fixed disclaimer always appended to explainability | Outputs reflect statistical association, not verified causal mechanism |
| Small-N amber banner on evaluation | N=4 patients -- numeric results presented with accurate confidence framing |

---

## 7. Future Scope

- Incorporate diffusion tensor imaging (DTI) for true white-matter-tract-aware anisotropic spread modeling
- Continual learning so the amortized model improves as multi-center data accumulates, without full retraining
- Mid-course adaptive replanning: update predictions during the multi-week radiotherapy schedule as the tumor responds
- Prospective multi-center validation via PREDICT-GBM
- Replace intensity-threshold GTV approximation with BraTS-trained nnU-Net for real patient data
- Monte Carlo Dropout as an alternative UQ method alongside deep ensemble, with calibration comparison
- Calibration curves (reliability diagrams, ECE) to assess whether model 70% confidence reflects 70% empirical accuracy

---

## 8. Limitations (stated upfront)

- This is a **research prototype** trained and validated on public retrospective data -- it is not validated for real clinical decision-making and would need years of further validation (multi-center retrospective studies, then prospective trials) before influencing an actual treatment plan
- Public longitudinal GBM datasets are small (low hundreds of patients), limiting amortized generalization to populations outside the training distribution
- Tissue-weighted diffusion is a proxy for true anisotropy, not a replacement for DTI-based modeling
- Automatic GTV segmentation uses intensity thresholding -- not a validated clinical segmentation method
- Improvement on average does not guarantee improvement for every individual patient; the uncertainty map is designed to communicate this variability, not hide it
- The ensemble captures epistemic (model) uncertainty, not aleatoric (MRI acquisition noise) uncertainty
- Small dataset size (N=4 patients) -- all cross-validation and quantitative results are explicitly labeled to treat as preliminary, exploratory evidence rather than statistically validated findings

---

## References

- GliODIL: Individualizing Glioma Radiotherapy Planning by Optimization of a Data and Physics-Informed Discrete Loss, Nature Communications, 2025
- PREDICT-GBM: Platform for Robust Evaluation and Development of Individualized Computational Tumor Models in Glioblastoma, 2025
- Menze et al., The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS), IEEE TMI, 2015 (HD95 standard)
- Nikolov et al., Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy, 2018 (Surface Dice @ 2mm)
- Hegi et al., MGMT Gene Silencing and Benefit from Temozolomide in Glioblastoma, NEJM, 2005
- Dang et al., Brain cancer metabolism, Cell, 2009 (IDH mutation / proliferation rate association)
- Swanson et al., A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma, British Journal of Cancer, 2008 (Fisher-KPP D/rho parameter ranges)
- Feature-wise Transformations (FiLM): Visual Reasoning with a General Conditioning Layer, Perez et al., AAAI 2018
- Deep Ensembles: A Loss Landscape Perspective, Fort et al., NeurIPS 2019
