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
Using a small ensemble (or Monte Carlo dropout) around the amortized PINN, MarginSense outputs a spatial uncertainty heatmap alongside the predicted infiltration boundary — showing oncologists not just "the estimated edge" but "how confident are we, region by region." This is closer to how a clinician actually needs to reason about a margin decision: where can we safely shrink the margin, and where should we be cautious even if the model's point estimate looks confident?

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

---

## 5. Future Scope

- Incorporate diffusion tensor imaging (DTI) where available for true anisotropic white-matter-tract-aware spread, rather than the coarse tissue-type proxy used in the hackathon version.
- Extend the amortized model with continual learning so it improves as more multi-center data becomes available, without full retraining.
- Couple the spatial infiltration model with a treatment-response timescale, updating predictions mid-course (during the multi-week radiotherapy schedule) as the tumor responds — moving from a one-time prediction to an adaptive planning tool.
- Pursue retrospective multi-center validation via platforms like PREDICT-GBM, which is already built for exactly this kind of model comparison, as a realistic path toward eventual prospective clinical evaluation.

---

## 6. Limitations (to state upfront, not hide)

- This is a research prototype trained/validated on public retrospective data — it is **not** validated for real clinical decision-making and would need years of further validation (multi-center retrospective studies, then prospective trials) before it could influence an actual treatment plan.
- Public longitudinal glioblastoma datasets are relatively small (low hundreds of patients), which limits how well an amortized model can generalize; performance on populations outside the training distribution is unverified.
- The tissue-weighted diffusion approximation is a proxy for true anisotropy, not a replacement for DTI-based modeling.
- Recurrence coverage improving on average does not guarantee improvement for every individual patient — as seen in prior work, some patients see large gains and others see none, and communicating that variability honestly (via the uncertainty map) is part of the design, not a flaw to be hidden.

---

## References
- GliODIL: Individualizing Glioma Radiotherapy Planning by Optimization of a Data and Physics-Informed Discrete Loss, *Nature Communications*, 2025.
- PREDICT-GBM: Platform for Robust Evaluation and Development of Individualized Computational Tumor Models in Glioblastoma, 2025.
- Single-snapshot PINN parameter estimation for glioblastoma reaction-diffusion modeling (Ezhov et al.).
