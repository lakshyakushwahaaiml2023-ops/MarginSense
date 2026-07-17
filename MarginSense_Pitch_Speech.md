# MarginSense — Pitch Speech & Full Technical Walkthrough

This is written in two parts. **Part 1** is what you actually say on stage — simple, attention-grabbing, timed. **Part 2** is the full technical deep-dive, for when a judge asks "okay, but how does it actually work" during Q&A. Read Part 1 out loud a few times until it feels like your own words, not a script.

---

## PART 1: THE PITCH (aim for 3–4 minutes)

### Opening hook — start here, simple, no jargon (20 seconds)

"I want to start with one image."

*[Show the pre-treatment scan next to the recurrence scan, standard margin overlaid, recurrence appearing outside it]*

"This is a real glioblastoma patient. On the left, their brain tumor before treatment. On the right, months later — the tumor came back. Doctors drew a margin around the tumor and treated everything inside it. The cancer came back just outside the line they drew.

That circle is the same shape, the same size, for every patient, everywhere in the world, and it hasn't meaningfully changed since the 1980s."

### Why this matters — the numbers (20 seconds)

"About 250,000 people get this diagnosis every year. Seventy percent of them will progress within a single year of finishing treatment. Five-year survival is under 5%.

The problem isn't that doctors aren't trying hard enough. It's that this tumor doesn't grow like a balloon — it infiltrates along the brain's own wiring, its white matter tracts, faster in some directions than others. A circle can't know that. Only this specific patient's anatomy knows that."

### What exists today, and where it falls short (30 seconds)

"This isn't an unexplored problem — some of the best labs in the world are working on it. In 2025, a system called GliODIL showed that patient-specific modeling of tumor spread genuinely improves on the standard margin, tested on over 150 real patients.

But the authors of that system told us something important themselves: the physics-based AI approach they started with — something called a Physics-Informed Neural Network — was too slow and had to be retrained from scratch for every single patient. So they had to move away from it.

That's exactly the gap we built for."

### Your solution, in one sentence (15 seconds)

"MarginSense is a physics-informed AI that learns once, across many patients, and then predicts a new patient's tumor spread in under a second — with a confidence map telling doctors exactly where to trust it, and where not to."

### The "wow" moment — hand off to the demo (rest of your time)

"Let me show you what that actually looks like."

*[Live demo: real brain, real MRI slices, toggle standard margin vs. MarginSense, watch the GPU timer, show the three maps — prediction, uncertainty, recommended margin]*

### Closing line — say this, don't skip it

"We're not claiming we cure glioblastoma. We're claiming something we can actually back up: for the same tumor coverage, we can meaningfully shrink how much healthy brain gets irradiated — which the research literature says matters, because larger margins carry real cognitive risk without much recurrence benefit. That's a real, honest, measurable improvement. And it runs on a single GPU, in under a second."

---

## PART 2: FULL TECHNICAL WALKTHROUGH (for Q&A, or a technical judge who wants depth)

Use this as your mental map — you don't need to recite all of it, just know where to go when someone asks "how does X work?"

### 1. What goes into the model

Every patient prediction is built from four kinds of input, not just one scan:
- **Imaging**: T1, T1ce (contrast-enhanced), T2, and FLAIR MRI sequences — the same four sequences used in the standard BraTS research dataset.
- **Segmentations**: a tumor mask, and a tissue map splitting the brain into white matter, gray matter, and CSF.
- **Manually entered clinical data**: age, IDH mutation status, MGMT methylation status, Karnofsky Performance Score, and extent of surgical resection.
- **Automatically computed features**: tumor volume, which hemisphere, approximate lobe location, distance from the ventricles, and a shape-irregularity (sphericity) score — all derived directly from the segmentation, no extra manual entry needed.

### 2. The patient encoder

All of that goes into an encoder that compresses it into a single vector — a "patient latent representation," which you can think of as this patient's unique biological fingerprint in a compact numerical form. That fingerprint is used two ways: it's decoded into two biological parameters specific to this patient — a diffusion coefficient (how fast their tumor cells migrate) and a proliferation rate (how fast they multiply) — and it also conditions the physics model that comes next, which is what makes this patient-specific rather than one-size-fits-all.

### 3. The physics core: Fisher-KPP reaction-diffusion

This is the part that makes it more than "just deep learning." We use a real, 25-year-old published model from mathematical oncology called the Fisher-Kolmogorov-Petrovsky-Piskunov equation. In plain language: at every point in the brain, the tumor's cell density changes over time because of two things — cells spreading into neighboring tissue, and cells multiplying locally until they hit a saturation limit. The equation is:

∂u/∂t = ∇·(D(x)∇u) + ρu(1−u)

The key upgrade over the textbook version: **D, the diffusion rate, isn't one number — it changes depending on what tissue you're in.** White matter gets a diffusion rate about five times higher than gray matter, because tumor cells physically travel faster along myelinated fiber tracts — that ratio comes directly from clinical observation in the literature. CSF — the fluid-filled ventricles — gets a diffusion rate of exactly zero, because tumor cells can't migrate through fluid. So the model doesn't just grow a blob outward; it bends around ventricles and races along white matter, the same way real infiltration does.

### 4. The AI prediction module

The network doesn't output a yes/no "tumor here" answer. It outputs a continuous number between 0 and 1 at every point in the brain — a density/confidence field. That means instead of one hard line, we can show doctors an entire gradient: 95% likely, 80% likely, 50%, 20%, all the way down — like a weather map's probability bands, instead of a single boundary.

### 5. Margin optimization

Here's where it becomes a decision-support tool, not just a picture. We don't pick one arbitrary cutoff and call it done. We sweep every possible threshold from 0 to 1, and for each one, calculate two things at once: how much of the eventual recurrence it would have covered, and how much healthy tissue it would irradiate. That gives us a full trade-off curve. Then we mathematically select the point on that curve that maximizes coverage while minimizing healthy-tissue damage, weighted by an adjustable dial a clinician could tune based on how conservative they want to be. That's a computed, patient-specific, justifiable choice — not a number someone picked once and reused for everyone.

### 6. Uncertainty — three maps, not one

We don't train one model, we train an ensemble of several, and look at how much they agree. That gives us three distinct outputs, side by side:
- **Prediction** — where the tumor is likely to be.
- **Uncertainty** — where the models disagree with each other, meaning we should trust the answer less.
- **Recommended Safety Margin** — a combined map that automatically grows the margin in exactly the places where uncertainty is high, and stays tight where the model is confident. Mathematically, this is a well-known technique called an upper-confidence-bound rule: we add a multiple of the uncertainty to the prediction before applying the threshold, so uncertain regions get pulled toward "include this to be safe" automatically.

### 7. Explainability

Every prediction comes with a factual, computed explanation of *why* the spread looks the way it does — not a vague AI-generated paragraph, but real numbers pulled from the same pipeline: "spread extends furthest toward the front of the brain because we detected a higher fraction of white matter in that direction, and a higher local diffusion estimate there, consistent with this tumor's elongated shape in that direction." Every clause in that sentence is a number we actually computed, not something invented after the fact.

### 8. Evaluation — going beyond one accuracy number

We report a full clinical metrics suite, not just a single overlap score: how much of the actual recurrence is captured, how much healthy tissue is irradiated, how physically small the margin is, sensitivity and specificity, boundary-accuracy metrics used in real segmentation research (like the 95th-percentile Hausdorff distance, which is more robust than raw Hausdorff distance), how fast it runs, how much GPU memory it uses, and — uniquely — how well the network's own output actually satisfies the physics equation it's supposed to be solving, which is a sanity check specific to physics-informed models.

### The honest part — say this proactively, don't wait to be asked

Two things we're deliberately not claiming:

**We are not claiming this reduces recurrence rates.** That number can only come from a real clinical trial — treating actual patients this way and following their outcomes for years. What the research literature actually shows is more nuanced than people expect: most glioblastoma recurrences happen centrally, at the original tumor site, regardless of margin size — several real studies found expanding margins doesn't meaningfully change recurrence patterns. What larger margins *do* reliably cause is more radiation to healthy brain tissue, including risk to memory-related structures like the hippocampus. So our honest, defensible claim is: **for equivalent tumor coverage, we can meaningfully reduce how much healthy brain tissue gets irradiated** — which the literature says matters, even though we can't yet say it changes survival.

**This is a research prototype, not a validated clinical tool.** Every report and screen says so. Getting from here to an actual clinical decision-support system means multi-center validation on far more patients, then a prospective trial. We're not pretending otherwise — and being upfront about that is, if anything, what should make you trust the parts we *can* back up.

---

### Quick-reference cheat sheet for Q&A curveballs

- *"Why not just use a bigger margin for everyone?"* → Literature shows bigger margins don't meaningfully reduce recurrence, but do increase toxicity (hippocampus example) — so "just expand it" isn't actually a free win.
- *"How is this different from GliODIL?"* → They moved away from PINNs due to speed and per-patient retraining; we built specifically to fix that with an amortized, shared-weight model plus calibrated uncertainty.
- *"Is 0.35 a real clinical number?"* → No — we don't use a fixed threshold at all anymore; it's computed per patient via the optimization sweep.
- *"How many patients did you validate on?"* → Be honest about your actual N. Small-N results are labeled as preliminary everywhere in the report, on purpose.
