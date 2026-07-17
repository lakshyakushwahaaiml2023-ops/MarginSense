"""
MarginSense Upload Preprocessing Pipeline
==========================================
Accepts raw NIfTI files (T1, T1ce, T2, FLAIR) for a new patient,
runs bias-field correction, resampling, automatic GTV approximation,
tissue segmentation, and saves a patient-ready NPZ bundle.

Tumor segmentation: T1ce/FLAIR intensity threshold + morphological cleanup.
This is an approximation - NOT a validated clinical segmentation.

No SimpleITK, ANTs, or FSL required. Uses: nibabel, scipy, numpy.
"""

import os
import json
import time
import threading
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    zoom, gaussian_filter, binary_fill_holes,
    binary_erosion, binary_dilation, label as sc_label,
)


class PreprocessingError(Exception):
    def __init__(self, step, message):
        self.step = step
        self.message = message
        super().__init__("[Error][{}] {}".format(step, message))


_progress_logs = {}
_progress_done = {}
_progress_error = {}
_progress_lock = threading.Lock()


def _log(patient_id, msg, log_fn=None):
    with _progress_lock:
        _progress_logs.setdefault(patient_id, []).append(msg)
    if log_fn:
        log_fn(msg)
    print(msg, flush=True)


def get_progress(patient_id):
    with _progress_lock:
        logs  = list(_progress_logs.get(patient_id, []))
        done  = _progress_done.get(patient_id, False)
        error = _progress_error.get(patient_id, None)
    return logs, done, error


def _mark_done(patient_id, error=None):
    with _progress_lock:
        _progress_done[patient_id]  = True
        _progress_error[patient_id] = error


def load_nifti(path, step_name="load_nifti"):
    if not os.path.exists(path):
        raise PreprocessingError(step_name, "File not found: " + path)
    try:
        img   = nib.load(path)
        data  = img.get_fdata(dtype=np.float32)
        zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(step_name,
            "Cannot read {} -- {}".format(os.path.basename(path), exc))
    if data.ndim != 3:
        raise PreprocessingError(step_name,
            "Expected 3D volume, got shape {} in {}. "
            "For 4D volumes extract the correct frame first.".format(
                data.shape, os.path.basename(path)))
    if data.max() == 0:
        raise PreprocessingError(step_name,
            "{} is entirely zero -- corrupt or wrong scan?".format(
                os.path.basename(path)))
    return data, img.affine, zooms


def correct_bias_field(volume, mask=None, step_name="bias_field_correction"):
    """Simplified multiplicative bias field correction in log-domain.
    Estimates bias as heavily-smoothed log-intensities (N4 approximation).
    No SimpleITK or ANTs required.
    """
    try:
        eps = 1e-6
        vol = volume.astype(np.float64)
        if mask is None:
            pos = vol > 0
            mask = (vol > np.percentile(vol[pos], 5)) if np.any(pos) else np.ones(vol.shape, dtype=bool)
        mask = mask.astype(bool)
        if not np.any(mask):
            return volume.copy().astype(np.float32)
        log_vol    = np.where(mask, np.log(vol + eps), 0.0)
        sigma      = [max(s // 6, 4) for s in vol.shape]
        smooth_num = gaussian_filter(log_vol * mask.astype(np.float64), sigma=sigma)
        smooth_den = gaussian_filter(mask.astype(np.float64), sigma=sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            bias = np.where(smooth_den > 1e-8, smooth_num / smooth_den, 0.0)
        corrected = np.exp(log_vol - bias) - eps
        corrected = np.clip(corrected, 0.0, None)
        corrected[~mask] = 0.0
        return corrected.astype(np.float32)
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(step_name, "Bias correction failed -- {}".format(exc))


def resample_to_128(volume, orig_shape, order=1, step_name="resample"):
    """Resample to (128,128,128). order=1 for images, order=0 for masks."""
    try:
        factors   = [128.0 / s for s in orig_shape[:3]]
        resampled = zoom(
            volume[:orig_shape[0], :orig_shape[1], :orig_shape[2]],
            factors, order=order, mode="nearest"
        )
        return resampled.astype(np.float32)
    except Exception as exc:
        raise PreprocessingError(step_name, "Resampling failed -- {}".format(exc))


def normalize_volume(volume):
    """Z-score normalization over non-zero (brain) voxels."""
    mask = volume > 0
    if not np.any(mask):
        return volume.astype(np.float32)
    mean = float(volume[mask].mean())
    std  = float(volume[mask].std())
    out  = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / (std + 1e-8)
    return out


def segment_tumor_auto(t1_n, t1ce_n, t2_n, flair_n,
                        step_name="tumor_segmentation"):
    """Approximate GTV using T1ce contrast enhancement + FLAIR hyperintensity.

    GBM signature exploited
    -----------------------
    - Gadolinium-enhancing core: T1ce > T1 background (contrast uptake).
    - Peri-tumoral edema / non-enhancing tumor: FLAIR hyperintensity.

    Pipeline: threshold -> union -> fill holes -> morph open -> largest CC.

    Returns (label_128^3_int8, confidence_float).
    """
    try:
        brain_mask = (np.abs(t1ce_n) > 0.05) | (np.abs(flair_n) > 0.05)
        if not np.any(brain_mask):
            return np.zeros((128, 128, 128), dtype=np.int8), 0.0
        enhancement = t1ce_n - t1_n
        enh_thresh  = float(np.percentile(enhancement[brain_mask], 82))
        fl_thresh   = float(np.percentile(flair_n[brain_mask],     88))
        core_mask   = brain_mask & (enhancement > enh_thresh) & (enhancement > 0.08)
        flair_mask  = brain_mask & (flair_n > fl_thresh)
        combined    = core_mask | flair_mask
        if not np.any(combined):
            combined = brain_mask & (
                flair_n > float(np.percentile(flair_n[brain_mask], 92))
            )
        if not np.any(combined):
            return np.zeros((128, 128, 128), dtype=np.int8), 0.0
        filled  = binary_fill_holes(combined)
        struct  = np.ones((3, 3, 3), dtype=bool)
        eroded  = binary_erosion(filled,  structure=struct, iterations=2)
        dilated = binary_dilation(eroded, structure=struct, iterations=2)
        if not np.any(dilated):
            dilated = filled
        labeled_arr, n_comp = sc_label(dilated)
        if n_comp == 0:
            return np.zeros((128, 128, 128), dtype=np.int8), 0.0
        sizes   = [int((labeled_arr == i).sum()) for i in range(1, n_comp + 1)]
        largest = int(np.argmax(sizes)) + 1
        tumor   = (labeled_arr == largest).astype(np.int8)
        n_enh   = int(np.sum(core_mask & (tumor > 0)))
        conf    = min(float(n_enh) / max(float(tumor.sum()), 1.0), 1.0)
        return tumor, conf
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(step_name,
            "Tumor segmentation failed -- {}".format(exc))


def estimate_tissue_types(image, label):
    """WM/GM/CSF from T1 intensity. Identical to preprocess.py for NPZ compat.
    Returns int8: 0=Background/Tumor  1=WM  2=GM  3=CSF
    """
    t1, flair  = image[0], image[3]
    brain_mask = (flair > 0) & (label == 0)
    tissue_map = np.zeros_like(label, dtype=np.int8)
    if not np.any(brain_mask):
        return tissue_map
    normal_t1 = t1[brain_mask]
    if len(normal_t1) < 100:
        return tissue_map
    p33, p66 = np.percentile(normal_t1, [33, 66])
    tissue_map[brain_mask & (t1 > p66)]                 = 1  # White Matter
    tissue_map[brain_mask & (t1 >= p33) & (t1 <= p66)]  = 2  # Gray Matter
    tissue_map[brain_mask & (t1 < p33)]                  = 3  # CSF
    return tissue_map


def default_covariates():
    """Literature-default clinical covariates for GBM (Stupp et al. 2005, EANO 2021).
    Replace with actual patient values before clinical use.
    """
    return {
        "age":              58,
        "kps":              80,
        "idh_status":       "Wild-type",
        "mgmt_status":      "Unmethylated",
        "resection_extent": "GTR",
        "laterality":       "Left",
        "notes": (
            "Defaults from GBM population literature (Stupp et al. 2005). "
            "Replace with actual patient values before clinical interpretation."
        ),
    }


def run_pipeline(patient_id, t1_path, t1ce_path, t2_path, flair_path,
                 covariates=None, log_fn=None):
    """Full preprocessing pipeline for a new patient NIfTI upload.

    Outputs
    -------
    data/processed/{patient_id}.npz              NPZ bundle (pipeline-ready)
    data/processed/{patient_id}_covariates.json  Clinical covariate sidecar

    Raises PreprocessingError on any step failure.
    """
    def log(msg):
        _log(patient_id, msg, log_fn)

    with _progress_lock:
        _progress_logs[patient_id]  = []
        _progress_done[patient_id]  = False
        _progress_error[patient_id] = None

    t_total = time.perf_counter()

    try:
        # 1. Load
        log("[1/7] Loading NIfTI modalities for: " + patient_id)
        t1_vol,   _, t1_zooms = load_nifti(t1_path,    "load_t1")
        log("      T1   shape={}  spacing={} mm".format(t1_vol.shape, t1_zooms.tolist()))
        t1ce_vol, _, _        = load_nifti(t1ce_path,  "load_t1ce")
        log("      T1ce shape={}".format(t1ce_vol.shape))
        t2_vol,   _, _        = load_nifti(t2_path,    "load_t2")
        log("      T2   shape={}".format(t2_vol.shape))
        fl_vol,   _, _        = load_nifti(flair_path, "load_flair")
        log("      FLAIR shape={}".format(fl_vol.shape))
        orig_shape = tuple(t1_vol.shape[:3])

        # 2. Resampled spacing
        resampled_spacing = np.array(
            [t1_zooms[i] * orig_shape[i] / 128.0 for i in range(3)],
            dtype=np.float32,
        )
        log("[2/7] Resampled voxel spacing: {} mm".format(resampled_spacing.tolist()))

        # 3. Bias field correction
        log("[3/7] Bias field correction (log-Gaussian N4 approximation)...")
        pos       = t1_vol > 0
        brain_raw = (t1_vol > np.percentile(t1_vol[pos], 5)) if np.any(pos) else np.ones(orig_shape, dtype=bool)
        t1_c   = correct_bias_field(t1_vol,   mask=brain_raw, step_name="bias_t1")
        t1ce_c = correct_bias_field(t1ce_vol, mask=brain_raw, step_name="bias_t1ce")
        t2_c   = correct_bias_field(t2_vol,   mask=brain_raw, step_name="bias_t2")
        fl_c   = correct_bias_field(fl_vol,   mask=brain_raw, step_name="bias_flair")
        log("      Bias correction complete (T1, T1ce, T2, FLAIR).")

        # 4. Resample
        log("[4/7] Resampling all modalities to 128x128x128...")
        t1_r   = resample_to_128(t1_c,   orig_shape, order=1, step_name="resample_t1")
        t1ce_r = resample_to_128(t1ce_c, orig_shape, order=1, step_name="resample_t1ce")
        t2_r   = resample_to_128(t2_c,   orig_shape, order=1, step_name="resample_t2")
        fl_r   = resample_to_128(fl_c,   orig_shape, order=1, step_name="resample_flair")
        log("      Resampled to (128, 128, 128).")

        # 5. Normalize
        log("[5/7] Z-score normalizing per modality...")
        t1_n   = normalize_volume(t1_r)
        t1ce_n = normalize_volume(t1ce_r)
        t2_n   = normalize_volume(t2_r)
        fl_n   = normalize_volume(fl_r)
        image  = np.stack([t1_n, t1ce_n, t2_n, fl_n], axis=0)
        log("      Image stack: shape={}  dtype={}".format(image.shape, image.dtype))

        # 6. Tumor segmentation + tissue map
        log("[6/7] Automatic GTV approximation (T1ce/FLAIR threshold)...")
        log("      [NOTICE] Intensity-threshold approximation.")
        log("               Not a validated clinical segmentation.")
        log("               Review GTV overlay in dashboard before use.")
        label, conf = segment_tumor_auto(t1_n, t1ce_n, t2_n, fl_n)
        n_tumor = int(np.sum(label > 0))
        log("      GTV voxels={}  enhancement confidence={:.2f}".format(n_tumor, conf))
        if n_tumor == 0:
            log("      [Warning] No tumor detected. Verify T1ce is post-contrast.")
        tissue_map = estimate_tissue_types(image, label)
        log("      Tissue -- WM={}  GM={}  CSF={}".format(
            np.sum(tissue_map == 1), np.sum(tissue_map == 2), np.sum(tissue_map == 3)))
        recurrence = (label > 0).astype(np.int8)

        # 7. Derived features sidecar
        log("[7/8] Computing derived imaging features...")
        from src.compute_features import compute_all_imaging_features
        try:
            derived_feats = compute_all_imaging_features(label, resampled_spacing, tissue_map)
            derived_file = "data/processed/{}_derived_features.json".format(patient_id)
            with open(derived_file, "w") as fh:
                json.dump(derived_feats, fh, indent=4)
            log("      Derived features saved: " + derived_file)
        except Exception as e:
            log("      [Warning] Derived features extraction failed: {}".format(e))

        # 8. Save NPZ + covariates
        log("[8/8] Writing patient-ready NPZ bundle and covariates...")
        os.makedirs("data/processed", exist_ok=True)
        npz_path = "data/processed/{}.npz".format(patient_id)
        np.savez_compressed(
            npz_path,
            image=image, label=label, spacing=resampled_spacing,
            recurrence=recurrence, tissue_map=tissue_map,
        )
        log("      NPZ saved: " + npz_path)

        cov = default_covariates()
        if covariates:
            cov.update({k: v for k, v in covariates.items()
                        if v not in (None, "", "null", "undefined")})
        cov_path = "data/processed/{}_covariates.json".format(patient_id)
        with open(cov_path, "w") as fh:
            json.dump(cov, fh, indent=2)
        log("      Covariates saved: " + cov_path)

        elapsed = time.perf_counter() - t_total
        log("\n[Done] Preprocessing completed in {:.1f} s.".format(elapsed))
        log("[Ready] Patient '{}' now available in the patient selector.".format(patient_id))
        _mark_done(patient_id)

    except PreprocessingError as exc:
        log("\n" + str(exc))
        log("[Stopped] Halted at step: " + exc.step)
        log("[Action]  Correct the issue and re-upload.")
        _mark_done(patient_id, error=exc.message)
        raise

    except Exception as exc:
        msg = "Unexpected error: {}".format(exc)
        log("\n[Error][unexpected] " + msg)
        log("[Stopped] Preprocessing failed unexpectedly.")
        _mark_done(patient_id, error=msg)
        raise PreprocessingError("unexpected", msg)
