import os
import sys
# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import csv
import subprocess
import numpy as np
import threading
import datetime
import time
from flask import Flask, jsonify, request, send_from_directory, Response
from skimage.measure import find_contours
from werkzeug.utils import secure_filename
from src.preprocess_upload import run_pipeline, get_progress

app = Flask(__name__, static_folder=None)

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))

live_logs = []
is_running = False
live_logs_lock = threading.Lock()

def log_message(msg):
    global live_logs
    with live_logs_lock:
        live_logs.append(msg + "\n")
        print(msg)  # also print to server console

def run_pipeline_thread(patient_id):
    global is_running, live_logs
    is_running = True
    
    with live_logs_lock:
        live_logs.clear()
        
    is_synthetic = patient_id.startswith("synthetic_")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    log_message(f"[*] Starting live radiotherapy planning simulation for patient: {patient_id}")
    
    try:
        # 1. Uniform Margin Baseline
        log_message("[*] Running Uniform Margin Dilation (clinical standard)...")
        cmd_uniform = [sys.executable, "src/baseline_uniform_margin.py", patient_id]
        if is_synthetic and not os.path.exists(f"data/processed/{patient_id}.npz"):
            cmd_uniform.append("--synthetic")
        res = subprocess.run(cmd_uniform, env=env, capture_output=True, text=True)
        log_message(res.stdout + res.stderr)
        if res.returncode != 0:
            log_message(f"[Error] Uniform baseline execution failed with exit code {res.returncode}")
            return

        # 2. Vanilla per-patient PINN
        log_message("\n[*] Running traditional per-patient PINN optimization (100 epochs)...")
        cmd_pinn = [sys.executable, "src/baseline_vanilla_pinn.py", patient_id, "--epochs", "100"]
        if is_synthetic and not os.path.exists(f"data/processed/{patient_id}.npz"):
            cmd_pinn.append("--synthetic")
        res = subprocess.run(cmd_pinn, env=env, capture_output=True, text=True)
        log_message(res.stdout + res.stderr)
        if res.returncode != 0:
            log_message(f"[Error] Vanilla PINN execution failed with exit code {res.returncode}")
            return
            
        # 3. MarginSense Ensemble Inference
        log_message("\n[*] Running MarginSense Ensemble Amortized Inference (5 models)...")
        cmd_ms = [sys.executable, "src/evaluate_uncertainty.py", patient_id]
        res = subprocess.run(cmd_ms, env=env, capture_output=True, text=True)
        log_message(res.stdout + res.stderr)
        if res.returncode != 0:
            log_message(f"[Error] MarginSense Inference execution failed with exit code {res.returncode}")
            return
            
        # 4. Compare Models report compiler
        log_message("\n[*] Evaluating metrics (spatial coverage and healthy tissue sparing)...")
        cmd_compare = [sys.executable, "src/compare_models.py", patient_id, "--threshold", "0.35"]
        res = subprocess.run(cmd_compare, env=env, capture_output=True, text=True)
        log_message(res.stdout + res.stderr)
        if res.returncode != 0:
            log_message(f"[Error] Metrics evaluation failed with exit code {res.returncode}")
            return

        # 5. Rebuild GLB meshes and volume textures
        log_message("\n[*] Rebuilding surface meshes and 3D volumetric textures...")
        cmd_export = [sys.executable, "src/viz/export_meshes.py", patient_id]
        res = subprocess.run(cmd_export, env=env, capture_output=True, text=True)
        log_message(res.stdout + res.stderr)
        
        log_message("\n[Success] Live planning simulation completed successfully! Reloading UI components.")
    except Exception as e:
        log_message(f"\n[CRITICAL ERROR] Pipeline execution exception: {str(e)}")
    finally:
        is_running = False

@app.route("/")
def index():
    return send_from_directory(VIZ_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(VIZ_DIR, path)

@app.route("/api/patients")
def list_patients():
    """Lists available patient folders, preprocessed files, or fallbacks."""
    raw_dir = "data/raw"
    proc_dir = "data/processed"
    patients = set()
    
    # 1. Scan raw directories
    if os.path.exists(raw_dir):
        for d in os.listdir(raw_dir):
            if os.path.isdir(os.path.join(raw_dir, d)):
                patients.add(d)
                
    # 2. Scan preprocessed files (ends with .npz)
    if os.path.exists(proc_dir):
        for f in os.listdir(proc_dir):
            if f.endswith(".npz"):
                patients.add(f[:-4])
                
    # Ensure synthetic options are always available as fallback
    patients.add("synthetic_patient_1")
    patients.add("synthetic_patient_2")
    patients.add("synthetic_patient_3")
    
    return jsonify(sorted(list(patients)))

@app.route("/api/upload_patient", methods=["POST"])
def upload_patient():
    """Ingests, validates, and runs preprocessing on a new patient NIfTI upload."""
    patient_id = request.form.get("patient_id", "").strip()
    if not patient_id:
        f = request.files.get("t1")
        if f and f.filename:
            patient_id = secure_filename(f.filename).split(".")[0].replace("_t1", "")
        else:
            patient_id = f"patient_{int(time.time())}"
            
    patient_id = secure_filename(patient_id)
    
    # Collision check: auto-suffix timestamp if patient ID already exists
    npz_path = f"data/processed/{patient_id}.npz"
    if os.path.exists(npz_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_id = f"{patient_id}_{timestamp}"
        
    upload_dir = f"data/uploads/{patient_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    paths = {}
    for key in ["t1", "t1ce", "t2", "flair"]:
        file_obj = request.files.get(key)
        if not file_obj or not file_obj.filename:
            return jsonify({"error": f"Missing NIfTI file for modality: {key}"}), 400
            
        filename = secure_filename(file_obj.filename)
        if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
            return jsonify({"error": f"Invalid file format for {key}: must be .nii or .nii.gz"}), 400
            
        dest_path = os.path.join(upload_dir, filename)
        file_obj.save(dest_path)
        paths[key] = dest_path
        
    # Extract covariates
    covariates = {
        "age": request.form.get("age"),
        "kps": request.form.get("kps"),
        "idh_status": request.form.get("idh_status"),
        "mgmt_status": request.form.get("mgmt_status"),
        "resection_extent": request.form.get("resection_extent"),
        "laterality": request.form.get("laterality"),
    }
    
    def bg_preprocess():
        try:
            run_pipeline(
                patient_id=patient_id,
                t1_path=paths["t1"],
                t1ce_path=paths["t1ce"],
                t2_path=paths["t2"],
                flair_path=paths["flair"],
                covariates=covariates
            )
            # Create a raw folder in data/raw to register the patient ID if not present
            raw_patient_dir = f"data/raw/{patient_id}"
            os.makedirs(raw_patient_dir, exist_ok=True)
        except Exception as e:
            print(f"[Error] Background preprocessing failed for {patient_id}: {e}")
        finally:
            # Clean up upload directory to free disk space
            try:
                for p in paths.values():
                    if os.path.exists(p):
                        os.remove(p)
                if os.path.exists(upload_dir) and not os.listdir(upload_dir):
                    os.rmdir(upload_dir)
            except Exception:
                pass
                
    threading.Thread(target=bg_preprocess, daemon=True).start()
    return jsonify({"status": "started", "patient_id": patient_id})

@app.route("/api/upload_progress/<patient_id>")
def upload_progress(patient_id):
    """Streams SSE logs from the background preprocessing thread."""
    def generate():
        last_idx = 0
        while True:
            logs, done, error = get_progress(patient_id)
            if last_idx < len(logs):
                for i in range(last_idx, len(logs)):
                    yield f"data: {json.dumps({'log': logs[i]})}\n\n"
                last_idx = len(logs)
            if done:
                if error:
                    yield f"data: {json.dumps({'event': 'error', 'message': error})}\n\n"
                else:
                    yield f"data: {json.dumps({'event': 'done'})}\n\n"
                break
            time.sleep(0.3)
    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/patient_covariates/<patient_id>")
def get_patient_covariates(patient_id):
    """Loads and serves clinical covariates and auto-derived features for the patient."""
    from src.preprocess_upload import default_covariates
    cov = default_covariates()
    cov_path = f"data/processed/{patient_id}_covariates.json"
    if os.path.exists(cov_path):
        try:
            with open(cov_path, "r") as f:
                cov.update(json.load(f))
        except Exception:
            pass

    derived = {}
    derived_path = f"data/processed/{patient_id}_derived_features.json"
    if os.path.exists(derived_path):
        try:
            with open(derived_path, "r") as f:
                derived = json.load(f)
        except Exception:
            pass
    else:
        # Check if NPZ exists to compute derived features dynamically
        npz_path = f"data/processed/{patient_id}.npz"
        if os.path.exists(npz_path):
            try:
                npz_file = np.load(npz_path)
                from src.compute_features import compute_all_imaging_features
                lbl = npz_file['label']
                spc = npz_file['spacing']
                tmap = npz_file['tissue_map']
                derived = compute_all_imaging_features(lbl, spc, tmap)
            except Exception:
                pass

    # Return merged dict with metadata info
    response_data = {}
    # Manual features
    for k, v in cov.items():
        if k != "notes":
            response_data[k] = {
                "value": v,
                "is_auto_computed": False
            }
            
    # Auto-computed features
    for k, v in derived.items():
        response_data[k] = {
            "value": v,
            "is_auto_computed": True
        }
        
    return jsonify(response_data)

@app.route("/api/gpu_status")
def gpu_status():
    """Queries GPU status: nvidia-smi utilization."""
    status = {"utilization": 0, "memory_used": 0, "memory_total": 6141, "torch_allocated": 0}
    try:
        # Query nvidia-smi
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            parts = res.stdout.strip().split(",")
            status["utilization"] = float(parts[0])
            status["memory_used"] = float(parts[1])
            status["memory_total"] = float(parts[2])
    except Exception as e:
        pass
    return jsonify(status)

@app.route("/api/start_pipeline", methods=["POST"])
def start_pipeline():
    """Triggers the pipeline runner in a background thread."""
    global is_running
    if is_running:
        return jsonify({"status": "error", "message": "Pipeline is already running."}), 400
        
    data = request.json or {}
    patient_id = data.get("patient_id", "synthetic_patient_2")
    
    threading.Thread(target=run_pipeline_thread, args=(patient_id,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/live_logs")
def get_live_logs():
    """Retrieves current in-progress log buffer and active state."""
    global live_logs, is_running
    with live_logs_lock:
        logs_str = "".join(live_logs)
    return jsonify({"logs": logs_str, "is_running": is_running})

@app.route("/api/patient_data/<patient_id>")
def get_patient_data(patient_id):
    """Serves cached per-patient volume data. Runs exporter only if cache is missing."""
    try:
        # Check per-patient JSON cache first (fast path)
        model_dir = os.path.join(os.getcwd(), "src", "viz", "models")
        cache_path = os.path.join(model_dir, f"data_{patient_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return jsonify(json.load(f))
        
        # Cache miss — run exporter (slow path, only first time)
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        cmd = [sys.executable, "src/viz/export_meshes.py", patient_id]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            return jsonify({"error": f"Failed to export patient data: {res.stderr}"}), 500
            
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return jsonify(json.load(f))
                
        return jsonify({"error": "Data file not generated"}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/models/<patient_id>/<model_type>")
def serve_glb(patient_id, model_type):
    """Serves GLB mesh files for brain, tumor, margin, and predicted_margin."""
    allowed = {"brain", "tumor", "margin", "predicted_margin"}
    if model_type not in allowed:
        return jsonify({"error": "Unknown model type"}), 400
    model_dir = os.path.join(os.getcwd(), "src", "viz", "models")
    if model_type == "brain":
        filename = "brain_template.glb"
    else:
        filename = f"{model_type}_{patient_id}.glb"
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": f"Model {filename} not found"}), 404
    from flask import send_file
    return send_file(filepath, mimetype="model/gltf-binary")

@app.route("/api/comparison/<patient_id>")
def get_comparison(patient_id):
    """Loads and serves the computed comparative JSON report."""
    report_path = f"outputs/{patient_id}_comparison_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Report not found. Please click 'Run Radiotherapy Planning' first."}), 404


@app.route("/api/generate_report/<patient_id>")
def generate_report(patient_id):
    """
    Assembles a structured patient report entirely from computed pipeline outputs.
    """
    optimal_threshold = float(request.args.get('optimal_threshold', 0.35))
    lambda_val = float(request.args.get('lambda_val', 50.0))
    z_val = float(request.args.get('z_val', 1.0))
    report_path = f"outputs/{patient_id}_comparison_report.json"
    if not os.path.exists(report_path):
        return jsonify({"error": "Pipeline outputs not found. Run 'Run Target Planning' first."}), 404

    with open(report_path, "r") as f:
        comp = json.load(f)

    # ── Covariates: detect which fields used literature defaults ─────────────
    from src.preprocess_upload import default_covariates
    defaults = default_covariates()
    cov_path = f"data/processed/{patient_id}_covariates.json"
    raw_cov = defaults.copy()
    if os.path.exists(cov_path):
        with open(cov_path, "r") as f:
            raw_cov = json.load(f)

    cov_keys = ["age", "kps", "idh_status", "mgmt_status", "resection_extent", "laterality"]
    covariates_flagged = {}
    for key in cov_keys:
        val = raw_cov.get(key, defaults.get(key))
        is_default = str(val) == str(defaults.get(key))
        covariates_flagged[key] = {
            "value": val,
            "is_literature_default": is_default,
            "is_auto_computed": False
        }

    # Load derived features
    derived = {}
    derived_path = f"data/processed/{patient_id}_derived_features.json"
    if os.path.exists(derived_path):
        try:
            with open(derived_path, "r") as f:
                derived = json.load(f)
        except Exception:
            pass
    else:
        # fallback/dynamic computation
        npz_path = f"data/processed/{patient_id}.npz"
        if os.path.exists(npz_path):
            try:
                npz_file = np.load(npz_path)
                from src.compute_features import compute_all_imaging_features
                lbl = npz_file['label']
                spc = npz_file['spacing']
                tmap = npz_file['tissue_map']
                derived = compute_all_imaging_features(lbl, spc, tmap)
            except Exception:
                pass

    for key, val in derived.items():
        covariates_flagged[key] = {
            "value": val,
            "is_literature_default": False,
            "is_auto_computed": True
        }

    # ── Extract per-method metrics ───────────────────────────────────────────
    results = comp.get("results", {})
    ms_key  = next((k for k in results if "MarginSense" in k), None)
    std_key = next((k for k in results if "Clinical Standard" in k or "Uniform" in k), None)
    ms  = results.get(ms_key,  {}) if ms_key  else {}
    std = results.get(std_key, {}) if std_key else {}
    threshold = comp.get("threshold", 0.35)

    # Calculate optimized MarginSense metrics on the fly based on the optimal threshold
    opt_coverage = ms.get("recurrence_coverage_percent", 0.0)
    opt_healthy = ms.get("treated_healthy_tissue_volume_cm3", 0.0)
    opt_vol = ms.get("total_target_volume_cm3", 0.0)
    
    safety_coverage = ms.get("recurrence_coverage_percent", 0.0)
    safety_healthy = ms.get("treated_healthy_tissue_volume_cm3", 0.0)
    safety_vol = ms.get("total_target_volume_cm3", 0.0)
    
    ensemble_path = f"outputs/{patient_id}_prediction_ensemble.npz"
    if os.path.exists(ensemble_path):
        try:
            ens = np.load(ensemble_path)
            mean_dens = ens["mean_density"]
            std_dens = ens["std_density"] if "std_density" in ens else np.zeros_like(mean_dens)
            
            # Load processed data
            npz_proc = f"data/processed/{patient_id}.npz"
            voxel_vol_cm3 = 1e-3
            proc_label = np.zeros_like(mean_dens)
            proc_recurrence = np.zeros_like(mean_dens)
            proc_tmap = np.zeros_like(mean_dens)
            if os.path.exists(npz_proc):
                proc = np.load(npz_proc)
                if "spacing" in proc:
                    voxel_vol_cm3 = float(np.prod(proc["spacing"])) / 1000.0
                if "label" in proc:
                    proc_label = proc["label"]
                if "recurrence" in proc:
                    proc_recurrence = proc["recurrence"]
                if "tissue_map" in proc:
                    proc_tmap = proc["tissue_map"]
                    
            opt_mask = (mean_dens >= optimal_threshold) & (proc_tmap != 3)
            
            total_rec = np.sum(proc_recurrence > 0)
            if total_rec > 0:
                opt_rec_in = np.sum((opt_mask > 0) & (proc_recurrence > 0))
                opt_coverage = float(opt_rec_in / total_rec * 100.0)
            
            opt_healthy = float(np.sum((opt_mask > 0) & (proc_label == 0)) * voxel_vol_cm3)
            opt_vol = float(np.sum(opt_mask > 0) * voxel_vol_cm3)
            
            # Recommended Safety Margin (UCB) computation:
            safety_vol_map = mean_dens + z_val * std_dens
            safety_mask = (safety_vol_map >= optimal_threshold) & (proc_tmap != 3)
            
            if total_rec > 0:
                safety_rec_in = np.sum((safety_mask > 0) & (proc_recurrence > 0))
                safety_coverage = float(safety_rec_in / total_rec * 100.0)
                
            safety_healthy = float(np.sum((safety_mask > 0) & (proc_label == 0)) * voxel_vol_cm3)
            safety_vol = float(np.sum(safety_mask > 0) * voxel_vol_cm3)
        except Exception as e:
            print(f"[Warn] Dynamic metric calculation failed: {e}")

    # ── Uncertainty metrics from ensemble NPZ ────────────────────────────────
    uncertainty_data = {
        "high_conf_pct": None,
        "mean_std_boundary": None,
        "max_std": None,
        "max_uncertainty_region": "boundary region (insufficient data)",
        "uncertainty_threshold": 0.05
    }

    ensemble_path = f"outputs/{patient_id}_prediction_ensemble.npz"
    if os.path.exists(ensemble_path):
        try:
            ens = np.load(ensemble_path)
            mean_dens = ens["mean_density"]   # (128,128,128)
            std_dens  = ens["std_density"]    # (128,128,128)
            D, H, W   = mean_dens.shape

            # Boundary band: ±0.1 around the density threshold
            boundary_mask = (mean_dens >= threshold - 0.1) & (mean_dens <= threshold + 0.1)
            conf_thresh = 0.05  # std < 0.05 → "high confidence"

            if np.any(boundary_mask):
                bdy_std = std_dens[boundary_mask]
                high_conf_pct   = float(np.mean(bdy_std < conf_thresh) * 100.0)
                mean_std_bdry   = float(np.mean(bdy_std))
                max_std_val     = float(np.max(std_dens))

                # Anatomical region of highest uncertainty on the boundary
                pct90 = np.percentile(bdy_std, 90) if len(bdy_std) > 10 else bdy_std.max()
                high_unc_mask   = boundary_mask & (std_dens >= pct90)

                region_label = "boundary periphery"
                if np.any(high_unc_mask):
                    coords   = np.array(np.where(high_unc_mask)).T.astype(float)  # (N,3)
                    centroid = coords.mean(axis=0)                                  # z, y, x
                    devs = [abs(centroid[0] - D/2), abs(centroid[1] - H/2), abs(centroid[2] - W/2)]
                    dom  = int(np.argmax(devs))
                    if dom == 0:
                        region_label = "superior margin" if centroid[0] > D/2 else "inferior margin"
                    elif dom == 1:
                        region_label = "anterior margin" if centroid[1] > H/2 else "posterior margin"
                    else:
                        region_label = "left hemisphere margin" if centroid[2] > W/2 else "right hemisphere margin"

                uncertainty_data = {
                    "high_conf_pct":          round(high_conf_pct, 1),
                    "mean_std_boundary":      round(mean_std_bdry, 4),
                    "max_std":                round(max_std_val, 4),
                    "max_uncertainty_region": region_label,
                    "uncertainty_threshold":  conf_thresh,
                }
        except Exception as e:
            uncertainty_data["parse_error"] = str(e)

    # ── Boundary deviation analysis (MarginSense vs. standard) ──────────────
    discussion_flags = []
    uniform_path = f"outputs/{patient_id}_baseline_uniform.npz"

    if os.path.exists(uniform_path) and os.path.exists(ensemble_path):
        try:
            uni_data  = np.load(uniform_path)
            uni_mask  = uni_data["dilated_mask"].astype(bool)
            ens_data  = np.load(ensemble_path)
            mean_dens = ens_data["mean_density"]
            ens_mask  = (mean_dens >= threshold)

            D, H, W = uni_mask.shape

            def _region_label(mask):
                if not np.any(mask):
                    return "periphery"
                coords   = np.array(np.where(mask)).T.astype(float)
                centroid = coords.mean(axis=0)
                devs = [abs(centroid[0]-D/2), abs(centroid[1]-H/2), abs(centroid[2]-W/2)]
                dom  = int(np.argmax(devs))
                if dom == 0:
                    return "superior region" if centroid[0] > D/2 else "inferior region"
                elif dom == 1:
                    return "anterior region" if centroid[1] > H/2 else "posterior region"
                else:
                    return "left hemisphere" if centroid[2] > W/2 else "right hemisphere"

            # Try to read voxel spacing from processed NPZ for accurate volume
            voxel_vol = 1e-3  # default: 1mm³ → cm³
            npz_proc = f"data/processed/{patient_id}.npz"
            if os.path.exists(npz_proc):
                try:
                    proc = np.load(npz_proc)
                    if "spacing" in proc:
                        voxel_vol = float(np.prod(proc["spacing"])) / 1000.0
                except Exception:
                    pass

            ms_only  = ens_mask & ~uni_mask
            std_only = uni_mask & ~ens_mask

            ms_ext_vol  = float(np.sum(ms_only))  * voxel_vol
            std_ext_vol = float(np.sum(std_only)) * voxel_vol

            if ms_ext_vol > 0.3:
                region = _region_label(ms_only)
                discussion_flags.append({
                    "type":       "ms_extends_beyond_standard",
                    "volume_cm3": round(ms_ext_vol, 2),
                    "region":     region,
                    "note": (
                        f"MarginSense extends approximately {ms_ext_vol:.1f} cm\u00b3 beyond the "
                        f"standard 1.5 cm uniform margin, concentrated in the {region}. "
                        f"The model predicts infiltration density above threshold in this zone "
                        f"that the fixed-margin approach does not cover. "
                        f"Warrants independent radiologist review before any use in planning."
                    )
                })
            if std_ext_vol > 0.3:
                region = _region_label(std_only)
                discussion_flags.append({
                    "type":       "standard_extends_beyond_ms",
                    "volume_cm3": round(std_ext_vol, 2),
                    "region":     region,
                    "note": (
                        f"The standard margin covers approximately {std_ext_vol:.1f} cm\u00b3 that "
                        f"MarginSense does not flag as high-density infiltration, concentrated in "
                        f"the {region}. This zone may represent lower predicted infiltration risk "
                        f"according to the model, but clinical judgment is required."
                    )
                })
        except Exception as e:
            discussion_flags.append({"type": "deviation_analysis_error", "note": str(e)})

    # ── Data quality caveats ─────────────────────────────────────────────────
    label_map = {
        "age": "Patient age", "kps": "KPS score",
        "idh_status": "IDH mutation status", "mgmt_status": "MGMT methylation status",
        "resection_extent": "Extent of resection", "laterality": "Tumor laterality",
        "tumor_volume_cm3": "Tumor Volume (cm³)", "hemisphere": "Hemisphere",
        "tumor_location": "Tumor Location (Lobe)", "ventricle_dist_mm": "Distance from Ventricles (mm)",
        "sphericity": "Sphericity"
    }
    quality_caveats = [
        "Infiltration probability is derived from ensemble agreement across 5 deep learning models; "
        "it represents a model-confidence estimate, not a clinically calibrated probability validated against outcome data."
    ]
    for key, info in covariates_flagged.items():
        if info["is_literature_default"]:
            quality_caveats.append(
                f"{label_map.get(key, key)} was not provided and used a literature default "
                f"value ({info['value']}). Replace with actual patient data before clinical use."
            )

    # ── Assemble final report ────────────────────────────────────────────────
    report = {
        "generated_at":      datetime.datetime.now().isoformat(timespec="seconds"),
        "patient_id":        patient_id,
        "model_version":     "MarginSense Ensemble v1.0 (5-model amortized PINN)",
        "imaging_sequences": ["T1", "T1ce (Gadolinium)", "T2", "FLAIR"],
        "pipeline_threshold": threshold,
        "optimal_threshold":  optimal_threshold,
        "optimization_lambda": lambda_val,
        "optimization_objective": "Maximize Recurrence Coverage - λ * (Normalized Healthy Volume)",
        "safety_margin_z":    z_val,
        "safety_margin_rule": "adjusted_value = mean_prediction + z * uncertainty (Thresholded at optimal threshold)",
        "disclaimer": (
            "The tumor infiltration probability is derived from ensemble agreement across 5 "
            "amortized PINN models. This is a model-confidence estimate representing model consensus, "
            "not a clinically calibrated probability validated against outcome data."
        ),
        "covariates":        covariates_flagged,
        "simulation": {
            "marginsense": {
                "infiltration_boundary_volume_cm3":   opt_vol,
                "recurrence_coverage_percent":        opt_coverage,
                "treated_healthy_tissue_volume_cm3":  opt_healthy,
                "processing_time_seconds":            ms.get("processing_time_seconds"),
                "hardware":                           ms.get("hardware_device", "GPU"),
            },
            "safety_margin": {
                "infiltration_boundary_volume_cm3":   safety_vol,
                "recurrence_coverage_percent":        safety_coverage,
                "treated_healthy_tissue_volume_cm3":  safety_healthy,
                "processing_time_seconds":            ms.get("processing_time_seconds"),
                "hardware":                           ms.get("hardware_device", "GPU"),
            },
            "raw_035": {
                "infiltration_boundary_volume_cm3":   ms.get("total_target_volume_cm3"),
                "recurrence_coverage_percent":        ms.get("recurrence_coverage_percent"),
                "treated_healthy_tissue_volume_cm3":  ms.get("treated_healthy_tissue_volume_cm3"),
                "processing_time_seconds":            ms.get("processing_time_seconds"),
                "hardware":                           ms.get("hardware_device", "GPU"),
            },
            "standard_margin": {
                "infiltration_boundary_volume_cm3":   std.get("total_target_volume_cm3"),
                "recurrence_coverage_percent":        std.get("recurrence_coverage_percent"),
                "treated_healthy_tissue_volume_cm3":  std.get("treated_healthy_tissue_volume_cm3"),
                "processing_time_seconds":            std.get("processing_time_seconds"),
                "hardware":                           std.get("hardware_device", "CPU"),
            },
            "recurrence_volume_cm3": comp.get("recurrence_volume_cm3"),
            "uncertainty":           uncertainty_data,
        },
        "discussion_flags":  discussion_flags,
        "quality_caveats":   quality_caveats,
    }

    # Persist to disk for subsequent loads
    out_path = f"outputs/{patient_id}_patient_report.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    return jsonify(report)

@app.route("/api/training_curves")
def training_curves():
    """Parses and serves model training metrics from the CSV logs."""
    log_path = "outputs/train_logs.csv"
    epochs = []
    losses = []
    losses_data = []
    losses_pde = []
    
    if os.path.exists(log_path):
        try:
            with open(log_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    losses.append(float(row['loss']))
                    losses_data.append(float(row['loss_data']))
                    losses_pde.append(float(row['loss_pde']))
        except Exception as e:
            pass
            
    return jsonify({
        "epochs": epochs,
        "loss": losses,
        "loss_data": losses_data,
        "loss_pde": losses_pde
    })

# ── 2D Radiology Slice API ──────────────────────────────────────────────

def _load_patient_volumes(patient_id):
    """Load all volumes needed for slice rendering and metrics computation."""
    npz_path = f"data/processed/{patient_id}.npz"
    if not os.path.exists(npz_path):
        return None
    try:
        data    = np.load(npz_path)
        image   = data['image']    # (4, D, H, W)
        label   = data['label']    # (D, H, W) — tumor label
        spacing = data['spacing']  # (3,)
        recurrence = data['recurrence'] if 'recurrence' in data else np.zeros_like(label)
        tissue_map = data['tissue_map'] if 'tissue_map' in data else np.zeros_like(label)
    except Exception:
        return None

    uniform_mask = None
    uniform_path = f"outputs/{patient_id}_baseline_uniform.npz"
    if os.path.exists(uniform_path):
        try:
            uniform_mask = np.load(uniform_path)['dilated_mask'].astype(bool)
        except Exception:
            pass

    density = None
    std_density = None
    ensemble_path = f"outputs/{patient_id}_prediction_ensemble.npz"
    if os.path.exists(ensemble_path):
        try:
            ens_data = np.load(ensemble_path)
            density = ens_data['mean_density']
            std_density = ens_data['std_density'] if 'std_density' in ens_data else None
        except Exception:
            pass

    return {'image': image, 'label': label, 'spacing': spacing,
            'uniform_mask': uniform_mask, 'density': density,
            'std_density': std_density,
            'recurrence': recurrence, 'tissue_map': tissue_map}


@app.route('/api/slices/<patient_id>/info')
def slices_info(patient_id):
    """Returns volume shape, voxel spacing, and default best-slice indices."""
    vol = _load_patient_volumes(patient_id)
    if vol is None:
        return jsonify({'error': 'Patient data not found'}), 404
    label = vol['label']  # (D, H, W)
    D, H, W = label.shape

    def best_slice(ax):
        sizes = label.sum(axis=tuple(a for a in range(3) if a != ax))
        idx   = int(np.argmax(sizes))
        return idx if sizes[idx] > 0 else label.shape[ax] // 2

    return jsonify({
        'shape':     [D, H, W],
        'spacing':   vol['spacing'].tolist(),
        'default_z': best_slice(0),
        'default_y': best_slice(1),
        'default_x': best_slice(2),
    })


@app.route('/api/slices/<patient_id>/<int:axis>/<int:index>')
def get_slice_data(patient_id, axis, index):
    """
    Returns a grayscale 2D slice + tumor/margin/density contours.
    axis: 0=axial(z-plane), 1=coronal(y-plane), 2=sagittal(x-plane)
    ?modality=  0=T1  1=T1ce  2=FLAIR(default)  3=T2
    """
    modality = int(request.args.get('modality', 2))  # FLAIR by default
    vol = _load_patient_volumes(patient_id)
    if vol is None:
        return jsonify({'error': 'Patient data not found'}), 404

    image, label = vol['image'], vol['label']
    uniform, density = vol['uniform_mask'], vol['density']
    std_density = vol.get('std_density')
    tissue_map = vol['tissue_map']
    D, H, W = label.shape
    limits  = [D, H, W]
    index   = int(np.clip(index, 0, limits[axis] - 1))

    # Extract 2D slices per axis
    if axis == 0:     # axial — fixed z
        img_sl = image[modality, index, :, :]
        lbl_sl = label[index, :, :]
        uni_sl = uniform[index, :, :] if uniform is not None else None
        den_sl = density[index, :, :] if density is not None else None
        std_sl = std_density[index, :, :] if std_density is not None else None
        tis_sl = tissue_map[index, :, :] if tissue_map is not None else None
    elif axis == 1:   # coronal — fixed y
        img_sl = image[modality, :, index, :]
        lbl_sl = label[:, index, :]
        uni_sl = uniform[:, index, :] if uniform is not None else None
        den_sl = density[:, index, :] if density is not None else None
        std_sl = std_density[:, index, :] if std_density is not None else None
        tis_sl = tissue_map[:, index, :] if tissue_map is not None else None
    else:             # sagittal — fixed x
        img_sl = image[modality, :, :, index]
        lbl_sl = label[:, :, index]
        uni_sl = uniform[:, :, index] if uniform is not None else None
        den_sl = density[:, :, index] if density is not None else None
        std_sl = std_density[:, :, index] if std_density is not None else None
        tis_sl = tissue_map[:, :, index] if tissue_map is not None else None

    h, w = img_sl.shape

    # Grayscale percentile windowing
    pos  = img_sl[img_sl > 0]
    p1   = float(np.percentile(pos, 1))  if len(pos) else 0.0
    p99  = float(np.percentile(pos, 99)) if len(pos) else 1.0
    norm = np.clip((img_sl - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
    pixels = (norm * 255).astype(np.uint8).flatten().tolist()

    # Extract contour polylines
    def get_contours(mask, level=0.5):
        if mask is None: return []
        bm = (mask > level).astype(float)
        if bm.sum() == 0: return []
        try:
            return [c.tolist() for c in find_contours(bm, level=0.5)]
        except Exception:
            return []

    tumor_c  = get_contours((lbl_sl > 0).astype(float))
    margin_c = get_contours(uni_sl.astype(float) if uni_sl is not None else None)
    density_c= get_contours(den_sl, level=0.35)  if den_sl is not None else []
    
    # Organs-At-Risk (OAR): CSF/Ventricles (tissue_map == 3)
    # We hard-exclude these from the final treatment contour to prevent ventricular toxicity.
    opt_sl = den_sl.copy() if den_sl is not None else None
    if opt_sl is not None and tis_sl is not None:
        opt_sl[tis_sl == 3] = 0.0

    optimal_threshold = float(request.args.get('optimal_threshold', 0.35))
    optimized_c = get_contours(opt_sl, level=optimal_threshold) if opt_sl is not None else []
    
    # Recommended Safety Margin (adjusted_value = mean_prediction + z * uncertainty)
    z_val = float(request.args.get('z_val', 1.0))
    safety_sl = None
    if den_sl is not None:
        safety_sl = den_sl.copy()
        if std_sl is not None:
            safety_sl = safety_sl + z_val * std_sl
        if tis_sl is not None:
            safety_sl[tis_sl == 3] = 0.0
            
    safety_margin_c = get_contours(safety_sl, level=optimal_threshold) if safety_sl is not None else []
    
    prob_95  = get_contours(den_sl, level=0.95) if den_sl is not None else []
    prob_80  = get_contours(den_sl, level=0.80) if den_sl is not None else []
    prob_50  = get_contours(den_sl, level=0.50) if den_sl is not None else []
    prob_20  = get_contours(den_sl, level=0.20) if den_sl is not None else []

    return jsonify({
        'pixels':  pixels,
        'width':   w,
        'height':  h,
        'contours': {
            'tumor':   tumor_c,
            'margin':  margin_c,
            'density': density_c,
            'optimized_density': optimized_c,
            'safety_margin': safety_margin_c,
            'probability_95': prob_95,
            'probability_80': prob_80,
            'probability_50': prob_50,
            'probability_20': prob_20,
        },
        'density_values': den_sl.flatten().tolist() if den_sl is not None else None,
        'std_values': std_sl.flatten().tolist() if std_sl is not None else None
    })
def compute_pca_2d(latents_dict):
    """
    Applies PCA using raw numpy to project a dictionary of {patient_id: [64-dim z]} into 2D coordinates.
    Returns a dictionary of {patient_id: [x, y]}.
    """
    import numpy as np
    
    # Convert dict to array
    patient_ids = list(latents_dict.keys())
    if len(patient_ids) == 0:
        return {}
        
    X = np.array([latents_dict[p_id] for p_id in patient_ids])
    
    # If only 1 patient, or all latents are identical, return dummy coordinates
    if len(patient_ids) < 2 or np.allclose(X, X[0]):
        return {p_id: [0.0, 0.0] for p_id in patient_ids}
        
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    # Eigenvalue decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns sorted in ascending order; get the indices of the top 2 largest
        idx = np.argsort(eigenvalues)[::-1][:2]
        top_vectors = eigenvectors[:, idx]
        
        # Project data onto top 2 eigenvectors
        X_projected = np.dot(X_centered, top_vectors)
        
        # Construct results
        res = {}
        for i, p_id in enumerate(patient_ids):
            res[p_id] = [float(X_projected[i, 0]), float(X_projected[i, 1])]
        return res
    except Exception as e:
        print(f"[Error] PCA computation failed: {e}")
        # Fallback to simple first two coordinates if PCA fails
        return {p_id: [float(latents_dict[p_id][0]), float(latents_dict[p_id][1])] for p_id in patient_ids}


@app.route("/api/latents_projection")
def get_latents_projection():
    """Computes and returns the 2D PCA projection of all saved patient latent vectors."""
    import json
    import os
    
    current_patient = request.args.get("current", "").strip()
    latents_path = "outputs/patient_latents.json"
    
    latents_dict = {}
    if os.path.exists(latents_path):
        try:
            with open(latents_path, "r") as f:
                latents_dict = json.load(f)
        except Exception:
            pass
            
    # Run PCA
    projections = compute_pca_2d(latents_dict)
    
    return jsonify({
        "projections": projections,
        "current_patient": current_patient
    })
@app.route("/api/margin_sweep/<patient_id>")
def get_margin_sweep(patient_id):
    """
    Computes a sweep of coverage vs healthy-tissue volume at thresholds from 0.0 to 1.0.
    """
    vol = _load_patient_volumes(patient_id)
    if vol is None or vol['density'] is None:
        return jsonify({'error': 'Prediction outputs not found'}), 404
        
    label      = vol['label']          # (D, H, W)
    recurrence = vol['recurrence']     # (D, H, W)
    spacing    = vol['spacing']        # (3,)
    density    = vol['density']        # (D, H, W)
    uniform    = vol['uniform_mask']   # (D, H, W)
    tissue_map = vol['tissue_map']     # (D, H, W)
    
    total_rec_voxels = int(np.sum(recurrence > 0))
    voxel_vol_cm3 = float(np.prod(spacing) / 1000.0)
    
    sweep_results = []
    thresholds = np.linspace(0.0, 1.0, 51) # 0.0, 0.02, ..., 1.0
    
    for t in thresholds:
        t_val = float(t)
        mask = (density >= t_val)
        
        # Organs-At-Risk (OAR): CSF/Ventricles (tissue_map == 3)
        # We hard-exclude these from the final treatment contour to prevent ventricular toxicity.
        mask_spared = mask & (tissue_map != 3)
        
        rec_in_target = int(np.sum((mask_spared > 0) & (recurrence > 0)))
        cov = (rec_in_target / total_rec_voxels * 100.0) if total_rec_voxels > 0 else 0.0
        
        healthy_treated_voxels = int(np.sum((mask_spared > 0) & (label == 0)))
        h_vol = healthy_treated_voxels * voxel_vol_cm3
        
        sweep_results.append({
            'threshold': round(t_val, 2),
            'coverage': round(cov, 2),
            'healthy_volume': round(h_vol, 3)
        })
        
    # Standard clinical margin metrics
    std_cov = 0.0
    std_h_vol = 0.0
    if uniform is not None:
        std_rec_in_target = int(np.sum((uniform > 0) & (recurrence > 0)))
        std_cov = (std_rec_in_target / total_rec_voxels * 100.0) if total_rec_voxels > 0 else 0.0
        std_h_vox = int(np.sum((uniform > 0) & (label == 0)))
        std_h_vol = std_h_vox * voxel_vol_cm3
        
    # Raw 0.35 threshold metrics (with OAR excluded for comparison)
    raw_mask = (density >= 0.35) & (tissue_map != 3)
    raw_rec_in = int(np.sum((raw_mask > 0) & (recurrence > 0)))
    raw_cov = (raw_rec_in / total_rec_voxels * 100.0) if total_rec_voxels > 0 else 0.0
    raw_h_vox = int(np.sum((raw_mask > 0) & (label == 0)))
    raw_h_vol = raw_h_vox * voxel_vol_cm3
    
    return jsonify({
        'sweep': sweep_results,
        'standard': {
            'coverage': round(std_cov, 2),
            'healthy_volume': round(std_h_vol, 3)
        },
        'raw_035': {
            'coverage': round(raw_cov, 2),
            'healthy_volume': round(raw_h_vol, 3)
        }
    })


@app.route("/api/safety_metrics/<patient_id>")
def get_safety_metrics(patient_id):
    """
    Computes 3D safety margin volume and coverage dynamically based on optimal_threshold and z_val.
    """
    optimal_threshold = float(request.args.get('optimal_threshold', 0.35))
    z_val = float(request.args.get('z_val', 1.0))
    
    vol = _load_patient_volumes(patient_id)
    if vol is None or vol['density'] is None or vol.get('std_density') is None:
        return jsonify({'error': 'Prediction outputs not found'}), 404
        
    label      = vol['label']
    recurrence = vol['recurrence']
    spacing    = vol['spacing']
    density    = vol['density']
    std_density = vol['std_density']
    tissue_map = vol['tissue_map']
    
    total_rec = np.sum(recurrence > 0)
    voxel_vol_cm3 = np.prod(spacing) / 1000.0
    
    # UCB safety mask
    ucb_vol = density + z_val * std_density
    ucb_mask = (ucb_vol >= optimal_threshold) & (tissue_map != 3)
    
    safety_cov = 0.0
    if total_rec > 0:
        safety_rec_in = np.sum((ucb_mask > 0) & (recurrence > 0))
        safety_cov = float(safety_rec_in / total_rec * 100.0)
        
    safety_healthy = float(np.sum((ucb_mask > 0) & (label == 0)) * voxel_vol_cm3)
    safety_volume = float(np.sum(ucb_mask > 0) * voxel_vol_cm3)
    
    return jsonify({
        'coverage': round(safety_cov, 2),
        'healthy_volume': round(safety_healthy, 3),
        'total_volume': round(safety_volume, 3)
    })


@app.route("/api/explain/<patient_id>")
def get_explainability(patient_id):
    """
    Factual Spread Explainability Module.

    Divides the peri-tumoral region into 6 anatomical sectors (superior/inferior/anterior/
    posterior/medial/lateral). For each sector measures infiltration extension beyond the
    visible tumor boundary, local WM fraction, and local mean diffusion coefficient D(x).
    Identifies the dominant sector and checks if the tumor shape elongates toward it.
    Fills a fixed text template — no free-form LLM text is used.

    D(x) values are literature-grounded:
      WM: 0.15 mm²/day, GM: 0.03 mm²/day, CSF: 0.0 (excluded)
      (Giese et al. clinical observations, consistent with Swanson glioma model)
    """
    npz_path = f"data/processed/{patient_id}.npz"
    ensemble_path = f"outputs/{patient_id}_prediction_ensemble.npz"
    cov_path = f"data/processed/{patient_id}_covariates.json"

    if not os.path.exists(npz_path) or not os.path.exists(ensemble_path):
        return jsonify({"error": "Pipeline outputs not found. Run target planning first."}), 404

    try:
        proc = np.load(npz_path)
        label      = proc['label'].astype(np.int8)      # (D, H, W) — tumor labels
        spacing    = proc['spacing'].astype(float)       # mm/voxel
        tissue_map = proc['tissue_map'].astype(np.int8) if 'tissue_map' in proc else np.zeros_like(label)

        ens = np.load(ensemble_path)
        density    = ens['mean_density'].astype(float)  # (D, H, W) in [0, 1]
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {e}"}), 500

    # ── Covariates ────────────────────────────────────────────────────────────
    from src.preprocess_upload import default_covariates
    defaults = default_covariates()
    idh_status   = defaults.get("idh_status",  "Unknown")
    mgmt_status  = defaults.get("mgmt_status", "Unknown")
    rho_estimate = 0.012   # literature GBM default (day⁻¹), Swanson model

    if os.path.exists(cov_path):
        try:
            with open(cov_path) as f:
                raw_cov = json.load(f)
            # Handle both flat {"idh_status": "Mutant"} and nested {"idh_status": {"value": ...}}
            def _get_cov(d, key, fallback):
                v = d.get(key, fallback)
                if isinstance(v, dict):
                    return v.get("value", fallback)
                return v if v else fallback
            idh_status  = _get_cov(raw_cov, "idh_status",  idh_status)
            mgmt_status = _get_cov(raw_cov, "mgmt_status", mgmt_status)
        except Exception:
            pass

    # ── Constants ─────────────────────────────────────────────────────────────
    D_WM  = 0.15   # mm²/day — white matter (Giese et al.; Swanson model)
    D_GM  = 0.03   # mm²/day — gray matter
    D_CSF = 0.0    # CSF excluded from infiltration computation
    INFILT_THRESH = 0.20  # probability threshold for "infiltrated" voxels

    # ── Geometry ─────────────────────────────────────────────────────────────
    tumor_voxels = np.argwhere(label > 0)
    if len(tumor_voxels) == 0:
        return jsonify({"error": "No tumor voxels found in label array"}), 404

    centroid = np.mean(tumor_voxels, axis=0)          # (z_c, y_c, x_c) in voxels
    z_c, y_c, x_c = centroid
    D, H, W = label.shape

    # Voxel coordinate grids
    zg = np.arange(D)[:, None, None] * np.ones((1, H, W))
    yg = np.arange(H)[None, :, None] * np.ones((D, 1, W))
    xg = np.arange(W)[None, None, :] * np.ones((D, H, 1))

    # Infiltration mask: density > threshold AND outside visible tumor
    infilt_mask = (density > INFILT_THRESH) & (label == 0)

    # ── Sector definitions ────────────────────────────────────────────────────
    # Radiological convention for 128³ isotropic volume:
    #   z axis (dim 0): z=0 → inferior, z=max → superior
    #   y axis (dim 1): y=0 → anterior,  y=max → posterior
    #   x axis (dim 2): x=0 → right,     x=max → left  (medial/lateral approximated as below/above centroid)
    sector_masks = {
        "superior":  zg > z_c,
        "inferior":  zg <= z_c,
        "anterior":  yg <= y_c,
        "posterior": yg > y_c,
        "medial":    xg <= x_c,
        "lateral":   xg > x_c,
    }

    # ── Global WM stats ───────────────────────────────────────────────────────
    brain_tissue_mask = (tissue_map >= 1) & (tissue_map <= 2)  # WM + GM (no CSF)
    global_wm_count   = int(np.sum(tissue_map == 1))
    global_brain_count = int(np.sum(brain_tissue_mask))
    global_wm_frac    = global_wm_count / max(global_brain_count, 1)

    # Global mean D(x) weighted by tissue type
    d_per_voxel_global = np.where(tissue_map == 1, D_WM,
                         np.where(tissue_map == 2, D_GM, 0.0))
    global_mean_D = float(np.mean(d_per_voxel_global[brain_tissue_mask])) if global_brain_count > 0 else D_GM

    # ── Per-sector analysis ───────────────────────────────────────────────────
    sector_results = {}
    for sname, smask in sector_masks.items():
        s_infilt = infilt_mask & smask
        s_tumor  = (label > 0) & smask

        # Extension = how far infiltrated voxels reach beyond the tumor boundary in this sector
        if s_infilt.sum() == 0:
            extension_mm = 0.0
        else:
            infilt_coords = np.argwhere(s_infilt).astype(float)
            infilt_dists  = np.sqrt(np.sum(((infilt_coords - centroid) * spacing) ** 2, axis=1))
            ext95 = float(np.percentile(infilt_dists, 95))

            if s_tumor.sum() > 0:
                tumor_coords = np.argwhere(s_tumor).astype(float)
                tumor_dists  = np.sqrt(np.sum(((tumor_coords - centroid) * spacing) ** 2, axis=1))
                tumor95 = float(np.percentile(tumor_dists, 95))
            else:
                tumor95 = 0.0

            extension_mm = max(0.0, ext95 - tumor95)

        # WM fraction in this sector (brain tissue only)
        s_brain = brain_tissue_mask & smask
        s_wm    = (tissue_map == 1) & smask
        s_wm_frac = int(s_wm.sum()) / max(int(s_brain.sum()), 1)

        # Mean D(x) in this sector
        s_d_vals = d_per_voxel_global[smask & brain_tissue_mask]
        s_mean_D = float(np.mean(s_d_vals)) if len(s_d_vals) > 0 else D_GM

        sector_results[sname] = {
            "extension_mm":  round(extension_mm, 1),
            "wm_fraction":   round(s_wm_frac * 100, 1),  # %
            "mean_D":        round(s_mean_D, 4),
        }

    # ── Dominant sector (greatest extension) ──────────────────────────────────
    dominant_sector = max(sector_results, key=lambda s: sector_results[s]["extension_mm"])
    dominant        = sector_results[dominant_sector]
    other_sectors   = [v for k, v in sector_results.items() if k != dominant_sector]
    other_mean_D    = round(float(np.mean([v["mean_D"] for v in other_sectors])), 4)

    # ── Shape elongation via PCA of tumor voxels ──────────────────────────────
    # Map sector names → unit direction vectors in (z, y, x) space
    sector_unit_dirs = {
        "superior":  np.array([ 1., 0., 0.]),
        "inferior":  np.array([-1., 0., 0.]),
        "anterior":  np.array([ 0.,-1., 0.]),
        "posterior": np.array([ 0., 1., 0.]),
        "medial":    np.array([ 0., 0.,-1.]),
        "lateral":   np.array([ 0., 0., 1.]),
    }
    elongation_toward_dominant = False
    alignment_score = 0.0
    sphericity_score = 1.0

    if len(tumor_voxels) >= 4:
        centered_mm = (tumor_voxels.astype(float) - centroid) * spacing
        cov_mat = np.cov(centered_mm.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
            # Principal axis = eigenvector with largest eigenvalue
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
            dom_vec = sector_unit_dirs[dominant_sector]
            alignment_score = float(abs(np.dot(principal_axis, dom_vec)))
            elongation_toward_dominant = alignment_score > 0.5

            lam_sorted = np.sort(eigenvalues)
            sphericity_score = round(float(lam_sorted[0] / max(lam_sorted[-1], 1e-9)), 3)
        except np.linalg.LinAlgError:
            pass

    # ── Fill fixed template ───────────────────────────────────────────────────
    elongation_text = (
        f"Yes (alignment {alignment_score:.2f}, sphericity {sphericity_score:.3f})"
        if elongation_toward_dominant
        else f"No (alignment {alignment_score:.2f}, sphericity {sphericity_score:.3f})"
    )

    DISCLAIMER = (
        "This is a computed summary of factors statistically associated with the model's "
        "predicted asymmetry, not a verified causal explanation of the network's internal computation."
    )

    summary_text = (
        f"Predicted spread extends furthest {dominant_sector} "
        f"({dominant['extension_mm']:.1f} mm beyond visible tumor boundary). "
        f"Contributing factors: white matter fraction in this direction: "
        f"{dominant['wm_fraction']:.1f}% vs. {round(global_wm_frac * 100, 1):.1f}% brain-wide average; "
        f"local diffusion estimate: {dominant['mean_D']:.4f}\u202fmm\u00b2/day vs. "
        f"{other_mean_D:.4f}\u202fmm\u00b2/day elsewhere; "
        f"tumor shape elongation toward this direction: {elongation_text}. "
        f"Overall spread magnitude is additionally informed by: estimated proliferation rate "
        f"\u03c1\u202f=\u202f{rho_estimate:.4f}\u202fday\u207b\u00b9; "
        f"IDH status: {idh_status}; MGMT status: {mgmt_status} "
        f"(note: MGMT-methylated tumors show different distant-recurrence patterns than "
        f"unmethylated tumors [Hegi et al., 2005]; IDH mutation is associated with slower "
        f"proliferation [Dang et al., 2009])."
    )

    return jsonify({
        "patient_id":                 patient_id,
        "dominant_sector":            dominant_sector,
        "sector_results":             sector_results,
        "dominant_extension_mm":      dominant["extension_mm"],
        "dominant_wm_fraction":       dominant["wm_fraction"],
        "global_wm_fraction":         round(global_wm_frac * 100, 1),
        "dominant_mean_D":            dominant["mean_D"],
        "other_sectors_mean_D":       other_mean_D,
        "global_mean_D":              round(global_mean_D, 4),
        "elongation_toward_dominant": elongation_toward_dominant,
        "alignment_score":            round(alignment_score, 3),
        "sphericity_score":           sphericity_score,
        "rho_estimate":               rho_estimate,
        "idh_status":                 idh_status,
        "mgmt_status":                mgmt_status,
        "summary_text":               summary_text,
        "disclaimer":                 DISCLAIMER,
    })



# ═══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE EVALUATION MODULE
# ═══════════════════════════════════════════════════════════════════════════════
# Computes per-patient and aggregate (mean ± std) metrics across the test set:
#
# CLINICAL ACCURACY
#   Recurrence Coverage (%)      — fraction of ground-truth recurrence inside contour
#   Healthy Tissue Irradiated    — non-tumor voxels inside contour (cm³)
#   Margin Volume (cm³)          — always reported alongside Coverage
#   Sensitivity / Specificity    — voxel-wise at the stated threshold
#   HD95 (mm)                    — 95th-pctl Hausdorff (NOT raw max, avoids outlier bias)
#   Surface Dice @ 2mm           — surface overlap with explicit 2mm tolerance
#   Average Surface Distance     — symmetric mean boundary-to-boundary (mm)
#
# EFFICIENCY
#   Inference Time (s)           — pulled from saved NPZ metadata
#   GPU Memory (MB)              — pulled from comparison report
#
# MODEL SANITY CHECK  ← labeled separately; this is NOT a clinical accuracy metric
#   Physics Residual (MSE)       — mean squared residual of Fisher-KPP PDE on
#                                  the model's own predicted field; measures how
#                                  well the network satisfies its governing equation
#
# All boundary metrics use scipy.ndimage.distance_transform_edt on the binary
# surface masks, which is equivalent to medpy's implementation.
#
# References:
#   HD95: Menze et al., IEEE TMI 2015 (BraTS benchmark standard)
#   Surface Dice: Nikolov et al., 2018 (tolerance=2mm following BraTS practice)
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.ndimage import binary_erosion, distance_transform_edt


def _get_surface_mask(binary_mask):
    """Boolean mask of surface voxels (border of a binary 3-D region)."""
    eroded = binary_erosion(binary_mask)
    return binary_mask & ~eroded


def _surface_distances(pred_mask, gt_mask, spacing):
    """
    Returns (d_pred_to_gt, d_gt_to_pred) — arrays of per-surface-voxel
    distances (in mm) between the two boundaries.
    Returns (None, None) if either mask is empty or has no surface.
    """
    pred_surf = _get_surface_mask(pred_mask)
    gt_surf   = _get_surface_mask(gt_mask)

    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return None, None

    # EDT from each surface — sampling gives correct mm distances
    gt_edt   = distance_transform_edt(~gt_surf,   sampling=spacing)
    pred_edt = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = gt_edt[pred_surf]
    d_gt_to_pred = pred_edt[gt_surf]
    return d_pred_to_gt, d_gt_to_pred


def _hd95(pred_mask, gt_mask, spacing):
    """95th-percentile symmetric Hausdorff distance (mm)."""
    d1, d2 = _surface_distances(pred_mask, gt_mask, spacing)
    if d1 is None:
        return float("nan")
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def _surface_dice(pred_mask, gt_mask, spacing, tol_mm=2.0):
    """
    Surface Dice at explicit tolerance tol_mm.
    = (pred_surf within tol + gt_surf within tol) / (|pred_surf| + |gt_surf|)
    """
    pred_surf = _get_surface_mask(pred_mask)
    gt_surf   = _get_surface_mask(gt_mask)
    n_pred = int(pred_surf.sum())
    n_gt   = int(gt_surf.sum())
    if n_pred == 0 or n_gt == 0:
        return float("nan")

    gt_edt   = distance_transform_edt(~gt_surf,   sampling=spacing)
    pred_edt = distance_transform_edt(~pred_surf, sampling=spacing)

    pred_within = int(np.sum(gt_edt[pred_surf]   <= tol_mm))
    gt_within   = int(np.sum(pred_edt[gt_surf]   <= tol_mm))
    return float((pred_within + gt_within) / (n_pred + n_gt))


def _asd(pred_mask, gt_mask, spacing):
    """Average symmetric surface distance (mm)."""
    d1, d2 = _surface_distances(pred_mask, gt_mask, spacing)
    if d1 is None:
        return float("nan")
    return float((np.mean(d1) + np.mean(d2)) / 2.0)


def _physics_residual_mse(density, tissue_map, spacing, rho=0.012):
    """
    Mean squared residual of the Fisher-KPP PDE evaluated on the model's
    own predicted field c (= density).

    PDE: ∂c/∂t = ∇·(D(x)∇c) + ρ·c·(1−c)

    At any spatial snapshot, the spatial residual is:
        R(x) = ∇·(D(x)∇c) + ρ·c·(1−c)

    For a perfect solution R ≡ 0.  We approximate ∇·(D∇c) ≈ D·∇²c using
    central finite differences (separable per axis, then summed).

    D(x) per tissue type (literature-grounded, Giese/Swanson model):
      WM=1 → 0.15 mm²/day,  GM=2 → 0.03 mm²/day,  CSF=3 → 0 (excluded)

    Returns: scalar MSE over WM+GM brain voxels.
    """
    D_map = np.where(tissue_map == 1, 0.15,
            np.where(tissue_map == 2, 0.03, 0.0)).astype(float)

    c = density.astype(float)

    # ∇²c via central finite differences per axis
    laplacian = np.zeros_like(c)
    for ax in range(3):
        dz = float(spacing[ax])
        laplacian += np.gradient(np.gradient(c, dz, axis=ax), dz, axis=ax)

    diffusion_term = D_map * laplacian                 # ∇·(D∇c) ≈ D·∇²c
    reaction_term  = rho * c * (1.0 - c)              # ρ·c·(1−c)
    residual       = diffusion_term + reaction_term    # should ≈ 0

    brain_mask = (tissue_map == 1) | (tissue_map == 2)
    n = int(brain_mask.sum())
    if n == 0:
        return float("nan")
    return float(np.mean(residual[brain_mask] ** 2))


def _compute_method_metrics(pred_mask, gt_recurrence, label, spacing,
                            threshold_used, voxel_vol_cm3, surface_tol_mm=2.0):
    """
    Compute all clinical accuracy metrics for one (predicted mask, patient) pair.
    pred_mask : bool 3-D array — the treatment contour
    gt_recurrence : bool 3-D array — ground-truth recurrence voxels
    label : int8 3-D array — original tumor label (for healthy-tissue calc)
    """
    rec = gt_recurrence.astype(bool)
    pred = pred_mask.astype(bool)

    total_rec = int(rec.sum())
    total_neg = int((~rec).sum())

    # Coverage / Sensitivity
    tp = int((pred & rec).sum())
    fp = int((pred & ~rec).sum())
    fn = int((~pred & rec).sum())
    tn = int((~pred & ~rec).sum())

    coverage    = float(tp / max(total_rec, 1) * 100.0)
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))

    # Volume metrics
    margin_vol    = float(pred.sum() * voxel_vol_cm3)
    healthy_vol   = float((pred & (label == 0)).sum() * voxel_vol_cm3)

    # Boundary metrics (only if recurrence mask is non-trivial)
    hd95_val = asd_val = sdice_val = float("nan")
    if total_rec > 0 and pred.sum() > 0:
        hd95_val  = _hd95(pred,  rec, spacing)
        sdice_val = _surface_dice(pred, rec, spacing, tol_mm=surface_tol_mm)
        asd_val   = _asd(pred, rec, spacing)

    return {
        "recurrence_coverage_pct":    round(coverage, 2),
        "sensitivity":                round(sensitivity, 4),
        "specificity":                round(specificity, 4),
        "margin_volume_cm3":          round(margin_vol, 3),
        "healthy_tissue_cm3":         round(healthy_vol, 3),
        "hd95_mm":                    round(hd95_val, 2) if not np.isnan(hd95_val) else None,
        "surface_dice":               round(sdice_val, 4) if not np.isnan(sdice_val) else None,
        "asd_mm":                     round(asd_val, 2) if not np.isnan(asd_val) else None,
        "threshold_used":             threshold_used,
        "surface_dice_tolerance_mm":  surface_tol_mm,
    }


@app.route("/api/evaluation_metrics")
def get_evaluation_metrics():
    """
    Comprehensive quantitative evaluation across all test patients.

    For each patient that has *_prediction_ensemble.npz:
      - Computes all metrics for three methods:
          1. Clinical Standard (uniform 1.5cm expansion, from baseline_uniform.npz)
          2. Vanilla PINN (from baseline_pinn.npz, threshold 0.35)
          3. MarginSense Ensemble (from prediction_ensemble.npz, threshold 0.35)
      - Boundary metrics (HD95, Surface Dice, ASD) vs ground-truth recurrence mask
      - Physics Residual (MarginSense only) — labeled separately as model sanity check

    Returns mean ± std across patients for every metric.
    """
    ms_threshold   = float(request.args.get("threshold", 0.35))
    surface_tol_mm = 2.0   # stated explicitly — Surface Dice at 2mm tolerance

    # Discover test patients that have all required files
    test_patients = []
    for f in sorted(os.listdir("outputs")):
        if f.endswith("_prediction_ensemble.npz"):
            pid = f.replace("_prediction_ensemble.npz", "")
            proc_path = f"data/processed/{pid}.npz"
            if os.path.exists(proc_path):
                test_patients.append(pid)

    if not test_patients:
        return jsonify({"error": "No processed patients with ensemble predictions found."}), 404

    n = len(test_patients)

    # Per-method per-patient accumulators
    METHOD_NAMES = [
        "Clinical Standard (Uniform Margin)",
        "Vanilla PINN (Per-Patient)",
        "MarginSense (Ensemble Amortized)",
    ]
    all_results = {m: [] for m in METHOD_NAMES}
    per_patient_data = {}

    for pid in test_patients:
        proc     = np.load(f"data/processed/{pid}.npz")
        label    = proc["label"].astype(np.int8)
        spacing  = proc["spacing"].astype(float)
        tissue_map = proc["tissue_map"].astype(np.int8) if "tissue_map" in proc else np.zeros_like(label)

        rec_raw  = proc.get("recurrence", None)
        if rec_raw is None:
            continue
        gt_recurrence = (rec_raw > 0)

        voxel_vol_cm3 = float(np.prod(spacing)) / 1000.0

        patient_row = {}

        # ── Method 1: Clinical Standard (dilated_mask from baseline_uniform.npz) ──
        std_path = f"outputs/{pid}_baseline_uniform.npz"
        if os.path.exists(std_path):
            std_data   = np.load(std_path)
            std_mask   = std_data["dilated_mask"].astype(bool)
            std_time   = float(std_data.get("elapsed_time", 0))
            std_metrics = _compute_method_metrics(
                std_mask, gt_recurrence, label, spacing,
                threshold_used="1.5cm expansion", voxel_vol_cm3=voxel_vol_cm3,
                surface_tol_mm=surface_tol_mm
            )
            std_metrics["inference_time_s"]  = round(std_time, 3)
            std_metrics["gpu_memory_mb"]     = None   # CPU method
            std_metrics["physics_residual"]  = None   # not applicable
            patient_row["Clinical Standard (Uniform Margin)"] = std_metrics
            all_results["Clinical Standard (Uniform Margin)"].append(std_metrics)

        # ── Method 2: Vanilla PINN ────────────────────────────────────────────
        pinn_path = f"outputs/{pid}_baseline_pinn.npz"
        if os.path.exists(pinn_path):
            pinn_data    = np.load(pinn_path)
            pinn_density = pinn_data["density"].astype(float)
            pinn_mask    = pinn_density >= ms_threshold
            # Exclude CSF/Ventricles from PINN mask (same OAR rule)
            pinn_mask    = pinn_mask & (tissue_map != 3)
            pinn_time    = float(pinn_data.get("elapsed_time", 0))
            rho_pinn     = float(pinn_data.get("rho", 0.012))

            pinn_metrics = _compute_method_metrics(
                pinn_mask, gt_recurrence, label, spacing,
                threshold_used=ms_threshold, voxel_vol_cm3=voxel_vol_cm3,
                surface_tol_mm=surface_tol_mm
            )
            pinn_metrics["inference_time_s"] = round(pinn_time, 3)
            pinn_metrics["gpu_memory_mb"]    = None  # not tracked in NPZ
            # Physics residual for PINN — using its own density and its own rho
            pinn_metrics["physics_residual"] = round(
                _physics_residual_mse(pinn_density, tissue_map, spacing, rho=rho_pinn), 6
            )
            patient_row["Vanilla PINN (Per-Patient)"] = pinn_metrics
            all_results["Vanilla PINN (Per-Patient)"].append(pinn_metrics)

        # ── Method 3: MarginSense Ensemble ────────────────────────────────────
        ens_path = f"outputs/{pid}_prediction_ensemble.npz"
        if os.path.exists(ens_path):
            ens_data   = np.load(ens_path)
            mean_dens  = ens_data["mean_density"].astype(float)
            ens_time   = float(ens_data.get("inference_time", 0))
            ms_mask    = (mean_dens >= ms_threshold) & (tissue_map != 3)

            ms_metrics = _compute_method_metrics(
                ms_mask, gt_recurrence, label, spacing,
                threshold_used=ms_threshold, voxel_vol_cm3=voxel_vol_cm3,
                surface_tol_mm=surface_tol_mm
            )
            ms_metrics["inference_time_s"] = round(ens_time, 3)
            ms_metrics["gpu_memory_mb"]    = None
            # Physics residual for MarginSense — model sanity check
            ms_metrics["physics_residual"] = round(
                _physics_residual_mse(mean_dens, tissue_map, spacing, rho=0.012), 6
            )
            patient_row["MarginSense (Ensemble Amortized)"] = ms_metrics
            all_results["MarginSense (Ensemble Amortized)"].append(ms_metrics)

        per_patient_data[pid] = patient_row

    # ── Aggregate mean ± std per method ──────────────────────────────────────
    NUMERIC_KEYS = [
        "recurrence_coverage_pct", "sensitivity", "specificity",
        "margin_volume_cm3", "healthy_tissue_cm3",
        "hd95_mm", "surface_dice", "asd_mm",
        "inference_time_s", "physics_residual",
    ]

    def _agg(rows):
        """Mean ± std across patients, skipping None values."""
        out = {}
        for key in NUMERIC_KEYS:
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                out[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)),  4),
                    "min":  round(float(np.min(vals)),  4),
                    "max":  round(float(np.max(vals)),  4),
                    "n":    len(vals),
                }
            else:
                out[key] = None
        return out

    aggregated = {m: _agg(all_results[m]) for m in METHOD_NAMES if all_results[m]}

    return jsonify({
        "n_patients":              n,
        "patients":                test_patients,
        "threshold_used":          ms_threshold,
        "surface_dice_tolerance_mm": surface_tol_mm,
        "note":                    f"n={n} patients \u2014 treat as preliminary. "
                                   f"Mean \u00b1 std reported across test set.",
        "methods":                 METHOD_NAMES,
        "per_patient":             per_patient_data,
        "aggregated":              aggregated,
        "metric_sections": {
            "clinical_accuracy": [
                "recurrence_coverage_pct", "sensitivity", "specificity",
                "margin_volume_cm3", "healthy_tissue_cm3",
                "hd95_mm", "surface_dice", "asd_mm"
            ],
            "efficiency": ["inference_time_s"],
            "model_sanity_check": ["physics_residual"],
        },
        "metric_labels": {
            "recurrence_coverage_pct": "Recurrence Coverage (%)",
            "sensitivity":             "Sensitivity",
            "specificity":             "Specificity",
            "margin_volume_cm3":       "Margin Volume (cm³)",
            "healthy_tissue_cm3":      "Healthy Tissue Irradiated (cm³)",
            "hd95_mm":                 "HD95 (mm) [95th-pctl Hausdorff]",
            "surface_dice":            f"Surface Dice @ {surface_tol_mm}mm",
            "asd_mm":                  "Avg Surface Distance (mm)",
            "inference_time_s":        "Inference Time (s)",
            "physics_residual":        "Physics Residual MSE [Fisher-KPP]",
        },
    })


if __name__ == "__main__":
    print("[*] Launching MarginSense Unified Dashboard Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
