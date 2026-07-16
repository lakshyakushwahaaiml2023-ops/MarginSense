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
    """Loads and serves clinical covariates for the patient."""
    cov_path = f"data/processed/{patient_id}_covariates.json"
    if os.path.exists(cov_path):
        with open(cov_path, "r") as f:
            return jsonify(json.load(f))
    # Fallback to defaults
    from src.preprocess_upload import default_covariates
    return jsonify(default_covariates())

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
    """Load all volumes needed for slice rendering."""
    npz_path = f"data/processed/{patient_id}.npz"
    if not os.path.exists(npz_path):
        return None
    try:
        data    = np.load(npz_path)
        image   = data['image']    # (4, D, H, W)
        label   = data['label']    # (D, H, W) — tumor label
        spacing = data['spacing']  # (3,)
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
    ensemble_path = f"outputs/{patient_id}_prediction_ensemble.npz"
    if os.path.exists(ensemble_path):
        try:
            density = np.load(ensemble_path)['mean_density']
        except Exception:
            pass

    return {'image': image, 'label': label, 'spacing': spacing,
            'uniform_mask': uniform_mask, 'density': density}


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
    D, H, W = label.shape
    limits  = [D, H, W]
    index   = int(np.clip(index, 0, limits[axis] - 1))

    # Extract 2D slices per axis
    if axis == 0:     # axial — fixed z
        img_sl = image[modality, index, :, :]
        lbl_sl = label[index, :, :]
        uni_sl = uniform[index, :, :] if uniform is not None else None
        den_sl = density[index, :, :] if density is not None else None
    elif axis == 1:   # coronal — fixed y
        img_sl = image[modality, :, index, :]
        lbl_sl = label[:, index, :]
        uni_sl = uniform[:, index, :] if uniform is not None else None
        den_sl = density[:, index, :] if density is not None else None
    else:             # sagittal — fixed x
        img_sl = image[modality, :, :, index]
        lbl_sl = label[:, :, index]
        uni_sl = uniform[:, :, index] if uniform is not None else None
        den_sl = density[:, :, index] if density is not None else None

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

    return jsonify({
        'pixels':  pixels,
        'width':   w,
        'height':  h,
        'contours': {
            'tumor':   tumor_c,
            'margin':  margin_c,
            'density': density_c,
        }
    })


if __name__ == "__main__":
    print("[*] Launching MarginSense Unified Dashboard Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
