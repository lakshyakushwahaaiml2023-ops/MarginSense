import os
import sys
import json
import csv
import subprocess
import torch
import threading
from flask import Flask, jsonify, request, send_from_directory

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
        cmd_compare = [sys.executable, "src/compare_models.py", patient_id, "--threshold", "0.2"]
        res = subprocess.run(cmd_compare, env=env, capture_output=True, text=True)
        log_message(res.stdout + res.stderr)
        if res.returncode != 0:
            log_message(f"[Error] Metrics evaluation failed with exit code {res.returncode}")
            return

        # 5. Run exporter to sync 3D WebUI data
        log_message("\n[*] Rebuilding and downsampling 3D WebGL point-cloud coordinates...")
        cmd_export = [sys.executable, "src/viz/export_json.py", patient_id]
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
    """Lists available patient folders or fallbacks to synthetic options."""
    raw_dir = "data/raw"
    patients = []
    if os.path.exists(raw_dir):
        patients = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if not patients:
        patients = ["synthetic_patient_1", "synthetic_patient_2", "synthetic_patient_3"]
    return jsonify(patients)

@app.route("/api/gpu_status")
def gpu_status():
    """Queries GPU status: PyTorch memory allocations and nvidia-smi utilization."""
    status = {"utilization": 0, "memory_used": 0, "memory_total": 6141, "torch_allocated": 0}
    try:
        if torch.cuda.is_available():
            status["torch_allocated"] = float(torch.cuda.memory_allocated(0) / (1024 * 1024))
            
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
    """Generates and serves 3D downsampled mesh points for a selected patient."""
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        # Re-run exporter to generate/refresh data.js
        cmd = [sys.executable, "src/viz/export_json.py", patient_id]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            return jsonify({"error": f"Failed to export patient data: {res.stderr}"}), 500
            
        data_js_path = os.path.join(VIZ_DIR, "data.js")
        if os.path.exists(data_js_path):
            with open(data_js_path, "r") as f:
                content = f.read().strip()
                # Extract JSON from window.patientData = { ... };
                json_str = content.replace("window.patientData = ", "").rstrip(";")
                return jsonify(json.loads(json_str))
                
        return jsonify({"error": "Data file not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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

if __name__ == "__main__":
    print("[*] Launching MarginSense Unified Dashboard Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
