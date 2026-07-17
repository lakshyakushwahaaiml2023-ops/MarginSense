import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import torch

def run_script_if_missing(script_path, patient_id, output_path, extra_args=[]):
    """Runs a script using subprocess if its output does not already exist."""
    if not os.path.exists(output_path):
        print(f"[*] Output not found at {output_path}. Running {script_path}...")
        cmd = [sys.executable, script_path, patient_id] + extra_args
        # Set PYTHONPATH to root of workspace
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        start = time.perf_counter()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"[Error] Failed to run {script_path}: {result.stderr}")
            sys.exit(1)
        print(f"[+] Successfully ran {script_path} (took {elapsed:.2f} seconds)")
    else:
        print(f"[+] Loaded existing outputs for {os.path.basename(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="MarginSense Baseline Comparison & Stats Engine")
    parser.add_argument("patient_id", type=str, nargs="?", default="synthetic_patient_2",
                        help="Patient ID to compare. Ignored if --synthetic is set.")
    parser.add_argument("--synthetic", action="store_true", help="Run comparison on synthetic patient 2.")
    parser.add_argument("--threshold", type=float, default=0.2, 
                        help="Density threshold to define the irradiated target volume (c >= threshold) for PINN models.")
    args = parser.parse_args()
    
    # 1. Setup Patient ID and Paths
    patient_id = "synthetic_patient_2" if args.synthetic else args.patient_id
    processed_path = f"data/processed/{patient_id}.npz"
    
    if not os.path.exists(processed_path):
        print(f"[Error] Preprocessed data not found for patient {patient_id} at {processed_path}.")
        print("Please run preprocessing first, or use the --synthetic flag.")
        sys.exit(1)
        
    print(f"[*] Target Patient: {patient_id}")
    print(f"[*] Density Threshold for PINNs: {args.threshold}")
    
    # Define script outputs
    uniform_out   = f"outputs/{patient_id}_baseline_uniform.npz"
    pinn_out      = f"outputs/{patient_id}_baseline_pinn.npz"
    ensemble_out  = f"outputs/{patient_id}_prediction_ensemble.npz"
    gliodil_out   = f"outputs/{patient_id}_baseline_gliodil.npz"
    
    # 2. Run baseline models if their output files are missing
    # We run the sub-scripts normally (without --synthetic) so that they load the 
    # actual preprocessed .npz file from data/processed/{patient_id}.npz
    run_script_if_missing("src/baseline_uniform_margin.py", patient_id, uniform_out, [])
    run_script_if_missing("src/baseline_vanilla_pinn.py", patient_id, pinn_out, ["--epochs", "500"])
    run_script_if_missing("src/evaluate_uncertainty.py", patient_id, ensemble_out, [])
    run_script_if_missing("src/baseline_gliodil.py", patient_id, gliodil_out, ["--iters", "1000"])
    
    # 3. Load processed ground-truth and outputs
    data_gt = np.load(processed_path)
    label = data_gt['label']           # Original pre-treatment tumor mask
    recurrence = data_gt['recurrence'] # Post-treatment recurrence mask
    spacing = data_gt['spacing']       # Voxel spacing in mm (e.g. 1.0, 1.0, 1.0)
    
    # Compute physical voxel volume in cubic centimeters (cm^3)
    # Voxel volume (mm^3) = dx * dy * dz
    # 1 cm^3 = 1000 mm^3 -> divide by 1000
    voxel_vol_cm3 = np.prod(spacing) / 1000.0
    
    # Total recurrence volume in voxels
    total_rec_voxels = np.sum(recurrence > 0)
    print(f"[*] Ground-truth recurrence volume: {total_rec_voxels} voxels ({total_rec_voxels * voxel_vol_cm3:.3f} cm^3)")
    
    # Load Uniform Margin Baseline
    data_uniform = np.load(uniform_out)
    uniform_mask = data_uniform['dilated_mask']
    uniform_time = data_uniform['elapsed_time']
    
    # Load Vanilla per-patient PINN
    data_pinn = np.load(pinn_out)
    pinn_density = data_pinn['density']
    pinn_time = data_pinn['elapsed_time']
    pinn_mask = (pinn_density >= args.threshold).astype(np.int8)
    
    # Load MarginSense (Ensemble Amortized PINN)
    data_ensemble = np.load(ensemble_out)
    ensemble_mean = data_ensemble['mean_density']
    ensemble_time = data_ensemble['inference_time']
    ensemble_mask = (ensemble_mean >= args.threshold).astype(np.int8)

    # Load GliODIL (Reproduced) — discrete field optimizer
    data_gliodil = np.load(gliodil_out)
    gliodil_density = data_gliodil['density']
    gliodil_time    = data_gliodil['elapsed_time']   # wall-clock per-patient optimization time
    gliodil_mask    = (gliodil_density >= args.threshold).astype(np.int8)
    gliodil_gpu_mb  = float(data_gliodil.get('peak_gpu_memory_mb', 0.0))
    
    # 4. Compute Metrics per Method
    # ── SECTION A: Measured on our test set (N=4, same pipeline, LOOCV) ──────
    methods = {
        "Clinical Standard (Uniform Margin)": {
            "mask": uniform_mask,
            "time": uniform_time,
            "type": "CPU (scipy EDT)"
        },
        "Vanilla PINN (Per-Patient)": {
            "mask": pinn_mask,
            "time": pinn_time,
            "type": f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        },
        "GliODIL (Reproduced)": {
            "mask": gliodil_mask,
            "time": gliodil_time,
            "type": f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        },
        "MarginSense (Ensemble Amortized)": {
            "mask": ensemble_mask,
            "time": ensemble_time,
            "type": f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        }
    }
    
    report_data = {
        "patient_id": patient_id,
        "threshold": args.threshold,
        "recurrence_volume_cm3": float(total_rec_voxels * voxel_vol_cm3),
        "results": {},
        # GliODIL paper's own reported numbers on their 152-patient cohort.
        # THESE ARE NOT FROM OUR N=4 TEST SET — they are non-comparable reference
        # figures included for context only. Do NOT merge these into the measured table.
        "literature_reference": {
            "source": "Balcerak et al., Nature Communications 2025",
            "cohort": "N=152 glioblastoma patients (different cohort, not our test set)",
            "non_comparable": True,
            "disclaimer": (
                "These numbers are from GliODIL's own 152-patient study cohort. "
                "They cannot be directly compared to our N=4 reproduction results. "
                "They are provided as context only."
            ),
            "reported_metrics": {
                "recurrence_coverage_vs_standard_margin_improvement": "+4pct (64->68%)",
                "cohort_size": 152,
                "paper_doi": "10.1038/s41467-024-56098-y"
            }
        }
    }
    
    for name, m_data in methods.items():
        mask = m_data["mask"]
        
        # Recurrence Coverage %: fraction of actual recurrence volume inside treatment mask
        rec_in_target = np.sum((mask > 0) & (recurrence > 0))
        coverage = (rec_in_target / total_rec_voxels * 100.0) if total_rec_voxels > 0 else 0.0
        
        # Treated Healthy-Tissue Volume (cm^3): volume of treatment mask outside original tumor (label == 0)
        healthy_treated_voxels = np.sum((mask > 0) & (label == 0))
        healthy_vol_cm3 = healthy_treated_voxels * voxel_vol_cm3
        
        m_data["coverage"] = coverage
        m_data["healthy_spared_vol_cm3"] = healthy_vol_cm3
        m_data["total_target_vol_cm3"] = float(np.sum(mask > 0) * voxel_vol_cm3)
        
        entry = {
            "processing_time_seconds": float(m_data["time"]),
            "recurrence_coverage_percent": float(coverage),
            "treated_healthy_tissue_volume_cm3": float(healthy_vol_cm3),
            "total_target_volume_cm3": float(m_data["total_target_vol_cm3"]),
            "hardware_device": m_data["type"]
        }
        # Record GliODIL peak GPU memory separately (prominent in dashboard)
        if name == "GliODIL (Reproduced)":
            entry["peak_gpu_memory_mb"] = gliodil_gpu_mb
            entry["method_type"] = "per_patient_optimization"  # for dashboard runtime badge
        elif "MarginSense" in name:
            entry["method_type"] = "amortized_single_forward_pass"
        elif "PINN" in name:
            entry["method_type"] = "per_patient_optimization"
        else:
            entry["method_type"] = "geometric_expansion"
        report_data["results"][name] = entry
        
    # 5. Print Formatted ASCII Table
    # ── SECTION A: Measured on our test set ──────────────────────────────────
    W = 105
    print("\n" + "="*W)
    print(f"          [SECTION A] MEASURED ON OUR TEST SET — COMPARATIVE REPORT FOR {patient_id.upper()}")
    print("="*W)
    header_fmt = "{:<37} | {:<18} | {:<21} | {:<18}"
    row_fmt    = "{:<37} | {:<18} | {:<21} | {:<18.3f}"
    print(header_fmt.format("Method", "Time (seconds)", "Recurrence Coverage", "Healthy Vol (cm³)"))
    print(header_fmt.format("", "[wall-clock/patient]", "(higher is better)", "(lower is better)"))
    print("-" * W)
    
    for name, m_data in methods.items():
        t_sec = m_data['time']
        # Format time: show seconds + minutes for long-running methods
        if t_sec > 60:
            t_str = f"{t_sec:.1f}s ({t_sec/60:.1f} min)"
        else:
            t_str = f"{t_sec:.4f}s"
        print(row_fmt.format(name, t_str, f"{m_data['coverage']:.2f}%", m_data['healthy_spared_vol_cm3']))
    print("="*W)

    # ── SECTION B: Literature Reference (NON-COMPARABLE) ─────────────────────
    print("\n" + "-"*W)
    print("  [SECTION B] LITERATURE REFERENCE ONLY — DIFFERENT COHORT — NOT DIRECTLY COMPARABLE")
    print("-"*W)
    print("  Source: Balcerak et al., Nature Communications 2025")
    print("  Cohort: N=152 glioblastoma patients (THEIR dataset, not our N=4 test set)")
    print("  GliODIL paper-reported recurrence coverage improvement: ~64% → ~68% vs. standard margin")
    print("  ⚠  These numbers cannot be compared to our N=4 reproduction — different patients, cohort, and setup.")
    print("-"*W)
    
    # 6. Save JSON Report
    report_file = f"outputs/{patient_id}_comparison_report.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"[+] Saved comparison report in JSON format to {report_file}\n")

if __name__ == "__main__":
    main()
