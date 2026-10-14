import os
import sys
import json
import time
import shutil
import hashlib
import numpy as np
import pandas as pd

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
RESULTS_DIR = os.path.join(WORKSPACE, "results")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")
LATEX_DIR = os.path.join(WORKSPACE, "latex_tables")
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")
PARAMS_DIR = os.path.join(RESULTS_DIR, "params")

SEED = 42

def run_master_pipeline():
    start_epoch = time.time()
    start_iso = time.strftime("%Y-%m-%d %H:%M:%S")

    print("=== STARTING REPRODUCIBLE RESEARCH PIPELINE ===", flush=True)

    # 1. Clear previous outputs completely (rm -rf results/, figures/, latex_tables/)
    for d in [RESULTS_DIR, FIGURES_DIR, LATEX_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)

    # Save manifest.json immediately at run start
    manifest_data = {
        'Run_Start_Time_ISO': start_iso,
        'Run_Start_Time_Epoch': start_epoch,
        'Seed': SEED,
        'Status': 'RUNNING'
    }
    with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest_data, f, indent=4)

    # 2. Stage 0: Data Audit
    from stage0_audit import run_stage0
    run_stage0()

    # 3. Stage 1: Three Evaluation Levels
    from stage1_levels import run_stage1
    run_stage1()

    # 4. Stage 2: Specialists, Tuning, Ensembles
    from stage2_models import run_stage2
    run_stage2()

    # 5. Stage 3: Universal Feature Study
    from stage3_universal import run_stage3
    run_stage3()

    # 6. Stage 4: Statistical Proof & Hypotheses
    from stage4_stats import run_stage4
    run_stage4()

    # 7. Stage 5: Explainability & HTML LIME
    from stage5_explainability import run_stage5
    run_stage5()

    # 8. Stage 6: Figures & LaTeX Tables
    from stage6_figures import run_stage6
    run_stage6()

    # Update manifest.json with completion details after all stages (including stage6) finish
    manifest_data['Run_End_Time_Epoch'] = time.time()
    manifest_data['Status'] = 'COMPLETE'
    with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest_data, f, indent=4)

    print("\nALL STAGES FINISHED. RUNNING VERIFICATION SCRIPT...", flush=True)
    import subprocess
    ret = subprocess.run([sys.executable, os.path.join(WORKSPACE, "verify.py")])
    if ret.returncode != 0:
        print("VERIFICATION SCRIPT FAILED!", flush=True)
        sys.exit(ret.returncode)
    else:
        print("VERIFICATION SCRIPT PASSED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    run_master_pipeline()
