#!/usr/bin/env python3
"""
Master Execution Pipeline for the CIC-Trap4Phish Benchmark Audit.

Usage:
  python run_pipeline.py --check        # Quick validation of all 13 artifacts (1 sec)
  python run_pipeline.py --verify       # Run comprehensive empirical verification (3 sec)
  python run_pipeline.py --smoke        # Run end-to-end smoke test on sample data (5 sec)
  python run_pipeline.py --stage <0-6>  # Execute a specific pipeline stage
  python run_pipeline.py --full         # Recompute the entire research pipeline from scratch
"""

import os
import sys
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
os.environ["WORKSPACE_DIR"] = REPO_ROOT

# Ensure code and scripts directories are on sys.path
for d in ["code", "scripts", "adapters"]:
    p = os.path.join(REPO_ROOT, d)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)


def run_cmd(cmd_list, desc="Command"):
    print(f"\n>>> Running: {desc} ({' '.join(cmd_list)})", flush=True)
    res = subprocess.run(cmd_list, cwd=REPO_ROOT)
    if res.returncode != 0:
        print(f"[ERROR] {desc} failed with exit code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"[SUCCESS] {desc} completed successfully.\n", flush=True)


def check_artifacts():
    print("Checking 13 required research artifacts...", flush=True)
    run_cmd([sys.executable, os.path.join(REPO_ROOT, "check_13_files.py")], "Artifact Existence & Integrity Check")


def verify_repo():
    print("Executing full research verification suite...", flush=True)
    run_cmd([sys.executable, os.path.join(REPO_ROOT, "verify.py")], "Automated Verification Suite")


def run_smoke_test():
    print("Running end-to-end smoke test on sample data...", flush=True)
    run_cmd([sys.executable, "-m", "unittest", "discover", "-s", "tests"], "Unit & Smoke Test Suite")


def run_stage(stage_num):
    stage_scripts = {
        0: "stage0_audit.py",
        1: "stage1_levels.py",
        2: "stage2_models.py",
        3: "stage3_universal.py",
        4: "stage4_stats.py",
        5: "stage5_explainability.py",
        6: "stage6_figures.py",
    }
    script = stage_scripts.get(stage_num)
    if not script:
        print(f"[ERROR] Unknown stage {stage_num}. Valid stages are 0 through 6.")
        sys.exit(1)

    script_path = os.path.join(REPO_ROOT, "code", script)
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    run_cmd([sys.executable, script_path], f"Stage {stage_num} Execution ({script})")


def run_full():
    print("=== EXECUTING FULL MASTER PIPELINE ===")
    print("Warning: Full hyperparameter tuning and repeated CV take approx 30 minutes on full data.")
    confirm = os.environ.get("AUTO_CONFIRM", "0")
    if confirm != "1":
        ans = input("Proceed with full pipeline? [y/N]: ")
        if ans.lower() not in ["y", "yes"]:
            print("Aborted.")
            sys.exit(0)
    run_cmd([sys.executable, os.path.join(REPO_ROOT, "code", "run_all.py")], "Full Master Pipeline Execution")


def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for CIC-Trap4Phish Audit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Quick check of the 13 required result artifacts")
    group.add_argument("--verify", action="store_true", help="Run full empirical verification suite (verify.py)")
    group.add_argument("--smoke", action="store_true", help="Run lightweight smoke tests using sample data")
    group.add_argument("--stage", type=int, choices=range(0, 7), help="Execute a specific stage (0 to 6)")
    group.add_argument("--full", action="store_true", help="Re-run entire experiment pipeline from scratch")

    args = parser.parse_args()

    if args.check:
        check_artifacts()
    elif args.verify:
        verify_repo()
    elif args.smoke:
        run_smoke_test()
    elif args.stage is not None:
        run_stage(args.stage)
    elif args.full:
        run_full()


if __name__ == "__main__":
    main()
