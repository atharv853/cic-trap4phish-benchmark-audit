import os
import sys
import json
import pandas as pd

TARGET_FILES = [
    "honest_leaderboard.csv",
    "audit_provenance.csv",
    "audit_shortcuts.csv",
    "l2_removed_features.csv",
    "twin_vs_nontwin.csv",
    "sensitivity_k.csv",
    "transfer_gap_significance.csv",
    "hypotheses.csv",
    "arm4_shap_share.csv",
    "lofo_matrix.csv",
    "ensemble_comparison.csv",
    "mapping_log.json",
    "html_error_analysis.md"
]

def validate_file(filepath):
    if not os.path.exists(filepath):
        return False, "File does not exist"
    if os.path.getsize(filepath) == 0:
        return False, "File is empty (0 bytes)"

    fname = os.path.basename(filepath)
    if fname.endswith(".csv"):
        try:
            df = pd.read_csv(filepath)
            if df.empty or len(df.columns) == 0:
                return False, "CSV is empty or has no columns"
        except Exception as e:
            return False, f"CSV parsing error: {e}"
    elif fname.endswith(".json"):
        try:
            with open(filepath, "r") as f:
                content = json.load(f)
                if content is None:
                    return False, "JSON content is null"
        except Exception as e:
            return False, f"JSON parsing error: {e}"
    elif fname.endswith(".md"):
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()
                if not content or content == "[FILE NOT FOUND]":
                    return False, "Markdown is empty or placeholder"
        except Exception as e:
            return False, f"Markdown reading error: {e}"

    return True, "Valid"

def check_all(search_dir):
    found_map = {}
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f in TARGET_FILES:
                full_path = os.path.join(root, f)
                valid, msg = validate_file(full_path)
                if valid:
                    found_map[f] = full_path

    complete_count = len(found_map)
    return complete_count, found_map

if __name__ == "__main__":
    default_dir = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.dirname(__file__)))
    search_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    count, found_map = check_all(search_dir)
    print(f"FOUND {count}/{len(TARGET_FILES)} FILES in {search_dir}")
    for tf in TARGET_FILES:
        if tf in found_map:
            print(f"[OK] {tf}")
        else:
            print(f"[MISSING] {tf}")

    if count == len(TARGET_FILES):
        sys.exit(0)
    else:
        sys.exit(1)
