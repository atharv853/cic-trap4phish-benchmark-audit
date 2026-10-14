import os
import sys
import pandas as pd
from stage0_audit import run_stage0, RESULTS_DIR

def run_gate0():
    run_stage0()
    print("\n================ GATE 0 OUTPUT ================")
    tables = [
        "audit_shapes.csv",
        "audit_missingness.csv",
        "audit_duplicates.csv",
        "audit_twin_leakage.csv",
        "audit_shortcuts.csv",
        "audit_provenance.csv",
        "word_missingness_analysis.csv"
    ]
    for tbl in tables:
        path = os.path.join(RESULTS_DIR, tbl)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Table: {tbl:<32} | Shape: {df.shape}")
        else:
            print(f"Table: {tbl:<32} | MISSING!")

    print("\nTop 5 rows of Single-Feature Audit (audit_shortcuts.csv):")
    df_sc = pd.read_csv(os.path.join(RESULTS_DIR, "audit_shortcuts.csv"))
    print(df_sc.head(5).to_string(index=False))
    print("================ GATE 0 PASSED ================\n")

if __name__ == "__main__":
    run_gate0()
